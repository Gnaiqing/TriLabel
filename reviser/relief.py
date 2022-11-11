from wrench.dataset.utils import check_weak_labels
import numpy as np
from utils import ABSTAIN, get_revision_model
import copy
import torch


class LFReviser:
    def __init__(self,
                 train_data,
                 encoder,
                 revision_model_class,
                 min_sample_size=20,
                 valid_data=None,
                 concensus_criterion=None,
                 revise_threshold=0.7,
                 seed=None,
                 **kwargs):
        """
        Initialize LF Reviser
        :param train_data: dataset to revise
        :param encoder: trained encoder model for feature transformation
        :param revision_model_class: classifier type used to revise LF ["logistic", "ensemble", "random-forest"]
        :param min_sample_size: minimum sample size to train a revision classifier
        :param valid_data: validation dataset with golden labels
        :param concensus_criterion: criterion for revision if multiple classifiers are used (for ensemble model)
        :param revise_threshold: classifier would revise LF when it predict P(y|x)>threshold (for single model)
        :param seed: seed for revision models
        :param kwargs: other arguments for revision model
        """
        self.train_data = train_data
        self.valid_data = valid_data
        self.encoder = encoder
        self.min_sample_size = min_sample_size
        self.revision_model_class = revision_model_class
        self.concensus_criterion = concensus_criterion
        self.revise_threshold = revise_threshold
        self.seed = seed
        self.kwargs = kwargs
        self.clf = None  # model for revision. If set to None, only expert labeled data will be used.
        self.sampled_indices = None
        self.sampled_labels = None

    def update_encoder(self, encoder):
        self.encoder = encoder

    def get_feature(self, dataset):
        if self.encoder is not None:
            return self.encoder(torch.tensor(dataset.features)).detach().cpu().numpy()
        else:
            return dataset.features

    def revise_label_functions(self, indices, labels):
        # train classifier (or ensemble of classifier) to predict P(Y|X)
        self.sampled_indices = indices
        self.sampled_labels = labels
        clf = get_revision_model(self.revision_model_class, seed=self.seed, **self.kwargs)
        X_sampled = self.get_feature(self.train_data)[indices, :]
        y_sampled = labels
        if clf is not None:
            clf.fit(X_sampled, y_sampled)
            self.clf = clf

    def predict_labels(self, dataset_type):
        if dataset_type == "train":
            dataset = self.train_data
        elif dataset_type == "valid":
            dataset = self.valid_data
        else:
            raise ValueError(f"dataset type {dataset_type} not supported.")

        clf = self.clf
        X = self.get_feature(dataset)
        if self.revision_model_class == "voting":
            # ensemble model
            ensemble_pred = clf.transform(X)
            committee_size = ensemble_pred.shape[1]
            y_pred = clf.predict(X)
            n_agree = np.sum(ensemble_pred == y_pred.reshape(-1, 1), axis=1)
            if self.concensus_criterion == "majority":
                y_pred[n_agree <= committee_size // 2] = ABSTAIN
            elif self.concensus_criterion == "all":
                y_pred[n_agree < committee_size] = ABSTAIN
        elif clf is not None:
            # single model
            y_pred = clf.predict(X)
            y_probs = clf.predict_proba(X).max(axis=1)
            y_pred[y_probs < self.revise_threshold] = ABSTAIN
        else:
            # use expert labels only
            y_pred = np.repeat(ABSTAIN, len(dataset))
            if dataset_type == "train":
                y_pred[self.sampled_indices] = self.sampled_labels

        return y_pred

    def get_revised_dataset(self, dataset_type, y_pred, revise_LF_method):
        """
        Revise dataset
        :param dataset_type: "train" or "valid"
        :param y_pred: predicted labels by revision model
        :param revise_LF_method: "correct" or "mute"
        :return:
        """
        if dataset_type == "train":
            dataset = self.train_data
        elif dataset_type == "valid":
            dataset = self.valid_data
        else:
            raise ValueError(f"dataset type {dataset_type} not supported.")

        revised_dataset = copy.copy(dataset)
        revised_weak_labels = check_weak_labels(dataset)
        if revise_LF_method == "mute":
            for lf_idx in range(dataset.n_lf):
                mute_mask = (revised_weak_labels[:, lf_idx] != y_pred) & (y_pred != ABSTAIN)
                revised_weak_labels[mute_mask, lf_idx] = ABSTAIN
        elif revise_LF_method == "correct":
            for lf_idx in range(dataset.n_lf):
                correct_mask = y_pred != ABSTAIN
                revised_weak_labels[correct_mask, lf_idx] = y_pred[correct_mask]
        else:
            raise ValueError(f"Revise LF method {revise_LF_method} not supported.")

        revised_dataset.weak_labels = revised_weak_labels.tolist()
        return revised_dataset

