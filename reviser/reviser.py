from wrench.dataset.utils import check_weak_labels
import numpy as np
from torch.utils.data import TensorDataset
from utils import ABSTAIN
from reviser.models import MLPNet, DropOutNet, MLPTempNet
from reviser.trainers import NeuralNetworkTrainer
import copy
import torch


class LFReviser:
    def __init__(self,
                 train_data,
                 encoder,
                 revision_model,
                 valid_data=None,
                 seed=None):
        """
        Initialize LF Reviser
        :param train_data: dataset to revise
        :param encoder: trained encoder model for feature transformation
        :param revision_model: model used to revise LF. One of ["mlp", "dropout", "mlp-temp"]
        :param valid_data: validation dataset with golden labels
        :param seed: seed for revision models
        :param kwargs: other arguments for revision model
        """
        self.train_data = train_data
        self.valid_data = valid_data
        self.encoder = encoder
        self.train_rep = self.get_feature(self.train_data)
        self.revision_model_type = revision_model
        if self.revision_model_type == "mlp":
            self.clf = MLPNet(input_dim=self.train_rep.shape[1], output_dim=self.train_data.n_class)
            self.trainer = NeuralNetworkTrainer(self.clf)
        elif self.revision_model_type == "mlp-temp":
            self.clf = MLPTempNet(input_dim=self.train_rep.shape[1], output_dim=self.train_data.n_class)
            self.trainer = NeuralNetworkTrainer(self.clf)
        elif self.revision_model_type == "dropout":
            self.clf = DropOutNet(input_dim=self.train_rep.shape[1], output_dim=self.train_data.n_class)
            self.trainer = NeuralNetworkTrainer(self.clf)
        elif self.revision_model_type == "expert-label":
            self.clf = None  # only expert labeled data will be used.
            self.trainer = None

        self.seed = seed
        self.sampled_indices = None
        self.sampled_labels = None

    def update_encoder(self, encoder):
        self.encoder = encoder

    def update_dataset(self, revised_train_data, revised_valid_data=None):
        self.train_data = revised_train_data
        self.valid_data = revised_valid_data

    def get_feature(self, dataset):
        if self.encoder is not None:
            return self.encoder(torch.tensor(dataset.features)).detach().cpu().numpy()
        else:
            return dataset.features

    def train_revision_model(self, indices, labels):
        # train the revision model which can abstain on some data points
        self.sampled_indices = indices
        self.sampled_labels = labels

        X_sampled = self.get_feature(self.train_data)[indices, :]
        y_sampled = labels
        training_dataset = TensorDataset(torch.tensor(X_sampled), torch.tensor(y_sampled))
        if self.valid_data is not None:
            X_eval = self.get_feature(self.valid_data)
            y_eval = self.valid_data.labels
            eval_dataset = TensorDataset(torch.tensor(X_eval), torch.tensor(y_eval))
        else:
            eval_dataset = None
        if self.trainer is not None:
            self.trainer.train_model_with_dataloader(training_dataset, eval_dataset)
        if self.revision_model_type == "mlp-temp":
            self.trainer.model.temp_scale(eval_dataset)

    def predict_labels(self, dataset_type, cost):
        """
        Use the
        """
        if dataset_type == "train":
            dataset = self.train_data
        elif dataset_type == "valid":
            dataset = self.valid_data
        else:
            raise ValueError(f"dataset type {dataset_type} not supported.")

        clf = self.clf
        X = self.get_feature(dataset)
        if clf is not None:
            threshold = 1 - cost  # optimal threshold based on Chow's rule
            y_pred = clf.predict(X)
            y_probs = clf.predict_proba(X).max(axis=1)
            y_pred[y_probs < threshold] = ABSTAIN
        else:
            # use expert labels only
            y_pred = np.repeat(ABSTAIN, len(dataset))
            if dataset_type == "train":
                y_pred[self.sampled_indices] = self.sampled_labels

        return y_pred

    def predict_proba(self, dataset_type):
        assert self.clf is not None
        if dataset_type == "train":
            dataset = self.train_data
        elif dataset_type == "valid":
            dataset = self.valid_data
        else:
            raise ValueError(f"dataset type {dataset_type} not supported.")

        X = self.get_feature(dataset)
        probs = self.clf.predict_proba(X)
        return probs

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

