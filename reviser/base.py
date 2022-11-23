from wrench.dataset.utils import check_weak_labels
from utils import ABSTAIN
import copy
import torch


class BaseReviser:
    def __init__(self,
                 train_data,
                 encoder,
                 device="cpu",
                 valid_data=None,
                 seed=None):
        """
        Initialize LF Reviser
        :param train_data: dataset to revise
        :param encoder: trained encoder model for feature transformation
        :param valid_data: validation dataset with golden labels
        :param seed: seed for revision models
        :param kwargs: other arguments for revision model
        """
        self.train_data = train_data
        self.valid_data = valid_data
        self.encoder = encoder
        self.train_rep = self.get_feature(self.train_data)
        self.device = device
        self.seed = seed
        self.clf = None
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

    def train_revision_model(self, indices, labels, cost):
        """
        Train a revision model using sampled training data
        """
        raise NotImplementedError

    def predict_labels(self, dataset, cost):
        """
        Predict labels for dataset with abstention
        :param dataset: dataset to predict
        :param cost: cost for wrong prediction
        """
        raise NotImplementedError

    def predict_proba(self, dataset):
        """
        Predict posterior class distribution of dataset. This method only apply for confidence-based revision model.
        :param dataset: dataset to predict
        """
        raise NotImplementedError

    def get_revised_dataset(self, dataset_type, cost):
        """
        Revise dataset
        :param dataset_type: "train" or "valid"
        :param cost: cost for wrong classification
        :return: revised dataset
        """
        if dataset_type == "train":
            dataset = self.train_data
        elif dataset_type == "valid":
            dataset = self.valid_data
        else:
            raise ValueError(f"dataset type {dataset_type} not supported.")

        revised_dataset = copy.copy(dataset)
        revised_weak_labels = check_weak_labels(dataset)
        y_pred = self.predict_labels(dataset, cost)
        # correct all weak labels using predicted labels
        for lf_idx in range(dataset.n_lf):
            correct_mask = y_pred != ABSTAIN
            revised_weak_labels[correct_mask, lf_idx] = y_pred[correct_mask]

        revised_dataset.weak_labels = revised_weak_labels.tolist()
        return revised_dataset