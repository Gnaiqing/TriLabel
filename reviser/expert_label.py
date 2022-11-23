from wrench.dataset.utils import check_weak_labels
from reviser.base import BaseReviser
import copy
from utils import ABSTAIN


class ExpertLabelReviser(BaseReviser):
    def train_revision_model(self, indices, labels, cost):
        self.sampled_indices = indices
        self.sampled_labels = labels
        return None

    def predict_labels(self, dataset, cost):
        return None

    def predict_proba(self, dataset):
        return None

    def get_revised_dataset(self, dataset_type, cost):
        if dataset_type == "train":
            dataset = self.train_data
            revised_dataset = copy.copy(dataset)
            revised_weak_labels = check_weak_labels(dataset)
            for lf_idx in range(dataset.n_lf):
                revised_weak_labels[self.sampled_indices, lf_idx] = self.sampled_labels

            revised_dataset.weak_labels = revised_weak_labels.tolist()

        elif dataset_type == "valid":
            revised_dataset = copy.copy(self.valid_data)
        else:
            raise ValueError(f"dataset type {dataset_type} not supported.")

        return revised_dataset

