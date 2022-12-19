from reviser.base import BaseReviser
import numpy as np


class ExpertLabelReviser(BaseReviser):
    def train_revision_model(self, indices, labels):
        self.sampled_indices = indices
        self.sampled_labels = labels

    def predict_proba(self, dataset):
        assert len(dataset) == len(self.train_data)
        proba = np.zeros((len(dataset), dataset.n_class), dtype=float)
        proba[self.sampled_indices, self.sampled_labels] = 1.0
        non_labeled_indices = np.setdiff1d(np.arange(len(dataset)), self.sampled_indices)
        proba[non_labeled_indices, :] = 1/dataset.n_class
        return proba


