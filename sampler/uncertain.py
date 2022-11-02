from .base import BaseSampler
from utils import get_label_model, ABSTAIN
from scipy.stats import entropy
import numpy as np


class UncertaintySampler(BaseSampler):
    def __init__(self, train_data, labeller, label_model_type,
                 only_sample_conflict=True, only_sample_covered=True, **kwargs):
        super(UncertaintySampler, self).__init__(train_data, labeller, **kwargs)
        self.label_model = get_label_model(label_model_type)
        self.label_model.fit(dataset_train=train_data)
        self.only_sample_conflict = only_sample_conflict
        self.only_sample_covered = only_sample_covered

        if self.only_sample_conflict:
            unique_weak_labels = np.zeros(shape=(len(train_data), train_data.n_class))
            for c in range(train_data.n_class):
                act = np.any(self.weak_labels == c, axis=1)
                unique_weak_labels[:, c] = act

            diff_weak_labels = unique_weak_labels.sum(axis=1)  # how many different classes predicted
            self.candidate_indices = np.nonzero((diff_weak_labels > 1) & (~self.sampled))[0]
        elif self.only_sample_covered:
            covered_mask = np.any(self.weak_labels != ABSTAIN, axis=1)
            self.candidate_indices = np.nonzero(covered_mask & (~self.sampled))[0]
        else:
            self.candidate_indices = np.nonzero(~self.sampled)[0]

    def update_dataset(self, train_data):
        super(UncertaintySampler, self).update_dataset(train_data)
        self.label_model.fit(dataset_train=train_data)
        if self.only_sample_conflict:
            unique_weak_labels = np.zeros(shape=(len(train_data), train_data.n_class))
            for c in range(train_data.n_class):
                act = np.any(self.weak_labels == c, axis=1)
                unique_weak_labels[:, c] = act

            diff_weak_labels = unique_weak_labels.sum(axis=1)  # how many different classes predicted
            self.candidate_indices = np.nonzero((diff_weak_labels > 1) & (~self.sampled))[0]
        elif self.only_sample_covered:
            covered_mask = np.any(self.weak_labels != ABSTAIN, axis=1)
            self.candidate_indices = np.nonzero(covered_mask & (~self.sampled))[0]
        else:
            self.candidate_indices = np.nonzero(~self.sampled)[0]

    def sample_distinct(self, n=1):
        probs = self.label_model.predict_proba(self.train_data)
        uncertainty = entropy(probs, axis=1)
        candidate_uncertainty = uncertainty[self.candidate_indices]
        order = np.argsort(candidate_uncertainty)[::-1]
        n_sampled = 0
        i = 0
        indices = []
        while n_sampled < n:
            idx = self.candidate_indices[order[i]]
            i += 1
            if self.sampled[idx]:
                continue
            self.sampled[idx] = True
            indices.append(idx)
            n_sampled += 1

        labels = self.label_selected_indices(indices)
        return indices, labels

