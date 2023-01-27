from .base import BaseSampler
from scipy.stats import entropy
import numpy as np


class UncertaintySampler(BaseSampler):
    def __init__(self, train_data, labeller, label_model=None, revision_model=None, encoder=None, **kwargs):
        super(UncertaintySampler, self).__init__(train_data, labeller, label_model, revision_model, encoder, **kwargs)
        self.initialized = False
        self.class_dist_est = None
        self.uncertain_type = kwargs["uncertain_type"]

    def sample_distinct(self, n=1):
        if not self.initialized:
            # perform random sampling in first batch
            indices = np.random.choice(self.candidate_indices, n, replace=False)  # random selection for first batch
            labels = self.label_selected_indices(indices)
            return indices, labels
        else:
            # compute margin scores for training data
            if self.uncertain_type == "lm":
                probs = self.label_model.predict_proba(self.train_data)
            elif self.uncertain_type == "rm":
                probs = self.revision_model.predict_proba(self.train_data)
            else:
                lm_probs = self.label_model.predict_proba(self.train_data)
                rm_probs = self.revision_model.predict_proba(self.train_data)
                probs = lm_probs * rm_probs / self.class_dist_est.reshape(1, -1)
                probs = probs / np.sum(probs, axis=1).reshape(-1, 1)

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

    def update_stats(self, train_data=None, label_model=None, revision_model=None):
        super(UncertaintySampler, self).update_stats(train_data=train_data, label_model=label_model, revision_model=revision_model)
        if not self.initialized:
            self.initialized = True
            indices, labels = self.get_sampled_points()
            self.class_dist_est = np.bincount(labels, minlength=self.train_data.n_class) / len(labels)
            self.class_dist_est = (self.class_dist_est + 1e-3) / np.sum(self.class_dist_est + 1e-3)