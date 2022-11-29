from .base import BaseSampler
from scipy.stats import entropy
import numpy as np


class RmUncertaintySampler(BaseSampler):
    def sample_distinct(self, n=1):
        probs = self.revision_model.predict_proba(self.train_data)
        if probs is None:
            order = np.random.permutation(len(self.candidate_indices))
        else:
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

