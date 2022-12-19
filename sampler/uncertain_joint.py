from .base import BaseSampler
from scipy.stats import entropy
import numpy as np


class JointUncertaintySampler(BaseSampler):
    def sample_distinct(self, n=1):
        lm_probs = self.label_model.predict_proba(self.train_data)
        rm_probs = self.revision_model.predict_proba(self.train_data)
        lm_uncertainty = entropy(lm_probs, axis=1)
        rm_uncertainty = entropy(rm_probs, axis=1)
        uncertainty = lm_uncertainty * rm_uncertainty
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

