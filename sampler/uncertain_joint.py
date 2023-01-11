from .base import BaseSampler
from scipy.stats import entropy
import numpy as np


class JointUncertaintySampler(BaseSampler):
    def __init__(self, train_data, labeller, label_model=None, revision_model=None, encoder=None, **kwargs):
        super(JointUncertaintySampler, self).__init__(train_data, labeller, label_model, revision_model, encoder, **kwargs)
        self.epsilon = 0.9

    def sample_distinct(self, n=1):
        lm_probs = self.label_model.predict_proba(self.train_data)
        rm_probs = self.revision_model.predict_proba(self.train_data)
        probs = lm_probs * rm_probs
        probs = probs / (np.sum(probs, axis=1).reshape(-1,1))
        uncertainty = entropy(probs, axis=1)
        candidate_uncertainty = uncertainty[self.candidate_indices]
        order = np.argsort(candidate_uncertainty)[::-1]

        n_sampled = 0
        i = 0
        indices = []
        while n_sampled < n:
            if self.rng.random() < self.epsilon:
                # exploration
                idx = self.rng.choice(self.candidate_indices)
            else:
                # exploitation
                idx = self.candidate_indices[order[i]]
                i += 1

            if self.sampled[idx]:
                continue
            self.sampled[idx] = True
            indices.append(idx)
            n_sampled += 1
            self.epsilon = self.epsilon * 0.99

        labels = self.label_selected_indices(indices)
        return indices, labels

