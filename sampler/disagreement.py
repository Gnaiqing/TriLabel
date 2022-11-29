from .base import BaseSampler
from scipy.stats import entropy
import numpy as np


class DisagreementSampler(BaseSampler):
    def sample_distinct(self, n=1):
        lm_probs = self.label_model.predict_proba(self.train_data)
        rm_probs = self.revision_model.predict_proba(self.train_data)
        if rm_probs is not None:
            diff = entropy(lm_probs, rm_probs, axis=1)
        else:
            # if revision model can not predict proba, use the entropy of label model as uncertainty sampler
            diff = entropy(lm_probs, axis=1)

        diff = diff[self.candidate_indices]
        order = np.argsort(diff)[::-1]
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


