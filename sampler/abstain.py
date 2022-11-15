from .base import BaseSampler
import numpy as np
from utils import ABSTAIN


"""
Abstain sampler: sample points where less LFs get activated
"""
class AbstainSampler(BaseSampler):
    def sample_distinct(self, n=1):
        active_counts = np.sum(self.weak_labels != ABSTAIN, axis=1)[self.candidate_indices]
        order = np.argsort(active_counts)
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