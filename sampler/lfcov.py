from .base import BaseSampler
from utils import ABSTAIN
import numpy as np


class LFCovSampler(BaseSampler):
    def get_n_unsampled(self, active_LF=None):
        if active_LF is None:
            active_mask = np.all(self.weak_labels == ABSTAIN, axis=1)
            active_mask = active_mask & (~self.sampled)  # filter out sampled points

        else:
            active_LF = np.array(active_LF)
            active_mask = np.any(self.weak_labels[:, active_LF] != ABSTAIN, axis=1)
            active_mask = active_mask & (~self.sampled)  # filter out sampled points

        n_unsampled = np.sum(active_mask)
        return n_unsampled

    """
    Sample data points based on LF coverage
    """
    def sample_distinct(self, n=1, active_LF=None):
        """
        Sample data points based on LF coverage. If active_LF set to None,
        it selects points where no LF get activated. Otherwise, it select points
        where one of active_LF get activated.
        :param n:
        :param active_LFs: list of coverage to include
        :return:
        """
        n_sampled = 0
        indices = []
        if active_LF is None:
            active_mask = np.all(self.weak_labels == ABSTAIN, axis=1)
            active_mask = active_mask & (~self.sampled) # filter out sampled points
            active_indices = np.nonzero(active_mask)[0]

        else:
            active_LF = np.array(active_LF)
            active_mask = np.any(self.weak_labels[:, active_LF] != ABSTAIN, axis=1)
            active_mask = active_mask & (~self.sampled) # filter out sampled points
            active_indices = np.nonzero(active_mask)[0]

        n_unsampled = np.sum(active_mask)

        if n > n_unsampled:
            raise ValueError(f"Sample size {n} Exceed remaining data size {n_unsampled}.")

        while n_sampled < n:
            idx = self.rng.choice(active_indices)
            if self.sampled[idx]:
                continue
            self.sampled[idx] = True
            indices.append(idx)
            n_sampled += 1

        labels = self.label_selected_indices(indices)
        return indices, labels