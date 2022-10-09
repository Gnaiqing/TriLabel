from .base import BaseSampler
from utils import ABSTAIN
import numpy as np

class LFCovSampler(BaseSampler):
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
            active_indices = np.nonzero(active_mask)[0]

        else:
            active_LF = np.array(active_LF)
            active_mask = np.any(self.weak_labels[:, active_LF] != ABSTAIN, axis=1)
            active_indices = np.nonzero(active_mask)[0]

        while n_sampled < n:
            idx = self.rng.choice(active_indices)
            if self.sampled[idx]:
                continue
            self.sampled[idx] = True
            indices.append(idx)
            n_sampled += 1

        labels = self.label_selected_indices(indices)
        return indices, labels