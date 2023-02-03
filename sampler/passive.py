from .base import BaseSampler


class PassiveSampler(BaseSampler):
    def sample_distinct(self, n=1):
        indices = self.rng.choice(self.candidate_indices, n, replace=False)
        labels = self.label_selected_indices(indices)
        return indices, labels

