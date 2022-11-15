from .base import BaseSampler
from utils import ABSTAIN
from scipy.stats import entropy
import numpy as np


class MaxKLSampler(BaseSampler):
    def sample_distinct(self, n=1):
        unique_combs, unique_idx, unique_inverse = np.unique(
            self.weak_labels, return_index=True, return_inverse=True, axis=0)
        all_abstain = np.all(self.weak_labels == ABSTAIN, axis=1)
        is_in_pool = (~self.sampled) & (~all_abstain)
        valid_buckets = np.unique(unique_inverse[is_in_pool])
        is_valid_bucket = np.array([
          True if i in valid_buckets else False for i in range(len(unique_idx))])
        probs = self.label_model.predict_proba(self.train_data)
        bucket_probs = probs[unique_idx]
        # Label model distributions
        lm_posteriors = (bucket_probs + 1e-5) / (bucket_probs + 1e-5).sum(axis=1).reshape(-1,1)
        # Sample distributions
        sample_posteriors = np.zeros(lm_posteriors.shape)
        for i in range(len(sample_posteriors)):
            sampled_bucket_items = (unique_inverse == i) & self.sampled
            sampled_labels = self.sampled_labels[sampled_bucket_items]
            if len(sampled_labels) > 0:
                dist = np.bincount(sampled_labels) / len(sampled_labels)
                if len(dist) < self.train_data.n_class:
                    n_pad = self.train_data.n_class - len(dist)
                    dist = np.pad(dist, (0, n_pad), "constant")

                sample_posteriors[i,:] = dist
            else:
                dist = np.ones(self.train_data.n_class) / self.train_data.n_class  # assume uniform dist
                sample_posteriors[i,:] = dist

        rel_entropy = entropy(lm_posteriors, sample_posteriors, axis=1)
        bucket_order = np.argsort(rel_entropy)[::-1]
        indices = []
        n_sampled = 0
        for bucket_idx in bucket_order:
            if is_valid_bucket[bucket_idx]:
                candidate_indices = np.nonzero((unique_inverse == bucket_idx) & (~self.sampled))[0]
                n_remain = len(candidate_indices)
                if n_sampled + n_remain <= n:
                    indices += candidate_indices.tolist()
                    n_sampled += n_remain

                else:
                    selected_indices = self.rng.choice(candidate_indices, n-n_sampled, replace=False)
                    indices += selected_indices.tolist()
                    n_sampled = n

                if n_sampled == n:
                    break
        labels = self.label_selected_indices(indices)
        return indices, labels







