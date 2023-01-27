from .base import BaseSampler
from sklearn.cluster import kmeans_plusplus
import numpy as np


class BadgeSampler(BaseSampler):
    def sample_distinct(self, n=1):
        if not self.initialized:
            indices = np.random.choice(self.candidate_indices, n, replace=False)  # random selection for first batch
            labels = self.label_selected_indices(indices)
            return indices, labels
        else:
            train_embedding = self.revision_model.get_pseudo_grads(self.train_data)
            unlabeled_train_embedding = train_embedding[self.candidate_indices, :]
            # use K-means++ to fetch data
            centers, indices = kmeans_plusplus(unlabeled_train_embedding, n_clusters=n)
            indices = self.candidate_indices[indices]
            labels = self.label_selected_indices(indices)
            return indices, labels

    def update_stats(self, train_data=None, label_model=None, revision_model=None):
        super(BadgeSampler, self).update_stats(train_data=train_data, label_model=label_model,
                                                     revision_model=revision_model)
        self.initialized = True



