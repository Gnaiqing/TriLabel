from .base import BaseSampler
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class CoreSetSampler(BaseSampler):
    def __init__(self, train_data, labeller, label_model=None, revision_model=None, encoder=None, **kwargs):
        super(CoreSetSampler, self).__init__(train_data, labeller, label_model, revision_model, encoder, **kwargs)
        self.train_embedding = None

    def sample_distinct(self, n=1):
        if self.revision_model.clf is None:
            indices = np.random.choice(self.candidate_indices, n, replace=False)  # random selection for first batch
            labels = self.label_selected_indices(indices)
            return indices, labels
        else:
            labeled_indices, _ = self.get_sampled_points()
            labeled_embeddings = self.train_embedding[labeled_indices,:]
            unlabeled_embedding = self.train_embedding[self.candidate_indices,:]
            indices = []
            while len(indices) < n:
                dist = euclidean_distances(unlabeled_embedding, labeled_embeddings)
                dist = np.min(dist, axis=1)
                idx = self.candidate_indices[np.argmax(dist)]
                indices.append(idx)
                self.sampled[idx] = True
                labeled_embeddings = np.vstack((labeled_embeddings, self.train_embedding[idx,:].reshape(1,-1)))
                self.candidate_indices = np.nonzero(~self.sampled)[0]
                unlabeled_embedding = self.train_embedding[self.candidate_indices, :]

            labels = self.label_selected_indices(indices)
            return indices, labels

    def update_stats(self, train_data, label_model=None, revision_model=None):
        super(CoreSetSampler, self).update_stats(train_data, label_model=label_model, revision_model=revision_model)
        if self.train_embedding is None:
            self.train_embedding = self.revision_model._features