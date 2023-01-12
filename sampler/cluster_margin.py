from .base import BaseSampler
from sklearn.cluster import AgglomerativeClustering
import numpy as np


class ClusterMarginSampler(BaseSampler):
    def __init__(self, train_data, labeller, label_model=None, revision_model=None, encoder=None, **kwargs):
        super(ClusterMarginSampler, self).__init__(train_data, labeller, label_model, revision_model, encoder, **kwargs)
        self.train_embedding = None
        self.batch_size = None
        self.cluster_labels = None
        self.n_clusters = None

    def sample_distinct(self, n=1):
        if self.revision_model.clf is None:
            self.batch_size = n
            indices = np.random.choice(self.candidate_indices, n, replace=False)  # random selection for first batch
            labels = self.label_selected_indices(indices)
            return indices, labels
        else:
            # compute margin scores for training data
            probs = self.revision_model.predict_proba(self.train_data)
            sorted_probs = np.sort(probs, axis=1)
            margin = sorted_probs[:,-1] - sorted_probs[:,-2]
            candidate_margin = margin[self.candidate_indices]
            order = np.argsort(candidate_margin)
            m = 10 * n  # candidate pool size
            candidates = self.candidate_indices[order[:m]]
            candidate_clusters = self.cluster_labels[candidates]
            cluster_size = np.zeros(self.n_clusters)
            cluster_map = {}
            cluster_sampled = np.zeros(self.n_clusters)
            for i in range(self.n_clusters):
                cluster_size[i] = np.sum(candidate_clusters == i).astype(int)
                cluster_map[i] = [idx for idx in candidates if self.cluster_labels[idx] == i]

            cluster_order = np.argsort(cluster_size)
            indices = []
            j = 0
            cur_cluster = cluster_order[j]
            while len(indices) < n:
                while cluster_sampled[cur_cluster] >= cluster_size[cur_cluster]:
                    j = (j + 1) % self.n_clusters
                    cur_cluster = cluster_order[j]

                idx = self.rng.choice(cluster_map[cur_cluster])
                indices.append(idx)
                cluster_sampled[cur_cluster] += 1
                cluster_map[cur_cluster].remove(idx)
                j = (j + 1) % self.n_clusters
                cur_cluster = cluster_order[j]

            labels = self.label_selected_indices(indices)
            return indices, labels

    def update_stats(self, train_data, label_model=None, revision_model=None):
        super(ClusterMarginSampler, self).update_stats(train_data, label_model=label_model, revision_model=revision_model)
        if self.train_embedding is None:
            self.train_embedding = self.revision_model._features
            clustering = AgglomerativeClustering(n_clusters=self.batch_size, linkage="average").fit(self.train_embedding)
            self.n_clusters = self.batch_size
            self.cluster_labels = clustering.labels_



