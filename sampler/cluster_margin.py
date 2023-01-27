from .base import BaseSampler
from sklearn.cluster import AgglomerativeClustering
import numpy as np


class ClusterMarginSampler(BaseSampler):
    def __init__(self, train_data, labeller, label_model=None, revision_model=None, encoder=None, **kwargs):
        super(ClusterMarginSampler, self).__init__(train_data, labeller, label_model, revision_model, encoder, **kwargs)
        self.initialized = False
        self.train_embedding = None
        self.cluster_labels = None
        if "n_clusters" in kwargs:
            self.n_clusters = kwargs["n_clusters"]
        else:
            self.n_clusters = 20  # default value
        self.class_dist_est = None
        self.uncertain_type = kwargs["uncertain_type"]

    def sample_distinct(self, n=1):
        if not self.initialized:
            # perform random sampling in first batch
            indices = np.random.choice(self.candidate_indices, n, replace=False)  # random selection for first batch
            labels = self.label_selected_indices(indices)
            return indices, labels
        else:
            if self.uncertain_type == "lm":
                probs = self.label_model.predict_proba(self.train_data)
            elif self.uncertain_type == "rm":
                probs = self.revision_model.predict_proba(self.train_data)
            else:
                lm_probs = self.label_model.predict_proba(self.train_data)
                rm_probs = self.revision_model.predict_proba(self.train_data)
                probs = lm_probs * rm_probs / self.class_dist_est.reshape(1,-1)
                probs = probs / np.sum(probs, axis=1).reshape(-1,1)

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

    def update_stats(self, train_data=None, label_model=None, revision_model=None):
        super(ClusterMarginSampler, self).update_stats(train_data=train_data, label_model=label_model, revision_model=revision_model)
        if not self.initialized:
            self.initialized = True
            self.train_embedding = self.revision_model._features
            clustering = AgglomerativeClustering(n_clusters=self.n_clusters, linkage="average").fit(self.train_embedding)
            self.cluster_labels = clustering.labels_
            indices, labels = self.get_sampled_points()
            self.class_dist_est = np.bincount(labels, minlength=self.train_data.n_class) / len(labels)
            self.class_dist_est = (self.class_dist_est + 1e-3) / np.sum(self.class_dist_est + 1e-3)





