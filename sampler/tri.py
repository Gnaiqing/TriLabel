from .base import BaseSampler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import numpy as np


class TriSampler(BaseSampler):
    def __init__(self, train_data, labeller, label_model=None, revision_model=None, encoder=None, **kwargs):
        super(TriSampler, self).__init__(train_data, labeller, label_model, revision_model, encoder, **kwargs)
        if "lm_threshold" in kwargs:
            self.lm_threshold = kwargs["lm_threshold"]  # confidence threshold for pseudo label for initialization
        else:
            self.lm_threshold = 0.95  # default threshold

        lm_probs = label_model.predict_proba(train_data)
        lm_conf = np.max(lm_probs, axis=1)
        self.lm_preds = np.argmax(lm_probs, axis=1)
        self.pseudo_labeled = lm_conf > self.lm_threshold
        self.pseudo_labeled_indices = np.nonzero(self.pseudo_labeled)[0]
        if len(self.pseudo_labeled_indices) < 0.01 * len(self.train_data) and self.kwargs["init_method"] == "pl":
            self.kwargs["init_method"] = "pl+random"

        if self.kwargs["init_method"] in ["pl", "pl+random"]:
            valid_class_dist = kwargs["valid_class_dist"]
            pl_class_dist = np.zeros(train_data.n_class, dtype=float)
            ratio = np.zeros(train_data.n_class, dtype=float)
            for i in range(train_data.n_class):
                pl_class_dist[i] = np.sum(self.lm_preds[self.pseudo_labeled] == i) / len(self.pseudo_labeled_indices)
                ratio[i] = pl_class_dist[i] / valid_class_dist[i]

            c = np.argmin(ratio)
            n_c = np.sum(self.lm_preds[self.pseudo_labeled] == c)
            for i in range(train_data.n_class):
                subsample_size = int(n_c * valid_class_dist[i] / valid_class_dist[c])
                candidate_indices = np.nonzero((self.lm_preds == i) & self.pseudo_labeled)[0]
                subsample_indices = self.rng.choice(candidate_indices, subsample_size, replace=False)
                self.sampled[subsample_indices] = True
                self.sampled_labels[subsample_indices] = self.lm_preds[subsample_indices]

            self.candidate_indices = np.nonzero(~self.sampled)[0]

            if kwargs["init_method"] == "pl":
                indices, labels = self.get_sampled_points()
                self.revision_model.train_revision_model(indices, labels)
                self.initialized = True

        if "print_result" in kwargs:
            indices, labels = self.get_sampled_points()
            gt_labels = np.array(self.train_data.labels)[indices]
            pl_acc = accuracy_score(gt_labels, labels)
            print("Pseudo_labeled size: ", len(indices))
            print("Pseudo_labeled frac: ", len(indices) / len(train_data))
            print("Pseudo_labeled acc: ", pl_acc)
            if len(indices) > 0:
                init_label_freq = np.bincount(labels) / len(labels)
                print("Label distribution (init):", init_label_freq)
            population_class_freq = np.bincount(train_data.labels) / len(train_data)
            print("Label distribution (population)", population_class_freq)

        self.uncertain_type = kwargs["uncertain_type"]

    def get_n_sampled(self):
        # get used label budget. Do not count pseudo-labeled instances since they do not consume budget.
        indices, labels = self.get_sampled_points()
        budget_indices = np.setdiff1d(indices, self.pseudo_labeled_indices)
        return len(budget_indices)

    def sample_distinct(self, n=1):
        if not self.initialized:
            # perform random exploration. Only use pseudo-labels when sampled
            n_labeled = 0
            indices = []
            labels = []
            while n_labeled < n:
                idx = self.rng.choice(self.candidate_indices)
                if self.sampled[idx]:
                    continue

                if self.pseudo_labeled[idx]:
                    self.sampled[idx] = True
                    self.sampled_labels[idx] = self.lm_preds[idx]
                    label = self.lm_preds[idx]
                else:
                    n_labeled += 1
                    label = self.label_selected_indices([idx])[0]

                indices.append(idx)
                labels.append(label)

            return indices, labels
        else:
            if self.uncertain_type == "lm":
                probs = self.label_model.predict_proba(self.train_data)
            elif self.uncertain_type == "rm":
                probs = self.revision_model.predict_proba(self.train_data)
            else:
                lm_probs = self.label_model.predict_proba(self.train_data)
                rm_probs = self.revision_model.predict_proba(self.train_data)
                probs = lm_probs * rm_probs
                probs = probs / np.sum(probs, axis=1).reshape(-1, 1)

            uncertainty = entropy(probs, axis=1)
            candidate_uncertainty = uncertainty[self.candidate_indices]
            order = np.argsort(candidate_uncertainty)[::-1]
            n_labeled = 0
            i = 0
            indices = []
            labels = []
            while n_labeled < n:
                idx = self.candidate_indices[order[i]]
                i += 1
                if self.sampled[idx]:
                    continue

                if self.pseudo_labeled[idx]:
                    self.sampled[idx] = True
                    self.sampled_labels[idx] = self.lm_preds[idx]
                    label = self.lm_preds[idx]
                else:
                    n_labeled += 1
                    label = self.label_selected_indices([idx])[0]

                indices.append(idx)
                labels.append(label)

            labels = self.label_selected_indices(indices)
            return indices, labels

    def update_stats(self, train_data=None, label_model=None, revision_model=None):
        super(TriSampler, self).update_stats(train_data=train_data, label_model=label_model,
                                             revision_model=revision_model)
        if not self.initialized:
            self.initialized = True