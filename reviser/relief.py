from wrench.dataset.utils import check_weak_labels
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
import numpy as np

from utils import ABSTAIN, plot_tsne, get_lf, get_revision_model
import copy
import torch


class LFReviser:
    def __init__(self, train_data, encoder, lf_class, revision_model_class,
                 acc_threshold=0.6, min_labeled_size=10, valid_data=None, seed=None):
        """
        Initialize LF Reviser
        :param train_data: dataset to revise
        :param encoder: trained encoder model for feature transformation
        :param lf_class: LF class for new label functions
        :param revision_model_class: classifier type used to revise LF
        :param acc_threshold: accuracy threshold for new LFs
        :param min_labeled_size: introduce new LF when sampled uncovered data reach that size for some class
        :param valid_data: validation dataset with golden labels
        """
        self.train_data = train_data
        self.valid_data = valid_data
        self.encoder = encoder
        self.lf_class = lf_class
        self.weak_labels = check_weak_labels(self.train_data)
        self.revision_model_class = revision_model_class
        self.acc_threshold = acc_threshold
        self.min_labeled_size = min_labeled_size
        self.seed = seed
        self.revision_models = {}
        self.append_lfs = []
        self.discard_lfs = []

    def update_encoder(self, encoder):
        self.encoder = encoder

    def update_dataset(self, train_data, valid_data=None):
        self.train_data = train_data
        self.valid_data = valid_data
        self.weak_labels = check_weak_labels(self.train_data)
        self.revision_models = {}
        self.append_lfs = []
        self.discard_lfs = []

    def get_overall_coverage(self):
        abstain_mask = np.all(self.weak_labels == ABSTAIN, axis=1)
        coverage = 1 - np.sum(abstain_mask) / len(self.train_data)
        return coverage

    def get_feature(self, dataset):
        if self.encoder is not None:
            return self.encoder(torch.tensor(dataset.features)).detach().cpu().numpy()
        else:
            return dataset.features

    def append_label_functions(self, indices, labels):
        """
        Append new LFs to improve coverage while keep LF accuracy above threshold
        :param indices:
        :param labels:
        :return:
        """
        abstain_mask = np.all(self.weak_labels == ABSTAIN, axis=1)
        abstain_indices = np.nonzero(abstain_mask)[0]
        labeled_abstain_mask = np.in1d(indices, abstain_indices)
        labeled_abstain_indices = np.array(indices)[labeled_abstain_mask]
        labeled_abstain_labels = np.array(labels)[labeled_abstain_mask]
        class_dist = np.bincount(labeled_abstain_labels)
        for c in range(len(class_dist)):
            if class_dist[c] > self.min_labeled_size:
                X = self.get_feature(self.train_data)[labeled_abstain_indices,:]
                clf = get_lf(self.lf_class, self.seed)
                clf.fit(X, labeled_abstain_labels)
                if self.valid_data is not None:
                    X_val = self.get_feature(self.valid_data)
                    y_val = np.array(self.valid_data.labels) == c
                    y_pred = clf.predict(X_val)
                    lf_val_acc = precision_score(y_val, y_pred)
                    if lf_val_acc >= self.acc_threshold:
                        print(f"Append new LF for class {c}")
                        self.append_lfs.append((clf, c))
                else:
                    # use covered labeled data as valid set
                    labeled_covered_indices = np.array(indices)[~labeled_abstain_mask]
                    labeled_covered_labels = np.array(labels)[~labeled_abstain_mask]
                    X_val = self.get_feature(self.train_data)[labeled_covered_indices]
                    y_val = labeled_covered_labels == c
                    y_pred = clf.predict(X_val)
                    lf_val_acc = precision_score(y_val, y_pred)
                    if lf_val_acc >= self.acc_threshold:
                        print(f"Append new LF for class {c}")
                        self.append_lfs.append((clf, c))

    def revise_label_functions(self, indices, labels):
        # train revision models to revise existing LFs
        for lf_idx in range(self.train_data.n_lf):
            active_mask_l = self.weak_labels[indices, lf_idx] != ABSTAIN
            active_lf_preds_l = self.weak_labels[indices, lf_idx][active_mask_l]
            active_indices_l = np.array(indices)[active_mask_l]
            active_labels_l = np.array(labels)[active_mask_l]
            X_act_l = self.get_feature(self.train_data)[active_indices_l, :]
            y_act_l = (active_lf_preds_l == active_labels_l).astype(int)
            clf = get_revision_model(self.revision_model_class, seed=self.seed)
            n_pos = np.sum(y_act_l == 1)
            n_neg = np.sum(y_act_l == 0)
            # first filter out LF with accuracy < 0.5
            if self.valid_data is not None:
                val_active_mask = check_weak_labels(self.valid_data)[:, lf_idx] != ABSTAIN
                y_val = np.array(self.valid_data.labels)[val_active_mask] == \
                        np.array(self.valid_data.weak_labels)[val_active_mask, lf_idx]
                lf_prev_acc_hat = np.mean(y_val)
                if len(y_val) > self.min_labeled_size and lf_prev_acc_hat < 0.5:
                    print(f"Discard LF {lf_idx} for low accuracy.")
                    self.discard_lfs.append(lf_idx)
                    continue
            else:
                if len(y_act_l) > self.min_labeled_size and n_pos <= n_neg:
                    print(f"Discard LF {lf_idx} for low accuracy.")
                    self.discard_lfs.append(lf_idx)
                    continue

            if min(n_pos, n_neg) >= self.min_labeled_size:
                # train revision model for that LF
                if self.valid_data is not None:
                    clf = clf.fit(X_act_l, y_act_l)
                    val_active_mask = check_weak_labels(self.valid_data)[:, lf_idx] != ABSTAIN
                    X_val = self.get_feature(self.valid_data)[val_active_mask, :]
                    y_val = np.array(self.valid_data.labels)[val_active_mask] == \
                            np.array(self.valid_data.weak_labels)[val_active_mask, lf_idx]
                    y_pred = clf.predict(X_val)

                else:
                    # split 20% data as valid set
                    X_train, X_val, y_train, y_val = train_test_split(X_act_l, y_act_l, test_size=0.2, random_state=self.seed)
                    clf = clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_val)

                lf_acc_hat = precision_score(y_val, y_pred)  # use accuracy after revision on valid set
                active_mask = self.weak_labels[:, lf_idx] != ABSTAIN
                X_act = self.get_feature(self.train_data)[active_mask, :]
                y_pred_act = clf.predict(X_act)
                lf_cov = np.sum(y_pred_act) / len(self.train_data)
                lf_prev_acc_hat = np.mean(y_val)
                lf_prev_cov = len(X_act) / len(self.train_data)
                if (2 * lf_acc_hat - 1) * lf_cov > (2 * lf_prev_acc_hat - 1) * lf_prev_cov:
                    print(f"Create revision model for LF {lf_idx}")
                    self.revision_models[lf_idx] = clf

    def get_revised_dataset(self, dataset, indices=None, labels=None):
        """
        Revise dataset
        :param dataset: dataset to revise. If set to None, revise train data
        :param indices: indices of sampled train data
        :param labels: labels of sampled train data
        :return:
        """
        use_train_set = False
        if dataset is None:
            use_train_set = True
            dataset = self.train_data
        revised_weak_labels = np.array(dataset.weak_labels)
        revised_dataset = copy.copy(dataset)

        X = self.get_feature(dataset)
        for lf_idx in range(dataset.n_lf):
            if lf_idx in self.revision_models:
                clf = self.revision_models[lf_idx]
                y_pred = clf.predict(X)
                revised_weak_labels[y_pred == 0, lf_idx] = ABSTAIN
            elif use_train_set and indices is not None:
                lf_output = revised_weak_labels[indices, lf_idx]
                diff_indices = np.array(indices)[lf_output != np.array(labels)]
                revised_weak_labels[diff_indices, lf_idx] = ABSTAIN

        for (clf, c) in self.append_lfs:
            pred = clf.predict(X)
            new_LF_out = np.where(pred == 1, c, ABSTAIN).reshape(-1, 1)
            revised_weak_labels = np.concatenate((revised_weak_labels, new_LF_out), axis=1)

        if len(self.discard_lfs) > 0:
            revised_weak_labels = np.delete(revised_weak_labels, self.discard_lfs, axis=1)

        revised_dataset.weak_labels = revised_weak_labels.tolist()
        revised_dataset.n_lf = revised_weak_labels.shape[1]
        return revised_dataset

