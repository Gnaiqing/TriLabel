from wrench.dataset.utils import check_weak_labels
from sklearn.metrics import precision_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

from utils import ABSTAIN, get_lf, get_revision_model
import copy
import torch


class LFReviser:
    def __init__(self, train_data, encoder, lf_class, revision_model_class, only_append_uncovered=True,
                 acc_threshold=0.6, min_labeled_size=1, valid_data=None, seed=None):
        """
        Initialize LF Reviser
        :param train_data: dataset to revise
        :param encoder: trained encoder model for feature transformation
        :param lf_class: LF class for new label functions
        :param revision_model_class: classifier type used to revise LF
        :param only_append_uncovered: new added LF will only get activated on uncovered data
        :param acc_threshold: accuracy threshold for new LFs
        :param min_labeled_size: introduce new LF when sampled uncovered data reach that size for some class
        :param valid_data: validation dataset with golden labels
        """
        self.train_data = train_data
        self.valid_data = valid_data
        self.encoder = encoder
        self.lf_class = lf_class
        self.only_append_uncovered = only_append_uncovered
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
        # update training set and valid set
        self.train_data = train_data
        self.valid_data = valid_data
        self.weak_labels = check_weak_labels(self.train_data)
        self.append_lfs = []
        self.discard_lfs = []
        self.revision_models = {}

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
        # generate new LFs based on not covered data
        abstain_mask = np.all(self.weak_labels == ABSTAIN, axis=1)
        abstain_indices = np.nonzero(abstain_mask)[0]
        labeled_abstain_mask = np.in1d(indices, abstain_indices)
        labeled_abstain_indices = np.array(indices)[labeled_abstain_mask]
        labeled_abstain_labels = np.array(labels)[labeled_abstain_mask]
        class_dist = np.bincount(labeled_abstain_labels)
        for c in range(len(class_dist)):
            if class_dist[c] > self.min_labeled_size:
                X = self.get_feature(self.train_data)[labeled_abstain_indices,:]
                y = (labeled_abstain_labels == c).astype(int)
                clf = get_lf(self.lf_class, self.seed)
                clf.fit(X, y)
                if self.valid_data is not None:
                    if self.only_append_uncovered:
                        uncovered_mask = np.all(np.array(self.valid_data.weak_labels) == ABSTAIN, axis=1)
                        X_val = self.get_feature(self.valid_data)[uncovered_mask,:]
                        y_val = np.array(self.valid_data.labels)[uncovered_mask] == c
                    else:
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
            if len(active_labels_l) == 0:
                continue
            X_act_l = self.get_feature(self.train_data)[active_indices_l, :]
            # y_act_l = (active_lf_preds_l == active_labels_l).astype(int)
            clf = get_revision_model(self.revision_model_class, seed=self.seed)
            n_pos = np.sum(active_labels_l == active_lf_preds_l)
            n_neg = np.sum(active_labels_l != active_lf_preds_l)
            # first filter out LF with accuracy < 0.5
            lf_prev_acc_hat = n_pos / (n_pos + n_neg)
            if lf_prev_acc_hat < 0.5 and (n_pos + n_neg) >= 20:
                print(f"Discard LF {lf_idx} for low accuracy.")
                self.discard_lfs.append(lf_idx)
                continue

            if np.min(active_labels_l) != np.max(active_labels_l):
                # train revision model for that LF
                if self.valid_data is not None:
                    clf = clf.fit(X_act_l, active_labels_l)
                    y_act_l_pred = clf.predict(X_act_l)
                    train_acc = accuracy_score(active_labels_l, y_act_l_pred)
                    # print(f"Revision model accuracy for LF {lf_idx} on labeled set: {train_acc:.2f}")
                    lf_preds_val = check_weak_labels(self.valid_data)[:, lf_idx]
                    val_active_mask = lf_preds_val != ABSTAIN
                    X_act_val = self.get_feature(self.valid_data)[val_active_mask, :]
                    y_act_val = np.array(self.valid_data.labels)[val_active_mask]
                    y_act_val_pred = clf.predict(X_act_val)  # class prediction

                else:
                    # split 20% data as valid set
                    X_act_train, X_act_val, y_act_train, y_act_val = train_test_split(
                        X_act_l, active_labels_l, test_size=0.2, random_state=self.seed)
                    clf = clf.fit(X_act_train, y_act_train)
                    y_act_val_pred = clf.predict(X_act_val)

                # keep prediction that consistent with classifier output
                revised_lf_preds = lf_preds_val[lf_preds_val != ABSTAIN] # active LF preds on valid set
                lf_prev_acc_hat = np.mean(revised_lf_preds == y_act_val)
                mute_mask = y_act_val_pred != revised_lf_preds # mute LF preds different from model preds
                revised_lf_preds[mute_mask] = ABSTAIN
                lf_revised_acc_hat = np.sum(revised_lf_preds == y_act_val) / np.sum(revised_lf_preds != ABSTAIN)

                active_mask_train = self.weak_labels[:, lf_idx] != ABSTAIN
                lf_prev_cov = np.mean(active_mask_train)
                X_act_train = self.get_feature(self.train_data)[active_mask_train, :]
                active_lf_preds_train = self.weak_labels[active_mask_train, lf_idx]
                active_y_preds_train = clf.predict(X_act_train)
                lf_revised_cov = np.sum(active_lf_preds_train == active_y_preds_train) / len(self.train_data)
                if lf_revised_acc_hat > lf_prev_acc_hat + 0.01:
                    print(f"Before revision: Cov={lf_prev_cov:.2f}, Acc={lf_prev_acc_hat:.2f}")
                    print(f"After  revision: Cov={lf_revised_cov:.2f}, Acc={lf_revised_acc_hat:.2f}")
                    if lf_idx in self.revision_models:
                        print(f"Update revision model for LF {lf_idx}")
                    else:
                        print(f"Create revision model for LF {lf_idx}")
                    self.revision_models[lf_idx] = clf

    def get_revised_dataset(self, dataset, apply_revision_models=True, indices=None, labels=None):
        """
        Revise dataset
        :param dataset: dataset to revise
        :param apply_revision_models: whether apply revision models to contrain LFs
        :param indices: indices of sampled train data
        :param labels: labels of sampled train data
        :return:
        """
        revised_weak_labels = np.array(dataset.weak_labels)
        revised_dataset = copy.copy(dataset)
        X = self.get_feature(dataset)
        if apply_revision_models:
            for lf_idx in range(dataset.n_lf):
                if lf_idx in self.revision_models:
                    lf_preds = check_weak_labels(revised_dataset)[:,lf_idx]
                    clf = self.revision_models[lf_idx]
                    y_pred = clf.predict(X)
                    revised_weak_labels[y_pred != lf_preds, lf_idx] = ABSTAIN
                elif indices is not None:
                    lf_output = revised_weak_labels[indices, lf_idx]
                    diff_indices = np.array(indices)[lf_output != np.array(labels)]
                    revised_weak_labels[diff_indices, lf_idx] = ABSTAIN

        for (clf, c) in self.append_lfs:
            pred = clf.predict(X)
            if self.only_append_uncovered:
                uncovered_mask = np.all(np.array(dataset.weak_labels) == ABSTAIN, axis=1)
                new_LF_out = np.where((pred == 1) & uncovered_mask, c, ABSTAIN).reshape(-1, 1)
            else:
                new_LF_out = np.where(pred == 1, c, ABSTAIN).reshape(-1, 1)
            revised_weak_labels = np.concatenate((revised_weak_labels, new_LF_out), axis=1)

        if len(self.discard_lfs) > 0:
            revised_weak_labels = np.delete(revised_weak_labels, self.discard_lfs, axis=1)

        revised_dataset.weak_labels = revised_weak_labels.tolist()
        revised_dataset.n_lf = revised_weak_labels.shape[1]
        return revised_dataset

