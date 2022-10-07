from wrench.dataset import BaseDataset
from wrench.dataset.utils import check_weak_labels
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from utils import ABSTAIN, plot_tsne
import copy
import torch


class LFReviser:
    def __init__(self, dataset: BaseDataset, encoder, classifier, seed=None):
        """
        Initialize LF Reviser
        :param dataset: dataset to revise
        :param encoder: trained encoder model
        :param classifier: classifier used to revise LF
        """
        self.dataset = dataset
        self.revised_dataset = copy.copy(self.dataset)
        self.encoder = encoder
        self.weak_labels = check_weak_labels(self.dataset)
        self.classifier = classifier
        self.seed = seed
        self.revision_models = {}
        self.append_lfs = []

    def update_encoder(self, encoder):
        self.encoder = encoder

    def train_revision_models(self, indices, labels):
        # train revision models to revise existing LFs
        for lf_idx in range(self.dataset.n_lf):
            active_mask = self.weak_labels[indices, lf_idx] != ABSTAIN
            active_lf_preds = self.weak_labels[indices, lf_idx][active_mask]
            active_indices = np.array(indices)[active_mask]
            active_labels = np.array(labels)[active_mask]
            if self.encoder is not None:
                X_act = self.encoder(torch.tensor(self.dataset.features[active_indices,:])).detach().cpu().numpy()
            else:
                X_act = self.dataset.features[active_indices,:]

            y_act = (active_lf_preds == active_labels).astype(int)
            if 0 in y_act and 1 in y_act:  # TODO: add more specific criterion for revision
                if self.classifier == "logistic":
                    clf = LogisticRegression(random_state=self.seed, class_weight="balanced", max_iter=500)
                elif self.classifier == "linear-svm":
                    clf = SVC(kernel="linear", probability=True, random_state=self.seed, class_weight="balanced")
                else:
                    raise ValueError(f"classifier {self.classifier} not supported.")
                clf = clf.fit(X_act,y_act)
                self.revision_models[lf_idx] = clf
        # revise dataset to increase accuracy of LF
        revised_weak_labels = np.copy(self.weak_labels)
        if self.encoder is not None:
            X_all = self.encoder(torch.tensor(self.dataset.features)).detach().cpu().numpy()
        else:
            X_all = self.dataset.features
        for lf_idx in self.revision_models:
            clf = self.revision_models[lf_idx]
            pred = clf.predict(X_all)
            revised_weak_labels[pred == 0, lf_idx] = ABSTAIN

        # create new LFs to increase coverage
        abstain_mask = np.all(revised_weak_labels == ABSTAIN, axis=1)
        abstain_indices = []
        abstain_labels = []
        for (idx, y) in zip(indices, labels):
            if abstain_mask[idx]:
                abstain_indices.append(idx)
                abstain_labels.append(y)

        abstain_indices = np.array(abstain_indices)
        abstain_labels = np.array(abstain_labels)
        c = np.bincount(abstain_labels).argmax()
        if self.encoder is not None:
            X_uncov = self.encoder(torch.tensor(self.dataset.features[indices, :])).detach().cpu().numpy()
        else:
            X_uncov = self.dataset.features[indices, :]
        y_uncov = np.array(labels) == c
        # TODO: explore more possible LFs
        if self.classifier == "logistic":
            clf = LogisticRegression(random_state=self.seed, class_weight="balanced", max_iter=500)
        elif self.classifier == "linear-svm":
            clf = SVC(kernel="linear", probability=True, random_state=self.seed, class_weight="balanced")
        else:
            raise ValueError(f"classifier {self.classifier} not supported.")
        clf.fit(X_uncov, y_uncov)
        pred = clf.predict(X_all)
        new_LF_out = np.where(pred == 1, c, ABSTAIN).reshape(-1,1)
        revised_weak_labels = np.concatenate((revised_weak_labels, new_LF_out), axis=1)
        self.append_lfs.append((c, clf))  # target_class, classifier
        self.revised_dataset.n_lf += 1
        self.revised_dataset.weak_labels = revised_weak_labels.tolist()

    def get_revised_dataset(self, dataset: BaseDataset):
        revised_weak_labels = np.array(dataset.weak_labels)
        revised_dataset = copy.copy(dataset)
        if self.encoder is not None:
            X = self.encoder(torch.tensor(dataset.features)).detach().cpu().numpy()
        else:
            X = dataset.features
        for lf_idx in self.revision_models:
            clf = self.revision_models[lf_idx]
            pred = clf.predict(X)
            revised_weak_labels[pred == 0, lf_idx] = ABSTAIN

        for (c, clf) in self.append_lfs:
            pred = clf.predict(X)
            new_LF_out = np.where(pred == 1, c, ABSTAIN).reshape(-1, 1)
            revised_weak_labels = np.concatenate((revised_weak_labels, new_LF_out), axis=1)

        revised_dataset.weak_labels = revised_weak_labels.tolist()
        revised_dataset.n_Lf = revised_weak_labels.shape[1]
        return revised_dataset

