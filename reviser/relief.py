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
        self.encoder = encoder
        self.weak_labels = check_weak_labels(self.dataset)
        self.classifier = classifier
        self.seed = seed
        self.revision_models = {}

    def update_encoder(self, encoder):
        self.encoder = encoder

    def train_revision_models(self, indices, labels):
        for lf_idx in range(self.dataset.n_lf):
            active_mask = self.weak_labels[indices, lf_idx] != ABSTAIN
            active_lf_preds = self.weak_labels[indices, lf_idx][active_mask]
            active_indices = np.array(indices)[active_mask]
            active_labels = np.array(labels)[active_mask]
            if self.encoder is not None:
                X = self.encoder(torch.tensor(self.dataset.features[active_indices,:])).detach().cpu().numpy()
            else:
                X = self.dataset.features[active_indices,:]

            y = (active_lf_preds == active_labels).astype(int)
            if 0 in y and 1 in y:
                if self.classifier == "logistic":
                    clf = LogisticRegression(random_state=self.seed, class_weight="balanced", max_iter=500)
                elif self.classifier == "linear-svm":
                    clf = SVC(kernel="linear", probability=True, random_state=self.seed, class_weight="balanced")
                else:
                    raise ValueError(f"classifier {self.classifier} not supported.")
                clf = clf.fit(X,y)
                y_pred = clf.predict(X)
                C = confusion_matrix(y, y_pred)
                self.revision_models[lf_idx] = clf

    def evaluate_revision_models(self):
        model_acc = - np.ones(self.dataset.n_lf)
        if self.encoder is not None:
            X = self.encoder(torch.tensor(self.dataset.features)).detach().cpu().numpy()
        else:
            X = self.dataset.features
        for lf_idx in self.revision_models:
            clf = self.revision_models[lf_idx]
            active_mask = self.weak_labels[:,lf_idx] != ABSTAIN
            pred = clf.predict(X[active_mask, :])
            gt = np.array(self.dataset.labels)[active_mask] == self.weak_labels[active_mask,lf_idx]
            C = confusion_matrix(gt, pred)
            model_acc[lf_idx] = accuracy_score(gt, pred)

        return model_acc

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

        revised_dataset.weak_labels = revised_weak_labels.tolist()
        return revised_dataset

