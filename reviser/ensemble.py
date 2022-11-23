from wrench.dataset.utils import check_weak_labels
from reviser.base import BaseReviser
from reviser.models import MLPNet
from reviser.trainers import NeuralNetworkTrainer
from torch.utils.data import TensorDataset
from utils import ABSTAIN
import copy
import numpy as np
import torch


class EnsembleReviser(BaseReviser):
    """
    Use deep ensemble for uncertainty estimation
    """
    def train_revision_model(self, indices, labels, cost):

        X_sampled = self.get_feature(self.train_data)[indices, :]
        y_sampled = labels
        training_dataset = TensorDataset(torch.tensor(X_sampled), torch.tensor(y_sampled))
        if self.valid_data is not None:
            X_eval = self.get_feature(self.valid_data)
            y_eval = self.valid_data.labels
            eval_dataset = TensorDataset(torch.tensor(X_eval), torch.tensor(y_eval))
        else:
            eval_dataset = None

        self.clf = []
        M = 10  # number of base classifiers
        np.random.seed(self.seed)
        model_seeds = np.random.randint(1, 100000, M)
        for i in range(M):
            torch.manual_seed(model_seeds[i])
            clf = MLPNet(input_dim=self.train_rep.shape[1], output_dim=self.train_data.n_class)
            trainer = NeuralNetworkTrainer(clf)
            trainer.train_model_with_dataloader(training_dataset, eval_dataset, device=self.device)
            self.clf.append(clf)

    def predict_labels(self, dataset, cost):
        proba = self.predict_proba(dataset)
        y_pred = np.argmax(proba, axis=1)
        max_prob = np.max(proba, axis=1)
        y_pred[max_prob < 1-cost] = ABSTAIN
        return y_pred

    def predict_proba(self, dataset):
        X = torch.tensor(self.get_feature(dataset)).to(self.device)
        M = len(self.clf)
        proba_list = []
        for i in range(M):
            proba = self.clf[i].predict_proba(X)
            proba_list.append(proba)

        proba = np.concatenate(proba_list).mean(axis=0)
        return proba


