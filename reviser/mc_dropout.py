from wrench.dataset.utils import check_weak_labels
from reviser.base import BaseReviser
from reviser.models import DropOutNet
from reviser.trainers import NeuralNetworkTrainer
from torch.utils.data import TensorDataset
from utils import ABSTAIN
import copy
import numpy as np
import torch


class MCDropoutReviser(BaseReviser):
    """
    Use MC Dropout to estimate uncertainty
    """
    def train_revision_model(self, indices, labels, cost):
        self.clf = DropOutNet(input_dim=self.train_rep.shape[1], output_dim=self.train_data.n_class)
        trainer = NeuralNetworkTrainer(self.clf)
        X_sampled = self.get_feature(self.train_data)[indices, :]
        y_sampled = labels
        training_dataset = TensorDataset(torch.tensor(X_sampled), torch.tensor(y_sampled))
        if self.valid_data is not None:
            X_eval = self.get_feature(self.valid_data)
            y_eval = self.valid_data.labels
            eval_dataset = TensorDataset(torch.tensor(X_eval), torch.tensor(y_eval))
        else:
            eval_dataset = None

        trainer.train_model_with_dataloader(training_dataset, eval_dataset, device=self.device)

    def predict_labels(self, dataset, cost):
        proba = self.predict_proba(dataset)
        y_pred = np.argmax(proba, axis=1)
        max_prob = np.max(proba, axis=1)
        y_pred[max_prob < 1-cost] = ABSTAIN
        return y_pred

    def predict_proba(self, dataset):
        if self.clf is None:
            return None

        X = torch.tensor(self.get_feature(dataset)).to(self.device)
        proba = self.clf.predict_proba(X)
        return proba