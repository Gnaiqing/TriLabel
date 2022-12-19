from reviser.base import BaseReviser
from reviser.models import MLPNet
from reviser.trainers import NeuralNetworkTrainer
from torch.utils.data import TensorDataset
import torch
import numpy as np


class MLPReviser(BaseReviser):
    """
    MLP trained with cross entropy loss
    """
    def train_revision_model(self, indices, labels):
        self.clf = MLPNet(input_dim=self.train_rep.shape[1], output_dim=self.train_data.n_class)
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

    def predict_proba(self, dataset):
        if self.clf is None:
            return np.ones((len(dataset), dataset.n_class)) / dataset.n_class
        X = torch.tensor(self.get_feature(dataset)).to(self.device)
        proba = self.clf.predict_proba(X)
        return proba


