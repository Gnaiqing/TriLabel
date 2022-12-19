from reviser.base import BaseReviser
from reviser.models import MLPNet
from reviser.trainers import NeuralNetworkTrainer
from torch.utils.data import TensorDataset
import numpy as np
import torch


class DalenReviser(BaseReviser):
    """
    Use deep ensemble for uncertainty estimation
    """
    def train_revision_model(self, indices, labels):
        # train ensemble of MLP for uncertainty estimation
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

        # train a classifier to predict whether the data comes from labeled distribution
        self.rejector = MLPNet(input_dim=self.train_rep.shape[1], output_dim=2)
        unlabeled_indices = np.setdiff1d(np.arange(len(self.train_data)), indices)
        if len(unlabeled_indices) > len(indices) * 5:
            unlabeled_indices = np.random.choice(unlabeled_indices, len(indices)*5, replace=False)

        X_l = self.get_feature(self.train_data)[indices, :]
        X_u = self.get_feature(self.train_data)[unlabeled_indices, :]
        X = np.vstack((X_l, X_u))
        y = np.concatenate((np.repeat(0, len(indices)), np.repeat(1, len(unlabeled_indices))))
        dis_dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
        trainer = NeuralNetworkTrainer(self.rejector)
        trainer.train_model_with_dataloader(dis_dataset, None, device=self.device, early_stopping=False)

    def predict_proba(self, dataset):
        if self.clf is None:
            return np.ones((len(dataset), dataset.n_class)) / dataset.n_class
        X = torch.tensor(self.get_feature(dataset)).to(self.device)
        M = len(self.clf)
        proba_list = []
        for i in range(M):
            proba = self.clf[i].predict_proba(X)
            proba_list.append(proba)

        proba = np.stack(proba_list, axis=0)
        proba = proba.mean(axis=0)

        reject = self.rejector.predict(X)
        proba[reject == 1, :] = 1 / self.train_data.n_class
        return proba


