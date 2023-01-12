from reviser.base import BaseReviser
from reviser.models import MLPNet
from reviser.trainers import NeuralNetworkTrainer
from torch.utils.data import TensorDataset
import numpy as np
import torch


class EnsembleReviser(BaseReviser):
    """
    Use deep ensemble for uncertainty estimation
    """
    def train_revision_model(self, indices, labels):
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

        if self._features is None:
            self._features = []
            self.predict_proba(self.train_data)
            for i in range(M):
                self._features.append(self.clf[i]._features.cpu().numpy())

            self._features = np.mean(self._features, axis=0)

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
        return proba

    def get_pseudo_grads(self, dataset):
        preds = self.predict(dataset)
        X = torch.tensor(self.get_feature(dataset)).to(self.device)
        y_hat = torch.tensor(preds).to(self.device)
        M = len(self.clf)
        N = X.shape[0]
        grads_list = []

        loss_func = torch.nn.CrossEntropyLoss()
        for i in range(M):
            optimizer = torch.optim.Adam(self.clf[i].parameters())  # used for clear gradient
            cur_grads = []
            for j in range(N):
                optimizer.zero_grad()
                output = self.clf[i](X[j,:])
                loss = loss_func(output, y_hat[j])
                loss.backward()
                cur_grads.append(self.clf[i].fc2.weight.grad.detach().cpu().numpy().flatten())

            cur_grads = np.vstack(cur_grads)
            grads_list.append(cur_grads)

        grads = np.mean(grads_list, axis=0)
        return grads




