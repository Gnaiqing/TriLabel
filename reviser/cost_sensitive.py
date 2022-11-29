from reviser.base import BaseReviser
from reviser.models import MLPNet
from reviser.trainers import CostSensitiveNetworkTrainer
from wrench.dataset.utils import check_weak_labels
from utils import ABSTAIN
import copy
import numpy as np
import torch
from torch.utils.data import TensorDataset


class CostSensitiveReviser(BaseReviser):
    def __init__(self,
                 train_data,
                 encoder,
                 device="cpu",
                 valid_data=None,
                 seed=None,
                 loss="cs-hinge"):
        super(CostSensitiveReviser, self).__init__(train_data, encoder, device, valid_data, seed)
        self.loss_type = loss

    def train_revision_model(self, indices, labels, cost):
        self.clf = MLPNet(input_dim=self.train_rep.shape[1], output_dim=self.train_data.n_class)
        trainer = CostSensitiveNetworkTrainer(self.clf, cost)
        X_sampled = self.get_feature(self.train_data)[indices, :]
        y_sampled = labels
        training_dataset = TensorDataset(torch.tensor(X_sampled), torch.tensor(y_sampled))
        if self.valid_data is not None:
            X_eval = self.get_feature(self.valid_data)
            y_eval = self.valid_data.labels
            eval_dataset = TensorDataset(torch.tensor(X_eval), torch.tensor(y_eval))
        else:
            eval_dataset = None

        trainer.train_model_with_dataloader(training_dataset, eval_dataset, device=self.device, which_loss=self.loss_type)

    def predict_labels(self, dataset, cost):
        X = torch.tensor(self.get_feature(dataset)).to(self.device)
        g = self.clf.inference_forward(X)
        y_pred = np.argmax(g, axis=1)
        g_max = np.max(g, axis=1)
        y_pred[g_max <= 0] = ABSTAIN
        return y_pred

    def predict_proba(self, dataset):
        return None