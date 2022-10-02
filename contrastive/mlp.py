import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from contrastive.losses import SupConLoss
import numpy as np


class MLP(pl.LightningModule):
    """
    Multilayer Perceptron for feature transformation
    """
    def __init__(self, dim_in, dim_out, dim_hidden=(200,), temp=0.07):
        super().__init__()
        layers = [nn.Linear(dim_in, dim_hidden[0]), nn.ReLU()]
        for i in range(len(dim_hidden) - 1):
            layers.append(nn.Linear(dim_hidden[i], dim_hidden[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dim_hidden[-1], dim_out))

        self.layers = nn.Sequential(*layers)
        self.scl = SupConLoss(temperature=temp)

    def forward(self, x):
        feat = self.layers(x)
        feat = F.normalize(feat, dim=1)
        return feat

    def get_mask(self, y):
        mask = np.eye(len(y))
        same = (y.reshape(-1,1) == y.reshape(1,-1)) & (y.reshape(-1,1) != -1)
        mask[same] = 1
        return torch.tensor(mask)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_in = x.reshape(-1,x.shape[-1])  # (n_data*n_views, n_features)
        feat = self.forward(x_in)
        feat = feat.reshape(x.shape[0], x.shape[1], -1) # (n_data, n_views, dim_out)
        mask = self.get_mask(y)
        loss = self.scl(feat, mask=mask)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer




