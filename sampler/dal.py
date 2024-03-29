from .base import BaseSampler
from sklearn.neural_network import MLPClassifier
import numpy as np
import torch


class DALSampler(BaseSampler):
    def __init__(self, train_data, labeller, label_model=None, revision_model=None, encoder=None, **kwargs):
        super(DALSampler, self).__init__(train_data, labeller, label_model, revision_model, encoder, **kwargs)
        if encoder is None:
            self.rep = self.train_data.features
        else:
            self.rep = self.encoder(torch.tensor(self.train_data.features)).detach().cpu().numpy()

    def train_discriminative_model(self, labeled_rep, unlabeled_rep):
        discriminator = MLPClassifier(hidden_layer_sizes=(100, 100),
                                      max_iter=500,
                                      early_stopping=True,
                                      random_state=0)
        y_l = np.zeros((labeled_rep.shape[0],1), dtype=int)
        y_u = np.ones((unlabeled_rep.shape[0],1), dtype=int)
        X_train = np.vstack((labeled_rep, unlabeled_rep))
        Y_train = np.vstack((y_l, y_u)).ravel()
        discriminator.fit(X_train,Y_train)
        return discriminator

    def sample_distinct(self, n=1):
        labeled_indices = np.nonzero(self.sampled != 0)[0]
        unlabeled_indices = np.nonzero(~self.sampled)[0]
        if len(labeled_indices) == 0:
            # do random sampling in first iteration
            order = np.random.permutation(len(self.candidate_indices))
        else:
            if len(unlabeled_indices) > len(labeled_indices) * 10:
                # subsampling to reduce class imbalance
                unlabeled_indices = np.random.choice(unlabeled_indices, len(labeled_indices)*10, replace=False)

            labeled_rep = self.rep[labeled_indices,:]
            unlabeled_rep = self.rep[unlabeled_indices, :]
            discriminator = self.train_discriminative_model(labeled_rep, unlabeled_rep)
            unlabeled_prob = discriminator.predict_proba(self.rep)[self.candidate_indices, 1]
            order = np.argsort(unlabeled_prob)[::-1]

        n_sampled, i, indices = 0, 0, []
        while n_sampled < n:
            idx = self.candidate_indices[order[i]]
            i += 1
            if self.sampled[idx]:
                continue
            self.sampled[idx] = True
            indices.append(idx)
            n_sampled += 1

        labels = self.label_selected_indices(indices)
        return indices, labels



