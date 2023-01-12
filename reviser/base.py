import numpy as np
import torch


class BaseReviser:
    def __init__(self,
                 train_data,
                 encoder,
                 device="cpu",
                 valid_data=None,
                 seed=None):
        """
        Initialize Reviser (active learning model)
        :param train_data: dataset to revise
        :param encoder: trained encoder model for feature transformation
        :param valid_data: validation dataset with golden labels
        :param seed: seed for revision models
        :param kwargs: other arguments for revision model
        """
        self.train_data = train_data
        self.valid_data = valid_data
        self.encoder = encoder
        self.train_rep = self.get_feature(self.train_data)
        self.device = device
        self.seed = seed
        self.clf = None
        self._features = None  # used by cluster margin sampler
        self._grads = None  # used by BADGE sampler
        self.sampled_indices = None
        self.sampled_labels = None

    def get_feature(self, dataset):
        if self.encoder is not None:
            return self.encoder(torch.tensor(dataset.features)).detach().cpu().numpy()
        else:
            return dataset.features

    def train_revision_model(self, indices, labels):
        """
        Train a revision model using sampled training data
        """
        raise NotImplementedError

    def get_pseudo_grads(self, dataset):
        """
        Get the pseudo gradient of data which is used by BADGE sampler
        """
        raise NotImplementedError

    def predict(self, dataset):
        """
        Predict labels for dataset with abstention
        :param dataset: dataset to predict
        """
        proba = self.predict_proba(dataset)
        preds = np.argmax(proba, axis=1)
        return preds

    def predict_proba(self, dataset):
        """
        Predict posterior class distribution of dataset.
        :param dataset: dataset to predict
        """
        raise NotImplementedError