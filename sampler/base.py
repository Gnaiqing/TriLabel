import numpy as np
from numpy.random import default_rng
from labeller.labeller import get_item


class BaseSampler:
    def __init__(self, train_data, labeller, label_model=None, revision_model=None, encoder=None, **kwargs):
        self.train_data = train_data
        self.weak_labels = np.array(train_data.weak_labels)
        self.labeller = labeller
        self.label_model = label_model
        self.revision_model = revision_model
        self.encoder = encoder
        self.initialized = False
        self.kwargs = kwargs
        self.sampled = np.repeat(False, len(self.train_data))
        self.sampled_labels = np.repeat(-1, len(self.train_data)).astype(int)
        if "sampled_indices" in kwargs:
            assert "sampled_labels" in kwargs
            assert len(kwargs["sampled_indices"]) == len(kwargs["sampled_labels"])
            self.sampled[kwargs["sampled_indices"]] = True
            self.sampled_labels[kwargs["sampled_indices"]] = np.array(kwargs["sampled_labels"])

        self.candidate_indices = np.nonzero(~self.sampled)[0]
        if "seed" in kwargs:
            self.rng = default_rng(kwargs["seed"])
        else:
            self.rng = default_rng()

    def sample_distinct(self, n=1):
        raise NotImplementedError()

    def update_stats(self, train_data=None, label_model=None, revision_model=None):
        if train_data is not None:
            self.train_data = train_data
            self.weak_labels = np.array(train_data.weak_labels)
        if label_model is not None:
            self.label_model = label_model
        if revision_model is not None:
            self.revision_model = revision_model
        self.candidate_indices = np.nonzero(~self.sampled)[0]

    def label_selected_indices(self, indices):
        """
        Label selected (unlabeled) indices
        :param indices: indices to label
        :return:
        """
        labels = []
        for idx in indices:
            self.sampled[idx] = True
            data = get_item(self.train_data, idx)
            label = self.labeller(data)
            self.sampled_labels[idx] = label
            labels.append(label)

        return labels

    def get_sampled_points(self):
        sampled_indices = np.nonzero(self.sampled != 0)[0]
        sampled_labels = self.sampled_labels[sampled_indices]
        return sampled_indices, sampled_labels

    def get_n_sampled(self):
        return int(np.sum(self.sampled))



