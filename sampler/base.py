import numpy as np
from numpy.random import default_rng
from labeller.labeller import get_item


class BaseSampler:
    def __init__(self, train_data, labeller, **kwargs):
        self.train_data = train_data
        self.labeller = labeller
        self.kwargs = kwargs
        self.sampled = np.zeros(len(self.train_data))
        self.sampled_labels = np.repeat(-1, len(self.train_data)).astype(int)
        if "sampled_indices" in kwargs:
            assert "sampled_labels" in kwargs
            assert len(kwargs["sampled_indices"]) == len(kwargs["sampled_labels"])
            self.sampled[kwargs["sampled_indices"]] = True
            self.sampled_labels[kwargs["sampled_indices"]] = np.array(kwargs["sampled_labels"])

        if "seed" in kwargs:
            self.rng = default_rng(kwargs["seed"])
        else:
            self.rng = default_rng()

    def sample_distinct(self, n=1):
        raise NotImplementedError()

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



