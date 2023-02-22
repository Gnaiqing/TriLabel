import numpy as np
from utils import get_label_model
import copy


class EnsembleCalibrator:
    def __init__(self, train_data, valid_data, label_model_type, ensemble_size, feature_size, bootstrap, seed):
        self.ensemble_size = ensemble_size
        self.train_data = train_data
        self.valid_data = valid_data
        self.bagging_lms = []
        self.selected_features = []
        np.random.seed(seed)
        weak_labels = np.array(train_data.weak_labels)
        N, M = weak_labels.shape
        valid_weak_labels = np.array(valid_data.weak_labels)
        for i in range(ensemble_size):
            if bootstrap:
                selected_indices = np.random.choice(N, N, replace=True)
            else:
                selected_indices = np.random.permutation(N)
            if feature_size < M:
                selected_features = np.random.choice(M, feature_size, replace=False)
            else:
                selected_features = np.arange(M)

            self.selected_features.append(selected_features)

            bagging_weak_labels = weak_labels[selected_indices,:]
            bagging_weak_labels = bagging_weak_labels[:, selected_features]
            bagging_valid_weak_labels = valid_weak_labels[:, selected_features]
            bagging_train_data = copy.copy(train_data)
            bagging_train_data.weak_labels = bagging_weak_labels.tolist()
            bagging_train_data.n_lf = len(selected_features)
            bagging_train_data = bagging_train_data.get_covered_subset()
            bagging_valid_data = copy.copy(valid_data)
            bagging_valid_data.weak_labels = bagging_valid_weak_labels.tolist()
            bagging_valid_data.n_lf = len(selected_features)
            label_model = get_label_model(label_model_type)
            label_model.fit(dataset_train=bagging_train_data, dataset_valid=bagging_valid_data)
            self.bagging_lms.append(label_model)

    def predict_proba(self, dataset):
        probs = []
        weak_labels = np.array(dataset.weak_labels)
        for i in range(self.ensemble_size):
            label_model = self.bagging_lms[i]
            selected_features = self.selected_features[i]
            bagging_probs = label_model.predict_proba(weak_labels[:, selected_features])
            probs.append(bagging_probs.tolist())

        probs = np.array(probs).mean(axis=0)
        return probs

    def predict(self, dataset):
        probs = self.predict_proba(dataset)
        preds = np.argmax(probs, axis=1)
        return preds