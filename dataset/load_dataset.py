import numpy as np
import copy
from wrench.dataset import load_dataset
from dataset.synthetic import generate_syn_1, generate_syn_2, generate_syn_3, generate_random_counterpart


def load_real_dataset(dataset_path, dataset, extract_fn):
    if extract_fn == "bert":
        train_data, valid_data, test_data = load_dataset(
            dataset_path,
            dataset,
            extract_feature=True,
            extract_fn='bert',  # extract bert embedding
            model_name='bert-base-cased',
            cache_name='bert'
        )
    elif extract_fn == "bow":
        train_data, valid_data, test_data = load_dataset(
            dataset_path,
            dataset,
            extract_feature=True,
            extract_fn='bow',  # extract bow embedding
        )
    elif extract_fn is None:
        train_data, valid_data, test_data = load_dataset(
            dataset_path,
            dataset,
            extract_feature=True,
            normalize=True
        )
    else:
        raise ValueError(f"Extract_fn {extract_fn} not supported.")
    return train_data, valid_data, test_data


def shuffle_dataset(train_data, valid_data, test_data, seed):
    """
    Shuffle the datasets to make sure they are i.i.d. distributed
    :param train_data: original split of training data
    :param valid_data: original split of valid data
    :param test_data: original split of test data
    :return:
    """
    # first, merge data into a large dataset
    merge_data = train_data.__class__()
    for dataset in [train_data, valid_data, test_data]:
        new_ids = [dataset.split + id for id in dataset.ids]
        merge_data.ids += new_ids
        merge_data.labels += dataset.labels
        merge_data.examples += dataset.examples
        merge_data.weak_labels += dataset.weak_labels
        if merge_data.features is None:
            merge_data.features = dataset.features
        else:
            merge_data.features = np.concatenate((merge_data.features, dataset.features), axis=0)

    merge_data.id2label = copy.deepcopy(train_data.id2label)
    merge_data.path = train_data.path
    merge_data.n_class = train_data.n_class
    merge_data.n_lf = train_data.n_lf
    # then split the dataset with their original size
    np.random.seed(seed)
    indices = np.arange(len(merge_data))
    train_indices = np.random.choice(indices, len(train_data), replace=False)
    indices = np.setdiff1d(indices, train_indices)
    valid_indices = np.random.choice(indices, len(valid_data), replace=False)
    test_indices = np.setdiff1d(indices, valid_indices)
    new_train_data = merge_data.create_subset(train_indices.tolist())
    new_train_data.split = "train"
    new_valid_data = merge_data.create_subset(valid_indices.tolist())
    new_valid_data.split = "valid"
    new_test_data = merge_data.create_subset(test_indices.tolist())
    new_test_data.split = "test"
    return new_train_data, new_valid_data, new_test_data


def load_synthetic_dataset(dataset):
    if dataset == "syn_1":
        train_data, valid_data, test_data = generate_syn_1()
    elif dataset == "syn_2":
        train_data, valid_data, test_data = generate_syn_2()
    elif dataset == "syn_3":
        train_data, valid_data, test_data = generate_syn_3()
    elif dataset == "syn_1_random":
        train_data, valid_data, test_data = generate_syn_1()
        train_data = generate_random_counterpart(train_data)
        valid_data = generate_random_counterpart(valid_data)
        test_data = generate_random_counterpart(test_data)
    elif dataset == "syn_2_random":
        train_data, valid_data, test_data = generate_syn_2()
        train_data = generate_random_counterpart(train_data)
        valid_data = generate_random_counterpart(valid_data)
        test_data = generate_random_counterpart(test_data)
    elif dataset == "syn_3_random":
        train_data, valid_data, test_data = generate_syn_3()
        train_data = generate_random_counterpart(train_data)
        valid_data = generate_random_counterpart(valid_data)
        test_data = generate_random_counterpart(test_data)
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    return train_data, valid_data, test_data
