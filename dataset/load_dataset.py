from wrench.dataset import load_dataset
from dataset.synthetic import generate_syn_1, generate_syn_2, generate_syn_3


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
            extract_feature=True
        )
    else:
        raise ValueError(f"Extract_fn {extract_fn} not supported.")
    return train_data, valid_data, test_data


def load_synthetic_dataset(dataset):
    if dataset == "syn_1":
        train_data, valid_data, test_data = generate_syn_1()
    elif dataset == "syn_2":
        train_data, valid_data, test_data = generate_syn_2()
    elif dataset == "syn_3":
        train_data, valid_data, test_data = generate_syn_3()
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    return train_data, valid_data, test_data
