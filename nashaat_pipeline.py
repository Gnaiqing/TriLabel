import argparse
import time
from dataset.load_dataset import load_real_dataset, load_synthetic_dataset
from labeller.labeller import get_labeller
from sklearn.decomposition import PCA
import numpy as np
import copy
from utils import evaluate_performance, get_label_model, save_results, evaluate_golden_performance, get_sampler
from typing import Union


def update_results(results, n_labeled, frac_labeled, label_acc, label_nll,
                   label_brier, test_acc, test_f1):
    results["n_labeled"].append(n_labeled)
    results["frac_labeled"].append(frac_labeled)

    results["label_acc"].append(label_acc)
    results["label_nll"].append(label_nll)
    results["label_brier"].append(label_brier)

    results["test_acc"].append(test_acc)
    results["test_f1"].append(test_f1)


def run_nashaat(train_data, valid_data, test_data, args, seed):
    """
    Run nashaat pipeline to clean noisy labels
    """
    start = time.process_time()
    results = {
        "n_labeled": [],  # number of expert labeled data
        "frac_labeled": [],  # fraction of expert labeled data
        "label_acc": [],  # AL label accuracy
        "label_nll": [],  # AL label NLL score
        "label_brier": [],  # AL label brier score
        "test_acc": [],  # end model's test accuracy when using active learning
        "test_f1": [],  # end model's test f1 score (macro) using active learning
        "golden_test_acc": np.nan,  # end model's test accuracy using golden labels
        "golden_test_f1": np.nan,  # end model's test f1 score (macro) using golden labels
    }
    # record original stats
    label_model = get_label_model(args.label_model)
    if args.use_valid_labels:
        label_model.fit(dataset_train=train_data, dataset_valid=valid_data)
    else:
        label_model.fit(dataset_train=train_data)

    aggregated_soft_labels = label_model.predict_proba(train_data)
    perf = evaluate_performance(train_data, valid_data, test_data, aggregated_soft_labels, args, seed=seed)
    golden_perf = evaluate_golden_performance(train_data, valid_data, test_data, args, seed=seed)
    update_results(results, n_labeled=0, frac_labeled=0.0,
                   label_acc=perf["label_acc"], label_nll=perf["label_nll"], label_brier=perf["label_brier"],
                   test_acc=perf["test_acc"], test_f1=perf["test_f1"])
    results["golden_test_acc"] = golden_perf["test_acc"]
    results["golden_test_f1"] = golden_perf["test_f1"]

    # set labeller, sampler, reviser and encoder
    revised_train_data = copy.copy(train_data)
    labeller = get_labeller(args.labeller)
    sampler = get_sampler(args.sampler, revised_train_data, labeller, label_model=label_model,
                          revision_model=None, encoder=None)

    n_labeled = 0
    while n_labeled < args.sample_budget:
        n_to_sample = min(args.sample_budget - sampler.get_n_sampled(), args.sample_per_iter)
        sampler.sample_distinct(n=n_to_sample)
        indices, labels = sampler.get_sampled_points()

        revised_weak_labels = np.array(revised_train_data.weak_labels)
        revised_weak_labels[indices, :] = labels.reshape((-1, 1))
        revised_train_data.weak_labels = revised_weak_labels.tolist()
        if args.use_valid_labels:
            label_model.fit(dataset_train=revised_train_data, dataset_valid=valid_data)
        else:
            label_model.fit(dataset_train=revised_train_data)

        aggregated_soft_labels = label_model.predict_proba(revised_train_data)
        perf = evaluate_performance(revised_train_data, valid_data, test_data, aggregated_soft_labels, args, seed=seed)

        n_labeled = sampler.get_n_sampled()
        frac_labeled = n_labeled / len(revised_train_data)
        update_results(results, n_labeled=n_labeled, frac_labeled=frac_labeled,
                       label_acc=perf["label_acc"], label_nll=perf["label_nll"], label_brier=perf["label_brier"],
                       test_acc=perf["test_acc"], test_f1=perf["test_f1"])
        sampler.update_stats(revised_train_data, label_model=label_model)

    end = time.process_time()
    results["time"] = end - start
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # device info
    parser.add_argument("--device", type=str, default="cuda:0")
    # dataset
    parser.add_argument("--dataset", type=str, default="spambase")
    parser.add_argument("--dataset_path", type=str, default="../datasets/")
    parser.add_argument("--extract_fn", type=str, default=None)  # method used to extract features
    parser.add_argument("--max_dim", type=int, default=None)
    # sampler
    parser.add_argument("--sampler", type=str, default="uncertain")
    parser.add_argument("--sample_budget", type=float, default=0.05)  # Total sample budget
    parser.add_argument("--sample_per_iter",type=float, default=0.01)  # sample budget per iteration
    # label model and end models
    parser.add_argument("--label_model", type=str, default="metal")
    parser.add_argument("--end_model", type=str, default="mlp")
    parser.add_argument("--em_epochs", type=int, default=100)
    parser.add_argument("--use_soft_labels", action="store_true")
    # other settings
    parser.add_argument("--use_valid_labels", action="store_true")
    parser.add_argument("--labeller", type=str, default="oracle")
    parser.add_argument("--metric", type=str, default="f1_macro")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tag", type=str, default="test")
    args = parser.parse_args()

    if args.dataset[:3] == "syn":
        train_data, valid_data, test_data = load_synthetic_dataset(args.dataset)
    else:
        train_data, valid_data, test_data = load_real_dataset(args.dataset_path, args.dataset, args.extract_fn)

    if args.max_dim is not None and train_data.features.shape[1] > args.max_dim:
        # use truncated SVD to reduce feature dimensions
        pca = PCA(n_components=args.max_dim)
        pca.fit(train_data.features)
        train_data.features = pca.transform(train_data.features)
        valid_data.features = pca.transform(valid_data.features)
        test_data.features = pca.transform(test_data.features)

    if args.sample_budget < 1:
        args.sample_budget = np.ceil(args.sample_budget * len(train_data)).astype(int)
        args.sample_per_iter = np.ceil(args.sample_per_iter * len(train_data)).astype(int)
    else:
        args.sample_budget = int(args.sample_budget)
        args.sample_per_iter = int(args.sample_per_iter)

    np.random.seed(args.seed)
    run_seeds = np.random.randint(1, 100000, args.repeats)

    results_list = []
    for i in range(args.repeats):
        print(f"Start run {i}")
        results = run_nashaat(train_data, valid_data, test_data, args, seed=run_seeds[i])
        results_list.append(results)

    id_tag = f"nashaat_{args.label_model}_{args.sampler}_{args.tag}"
    save_results(results_list, args.output_path, args.dataset,
                 f"{id_tag}.json")
