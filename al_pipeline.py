# pipeline for active learning
import sys
import time
import argparse
from sklearn.decomposition import PCA
from dataset.load_dataset import load_real_dataset, load_synthetic_dataset
from labeller.labeller import get_labeller
import numpy as np
from utils import evaluate_al_performance, save_results, evaluate_golden_performance, \
    get_sampler, ABSTAIN


def update_results(results, n_labeled, frac_labeled, al_label_acc, al_label_nll,
                   al_label_brier, al_test_acc, al_test_f1):
    results["n_labeled"].append(n_labeled)
    results["frac_labeled"].append(frac_labeled)

    results["al_label_acc"].append(al_label_acc)
    results["al_label_nll"].append(al_label_nll)
    results["al_label_brier"].append(al_label_brier)

    results["al_test_acc"].append(al_test_acc)
    results["al_test_f1"].append(al_test_f1)


def run_active_learning(train_data, valid_data, test_data, args, seed):
    start = time.process_time()
    results = {
        "n_labeled": [],  # number of expert labeled data
        "frac_labeled": [],  # fraction of expert labeled data
        "al_label_acc": [],  # AL label accuracy
        "al_label_nll": [],  # AL label NLL score
        "al_label_brier": [],  # AL label brier score
        "al_test_acc": [],  # end model's test accuracy when using active learning
        "al_test_f1": [],  # end model's test f1 score (macro) using active learning
        "golden_test_acc": np.nan,  # end model's test accuracy using golden labels
        "golden_test_f1": np.nan,  # end model's test f1 score (macro) using golden labels
    }
    # record original stats
    golden_perf = evaluate_golden_performance(train_data, valid_data, test_data, args, seed=seed)
    update_results(results, n_labeled=0, frac_labeled=0.0,
                   al_label_acc=np.nan, al_label_nll=np.nan, al_label_brier=np.nan,
                   al_test_acc=np.nan, al_test_f1=np.nan)
    results["golden_test_acc"] = golden_perf["test_acc"]
    results["golden_test_f1"] = golden_perf["test_f1"]

    n_sampled = 0
    labeller = get_labeller(args.labeller)
    sampler = get_sampler("passive", train_data, labeller,
                          label_model=None, revision_model=None, encoder=None, seed=seed)
    while n_sampled < args.sample_budget:
        n_to_sample = min(args.sample_budget - n_sampled, args.sample_per_iter)
        if n_sampled == 0:
            # passive sampling for first iteration
            sampler.sample_distinct(n_to_sample)
        else:
            indices = []
            for idx in uncertainty_order:
                if not sampler.sampled[idx]:
                    indices.append(idx)
                if len(indices) >= n_to_sample:
                    break

            sampler.label_selected_indices(indices)

        indices, labels = sampler.get_sampled_points()
        ground_truth_labels = np.repeat(ABSTAIN, len(train_data))
        ground_truth_labels[indices] = labels
        perf = evaluate_al_performance(train_data, valid_data, test_data, args, seed,
                                    ground_truth_labels=ground_truth_labels)
        uncertainty_order = perf["uncertainty_order"]

        n_sampled = sampler.get_n_sampled()
        frac_sampled = n_sampled / len(train_data)
        update_results(results, n_labeled=n_sampled, frac_labeled=frac_sampled,
                       al_label_acc=perf["label_acc"], al_label_nll=perf["label_nll"], al_label_brier=perf["label_brier"],
                       al_test_acc=perf["test_acc"], al_test_f1=perf["test_f1"])

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
    parser.add_argument("--sample_budget", type=float, default=0.05)  # Total sample budget
    parser.add_argument("--sample_per_iter",type=float, default=0.01)  # sample budget per iteration
    # end model
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
    parser.add_argument("--load_results", action="store_true")
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
        results = run_active_learning(train_data, valid_data, test_data, args, seed=run_seeds[i])
        results_list.append(results)

    id_tag = f"al_{args.end_model}_{args.tag}"
    save_results(results_list, args.output_path, args.dataset, f"{id_tag}.json")


