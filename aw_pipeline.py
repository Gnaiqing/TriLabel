# pipeline for active weasul
import sys
import argparse
from dataset.load_dataset import load_real_dataset, load_synthetic_dataset
from labeller.labeller import get_labeller
import json
import numpy as np
from utils import evaluate_performance, plot_tsne, plot_results, save_results, evaluate_golden_performance, get_sampler
import copy
from typing import Union
from main_rlf import update_results


def run_active_weasul(train_data, valid_data, test_data, args, seed):
    """
    Run nashaat pipeline to clean noisy labels
    """
    results = {
        "n_labeled": [],
        "frac_labeled": [],
        "train_coverage": [],
        "train_covered_acc": [],
        "em_test": [],
    }
    # record original stats
    perf = evaluate_performance(train_data, valid_data, test_data, args, seed=seed)
    golden_perf = evaluate_golden_performance(train_data, valid_data, test_data, args, seed=seed)
    update_results(results, perf, n_labeled=0, frac_labeled=0.0)
    results["em_test_golden"] = golden_perf["em_test"]

    if args.verbose:
        lf_sum = train_data.lf_summary()
        print("Original LF summary:\n", lf_sum)
        print("Train coverage: ", perf["train_coverage"])
        print("Train covered acc: ", perf["train_covered_acc"])
        print("Test set acc: ", perf["em_test"])
        print("Golden test set acc: ", golden_perf["em_test"])

    # set labeller, sampler, reviser and encoder
    labeller = get_labeller(args.labeller)
    sampler = get_sampler(args.sampler, train_data, labeller, label_model=args.label_model,
                          penalty_strength=args.penalty_strength)
    if args.sample_budget < 1:
        args.sample_budget = np.ceil(args.sample_budget * len(train_data)).astype(int)
        args.sample_per_iter = np.ceil(args.sample_per_iter * len(train_data)).astype(int)

    while sampler.get_n_sampled() < args.sample_budget:
        n_to_sample = min(args.sample_budget - sampler.get_n_sampled(), args.sample_per_iter)
        sampler.sample_distinct(n=n_to_sample)
        indices, labels = sampler.get_sampled_points()
        ground_truth_labels = np.repeat(-1, len(train_data))
        ground_truth_labels[indices] = labels
        perf = evaluate_performance(train_data, valid_data, test_data, args, seed,
                                    ground_truth_labels=ground_truth_labels)
        n_labeled = sampler.get_n_sampled()
        frac_labeled = n_labeled / len(train_data)
        update_results(results, perf, n_labeled=n_labeled, frac_labeled=frac_labeled)
        sampler.update_dataset(train_data, ground_truth_labels)
        if args.verbose:
            print(f"Revised summary at {n_labeled}({frac_labeled * 100:.1f}%):")
            print("Train coverage: ", perf["train_coverage"])
            print("Train covered acc: ", perf["train_covered_acc"])
            print("Test set acc: ", perf["em_test"])

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # device info
    parser.add_argument("--device", type=str, default="cuda:0")
    # dataset
    parser.add_argument("--dataset", type=str, default="youtube")
    parser.add_argument("--dataset_path", type=str, default="../datasets/")
    parser.add_argument("--extract_fn", type=str, default="bert")  # method used to extract features
    # sampler
    parser.add_argument("--sampler", type=str, default="maxkl")
    parser.add_argument("--sample_budget", type=Union[int, float], default=0.10)  # Total sample budget
    parser.add_argument("--sample_per_iter",type=Union[int, float], default=0.01)  # sample budget per iteration
    # label model and end models
    parser.add_argument("--label_model", type=str, default="aw")
    parser.add_argument("--penalty_strength", type=float, default=1.0)
    parser.add_argument("--end_model", type=str, default="roberta")
    parser.add_argument("--em_epochs", type=int, default=5)
    parser.add_argument("--use_soft_labels", action="store_true")
    # other settings
    parser.add_argument("--use_valid_labels", action="store_true")
    parser.add_argument("--labeller", type=str, default="oracle")
    parser.add_argument("--metric", type=str, default="acc")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tag", type=str, default="0")
    parser.add_argument("--load_results", type=str, default=None)
    args = parser.parse_args()

    if args.sample_budget < 1:
        plot_labeled_frac = True
    else:
        plot_labeled_frac = False

    if args.load_results is not None:
        readfile = open(args.load_results, "r")
        results = json.load(readfile)
        results_list = results["data"]
        plot_results(results_list, args.output_path, args.dataset, f"{args.dataset}_{args.tag}", plot_labeled_frac)
        sys.exit(0)

    if args.dataset[:3] == "syn":
        train_data, valid_data, test_data = load_synthetic_dataset(args.dataset)
    else:
        train_data, valid_data, test_data = load_real_dataset(args.dataset_path, args.dataset, args.extract_fn)

    np.random.seed(args.seed)
    run_seeds = np.random.randint(1, 100000, args.repeats)

    results_list = []
    for i in range(args.repeats):
        print(f"Start run {i}")
        results = run_active_weasul(train_data, valid_data, test_data, args, seed=run_seeds[i])
        results_list.append(results)

    save_results(results_list, args.output_path, args.dataset,
                 f"{args.label_model}-{args.end_model}-aw_{args.tag}.json")
    plot_results(results_list, args.output_path, args.dataset, args.dataset,
                 f"{args.label_model}-{args.end_model}-aw_{args.tag}.jpg",
                 plot_labeled_frac)


