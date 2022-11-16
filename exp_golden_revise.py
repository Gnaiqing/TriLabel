"""
Evaluate the classifier accuracy when revising LF by training classifiers with all golden labels
"""
import argparse
from dataset.load_dataset import load_real_dataset, load_synthetic_dataset
from reviser.reviser import LFReviser
import numpy as np
import pandas as pd
from pytorch_lightning import seed_everything
from utils import evaluate_performance, plot_results, save_results, evaluate_golden_performance, \
    evaluate_golden_noise_reduction_performance, ABSTAIN, get_sampler
from labeller.labeller import get_labeller
from main_rlf import update_results, get_feature_encoder
import copy
import json
from pathlib import Path
import matplotlib.pyplot as plt

def run_golden_rlf(train_data, valid_data, test_data, args, seeds):
    origin_results = {
        "em_test": [],
        "trainset_accuracy": [],
        "trainset_size": [],
        "trainset_coverage": [],
    }  # no revision
    golden_results = {
        "em_test": [],
        "trainset_accuracy": [],
        "trainset_size": [],
        "trainset_coverage": [],
    } # use golden labels
    filter_results = {
        "em_test": [],
        "trainset_accuracy": [],
        "trainset_size": [],
        "trainset_coverage": [],
    } # golden filter incorrect LF predictions
    relief_results = {
        "em_test": [],
        "trainset_accuracy": [],
        "trainset_size": [],
        "trainset_coverage": [],
    } # current pipeline results

    for seed in seeds:
        seed_everything(seed, workers=True)
        # evaluate results with no revision
        covered_train_data = train_data.get_covered_subset()
        perf = evaluate_performance(train_data, valid_data, test_data, args, seed=seed)
        origin_results["em_test"].append(perf["em_test"])
        origin_results["trainset_size"].append(len(covered_train_data))
        origin_results["trainset_accuracy"].append(perf["train_covered_acc"])
        origin_results["trainset_coverage"].append(perf["train_coverage"])
        # evaluate results with golden labels
        golden_perf = evaluate_golden_performance(train_data, valid_data, test_data, args, seed=seed)
        golden_results["em_test"].append(golden_perf["em_test"])
        golden_results["trainset_size"].append(len(train_data))
        golden_results["trainset_accuracy"].append(1.0)
        golden_results["trainset_coverage"].append(1.0)
        # evaluate results with filtered LF prediction
        revised_train_data = copy.copy(train_data)
        revised_weak_labels = np.array(revised_train_data.weak_labels)
        labels = np.array(revised_train_data.labels).reshape(-1,1)
        revised_weak_labels[revised_weak_labels != labels] = ABSTAIN
        revised_train_data.weak_labels = revised_weak_labels.tolist()
        covered_train_data = revised_train_data.get_covered_subset()
        filtered_perf = evaluate_performance(revised_train_data, valid_data, test_data, args, seed=seed)
        filter_results["em_test"].append(filtered_perf["em_test"])
        filter_results["trainset_size"].append(len(covered_train_data))
        filter_results["trainset_accuracy"].append(filtered_perf["train_covered_acc"])
        filter_results["trainset_coverage"].append(filtered_perf["train_coverage"])
        # evaluate results with current LF revision framework
        n_to_sample = int(len(train_data) * args.sample_frac)
        labeller = get_labeller(args.labeller)
        sampler = get_sampler(args.sampler, train_data, labeller)
        active_LF = [i for i in range(train_data.n_lf)]
        sampler.sample_distinct(n_to_sample, active_LF=active_LF)
        indices, labels = sampler.get_sampled_points()
        encoder = get_feature_encoder(train_data, indices, labels, args)
        reviser = LFReviser(train_data, encoder, args.lf_class, args.revision_model_class,
                            valid_data=valid_data, seed=seed)
        reviser.revise_label_functions(indices, labels)
        revised_train_data = reviser.get_revised_dataset(dataset=train_data)
        covered_train_data = revised_train_data.get_covered_subset()
        perf = evaluate_performance(revised_train_data, valid_data, test_data, args, seed=seed)
        relief_results["em_test"].append(perf["em_test"])
        relief_results["trainset_size"].append(len(covered_train_data))
        relief_results["trainset_accuracy"].append(perf["train_covered_acc"])
        relief_results["trainset_coverage"].append(perf["train_coverage"])



    return origin_results, golden_results, filter_results, relief_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # device info
    parser.add_argument("--device", type=str, default="cuda:0")
    # dataset
    parser.add_argument("--dataset", type=str, default="youtube")
    parser.add_argument("--dataset_path", type=str, default="../wrench-1.1/datasets/")
    parser.add_argument("--extract_fn", type=str, default="bert")  # method used to extract features
    # contrastive learning
    parser.add_argument("--contrastive_mode", type=str, default=None)
    parser.add_argument("--data_augment", type=str, default="eda")
    parser.add_argument("--n_aug", type=int, default=2)
    parser.add_argument("--dim_out", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=10)
    # sampler
    parser.add_argument("--sampler", type=str, default="lfcov")
    parser.add_argument("--sample_frac", type=float, default=0.05)  # fraction of data points sampled
    # revision model
    parser.add_argument("--revision_method", type=str, default="relief")  # "nashaat": only revise labeled points
    parser.add_argument("--revision_model_class", type=str, default="logistic")
    parser.add_argument("--lf_class", type=str, default="logistic")
    parser.add_argument("--only_append_uncovered",
                        action="store_true")  # let new LF only get activated on uncovered data
    # label model and end model
    parser.add_argument("--label_model", type=str, default="mv")
    parser.add_argument("--end_model", type=str, default="mlp")
    parser.add_argument("--em_epochs", type=int, default=100)
    parser.add_argument("--em_batch_size", type=int, default=4096)
    parser.add_argument("--em_patience", type=int, default=100)
    parser.add_argument("--em_lr", type=float, default=0.01)
    parser.add_argument("--em_weight_decay", type=float, default=0.0001)
    parser.add_argument("--use_soft_labels", action="store_true")
    # other settings
    parser.add_argument("--labeller", type=str, default="oracle")
    parser.add_argument("--metric", type=str, default="acc")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--tag", type=str, default="GOLD")
    args = parser.parse_args()

    filename = "golden_cmp.json"
    filepath = Path(args.output_path) / args.dataset / filename
    if Path.exists(filepath):
        f = open(filepath)
        results = json.load(f)
        origin_results = pd.DataFrame(results["origin_results"])
        golden_results = pd.DataFrame(results["golden_results"])
        filter_results = pd.DataFrame(results["filter_results"])
        relief_results = pd.DataFrame(results["relief_results"])
        f.close()
    else:
        if args.dataset[:3] == "syn":
            train_data, valid_data, test_data = load_synthetic_dataset(args.dataset)
        else:
            train_data, valid_data, test_data = load_real_dataset(args.dataset_path, args.dataset, args.extract_fn)

        np.random.seed(args.seed)
        run_seeds = np.random.randint(1, 100000, args.repeats)
        origin_results, golden_results, filter_results, relief_results = run_golden_rlf(
            train_data, valid_data, test_data, args, run_seeds)
        results = {
            "origin_results": origin_results,
            "golden_results": golden_results,
            "filter_results": filter_results,
            "relief_results": relief_results
        }
        filename = "golden_cmp.json"
        filepath = Path(args.output_path) / args.dataset / filename
        dirname = Path(args.output_path) / args.dataset
        Path(dirname).mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as write_file:
            json.dump(results, write_file, indent=4)

        origin_results = pd.DataFrame(origin_results)
        golden_results = pd.DataFrame(golden_results)
        filter_results = pd.DataFrame(filter_results)
        relief_results = pd.DataFrame(relief_results)

    x = np.arange(4)
    em_test = []
    trainset_accuracy = []
    trainset_coverage = []
    for results in [origin_results, golden_results, filter_results, relief_results]:
        em_test.append(results["em_test"].mean())
        trainset_accuracy.append(results["trainset_accuracy"].mean())
        trainset_coverage.append(results["trainset_coverage"].mean())

    width = 0.3
    plt.bar(x-width, em_test, width, color="cyan")
    plt.bar(x, trainset_accuracy, width, color="orange")
    plt.bar(x+width, trainset_coverage, width, color="green")
    plt.ylim((0, 1.05))
    plt.xticks(x, ["origin", "golden", "filter", "revise"])
    plt.legend(["test set accuracy", "train label accuracy", "train label coverage"], loc="lower right")
    plt.xlabel("groups")
    plt.ylabel("scores")
    ax = plt.gca()
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f")

    plt.title(args.dataset)
    figpath = Path(args.output_path) / args.dataset / "golden_cmp.jpg"
    plt.savefig(figpath)








