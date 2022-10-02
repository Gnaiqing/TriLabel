import argparse
from dataset.load_dataset import load_real_dataset, load_synthetic_dataset
from utils import get_end_model
from numpy.random import default_rng
import numpy as np
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


def simulate_label_process(train_data, valid_data, test_data, args, seed=42):
    n_class = train_data.n_class
    results = {
        "lm_coverage": [],
        "lm_accuracy": [],
        "em_test": []
    }
    for lm_coverage in np.linspace(0.6, 1.0, 9):
        rng = default_rng(seed=seed)
        n_covered = int(len(train_data) * lm_coverage)
        covered_indices = rng.choice(len(train_data), n_covered, replace=False).tolist()
        covered_train_data = train_data.create_subset(covered_indices)
        covered_labels = covered_train_data.labels
        for lm_accuracy in np.linspace(0.6, 1.0, 9):
            lm_labels = np.array(covered_labels)
            n_flip = int(len(covered_train_data) * (1-lm_accuracy))
            flip_indices = rng.choice(len(covered_train_data), n_flip, replace=False)

            for idx in flip_indices:
                candidate_labels = [ i for i in np.arange(n_class) if i != covered_labels[idx]]
                flip_label = np.random.choice(candidate_labels)
                lm_labels[idx] = flip_label

            end_model = get_end_model(args.end_model)
            em_test_list = []
            for i in range(args.em_repeats):
                end_model.fit(
                    dataset_train=covered_train_data,
                    y_train=lm_labels,
                    dataset_valid=valid_data,
                    evaluation_step=100,
                    metric=args.metric,
                    patience=500,
                    device=args.device,
                    verbose=True
                )
                em_test = end_model.test(test_data, args.metric, device=args.device)
                em_test_list.append(em_test)
            em_test = np.mean(em_test_list)
            results["lm_accuracy"].append(lm_accuracy)
            results["lm_coverage"].append(lm_coverage)
            results["em_test"].append(em_test)

    return results


def print_cov_acc(results, figpath):
    results = pd.DataFrame(results)
    plt.figure()
    for lm_coverage in np.linspace(0.6,1.0,9):
        cur_results = results[results["lm_coverage"] == lm_coverage]
        plt.plot(cur_results["lm_accuracy"], cur_results["em_test"], label=f"cov={lm_coverage}")

    plt.xlabel("train set accuracy")
    plt.ylabel("end model test accuracy")
    plt.legend()
    plt.savefig(figpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # device info
    parser.add_argument("--device", type=str, default="cuda")
    # dataset
    parser.add_argument("--dataset", type=str, default="youtube")
    parser.add_argument("--dataset_path", type=str, default="../wrench-1.1/datasets/")
    parser.add_argument("--extract_fn", type=str, default="bert")  # method used to extract features
    # end model
    parser.add_argument("--end_model", type=str, default="mlp")
    parser.add_argument("--em_repeats", type=int, default=5)
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--metric", type=str, default="acc")
    parser.add_argument("--save_plot", action="store_true")
    args = parser.parse_args()
    if args.dataset[:3] == "syn":
        train_data, valid_data, test_data = load_synthetic_dataset(args.dataset)
    else:
        train_data, valid_data, test_data = load_real_dataset(args.dataset_path, args.dataset, args.extract_fn)

    savefile = Path(args.output_path) / args.dataset / "cov_acc.json"
    if args.save_plot:
        readfile = open(savefile, "r")
        results = json.load(readfile)
        figpath = savefile = Path(args.output_path) / args.dataset / "cov_acc.jpg"
        print_cov_acc(results, figpath)

    else:
        results = simulate_label_process(train_data, valid_data, test_data, args)
        with open(savefile, "w") as write_file:
            json.dump(results, write_file, indent=4)



