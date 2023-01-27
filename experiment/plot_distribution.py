import argparse
from utils import preprocess_data
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_lf_stats(dataset, train_lf_summary, valid_lf_summary, test_lf_summary, metric, output_path):
    train_stats = train_lf_summary[metric]
    valid_stats = valid_lf_summary[metric]
    test_stats = test_lf_summary[metric]
    plt.figure()
    width = 0.2
    x = np.arange(len(train_stats))
    plt.bar(x-width, train_stats, width)
    plt.bar(x, valid_stats, width)
    plt.bar(x+width, test_stats, width)
    plt.xticks(x, x)
    plt.xlabel("LF")
    plt.ylabel(metric)
    plt.legend(["train", "valid", "test"])
    plt.title(dataset)
    figpath = Path(output_path) / dataset / f"LF_{metric}.jpg"
    plt.savefig(figpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--dataset", type=str, default="youtube")
    parser.add_argument("--dataset_path", type=str, default="../datasets/")
    parser.add_argument("--extract_fn", type=str, default="bert")  # method used to extract features
    parser.add_argument("--max_dim", type=int, default=None)
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--sample_budget", type=float, default=0.0)
    parser.add_argument("--sample_per_iter", type=float, default=0.0)
    args = parser.parse_args()
    train_data, valid_data, test_data = preprocess_data(args)
    train_lf_summary = train_data.lf_summary()
    valid_lf_summary = valid_data.lf_summary()
    test_lf_summary = test_data.lf_summary()
    print(train_lf_summary)
    print(valid_lf_summary)
    print(test_lf_summary)
    if len(train_lf_summary) > 10:
        # sort LF based on coverage and select LFs with highest coverage
        lf_order = np.argsort(train_lf_summary["Coverage"].to_numpy())[-1:-10:-1]
        train_lf_summary = train_lf_summary.iloc[lf_order,:]
        valid_lf_summary = valid_lf_summary.iloc[lf_order,:]
        test_lf_summary = test_lf_summary.iloc[lf_order,:]

    plot_lf_stats(args.dataset, train_lf_summary, valid_lf_summary, test_lf_summary, "Emp. Acc.", args.output_path)
    plot_lf_stats(args.dataset, train_lf_summary, valid_lf_summary, test_lf_summary, "Coverage", args.output_path)
