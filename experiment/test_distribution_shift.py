import argparse
from utils import preprocess_data, ABSTAIN
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def get_active_LF_summary(dataset):
    """
    Summarize the dataset based on active LF number of each data point
    """
    weak_labels = np.array(dataset.weak_labels)
    n_active = np.sum(weak_labels != ABSTAIN, axis=1)
    n_active_summary = np.bincount(n_active) / len(n_active)
    return n_active_summary

def get_LF_quality_summary(dataset):
    """
    Summarize the dataset based on n_correct_LF / n_active_LF per data point
    """
    weak_labels = np.array(dataset.weak_labels)
    y = np.array(dataset.labels)
    n_active = np.sum(weak_labels != ABSTAIN, axis=1).astype(float)
    n_correct = np.sum(weak_labels == (y.reshape(-1,1)), axis=1).astype(float)
    score = np.divide(n_correct, n_active, out=np.zeros_like(n_active), where=n_active!=0)
    hist, bin_edges = np.histogram(score, bins=10, range=(0,1))
    hist = hist / np.sum(hist)
    return hist


def plot_dist(dataset, train_sum, valid_sum, test_sum, filepath):
    n_bar = max(len(train_sum), len(valid_sum), len(test_sum))
    epsilon = 1e-6 # prevent division by zero
    train_sum = np.pad(train_sum, (0, n_bar-len(train_sum)), mode="constant")
    train_sum = (train_sum + epsilon) / (1 + epsilon * n_bar)
    valid_sum = np.pad(valid_sum, (0, n_bar-len(valid_sum)), mode="constant")
    valid_sum = (valid_sum + epsilon) / (1 + epsilon * n_bar)
    test_sum = np.pad(test_sum, (0, n_bar-len(test_sum)), mode="constant")
    test_sum = (test_sum + epsilon) / (1 + epsilon * n_bar)
    x = np.arange(n_bar)
    width = 0.2
    fig, ax = plt.subplots()
    ax.bar(x - width, train_sum, width, color='blue')
    ax.bar(x, valid_sum, width, color='orange')
    ax.bar(x + width, test_sum, width, color='green')
    psi_train_val = (valid_sum - train_sum) * np.log(valid_sum/train_sum)
    psi_train_val = np.sum(psi_train_val)
    psi_train_test = (test_sum - train_sum) * np.log(test_sum/train_sum)
    psi_train_test = np.sum(psi_train_test)
    ax.set_xticks(x, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    ax.legend(["Train", "Valid", "Test"])
    fig.suptitle(f"{dataset}", fontsize=14)
    ax.set_title(f"PSI: {psi_train_val:.2f}(Val) {psi_train_test:.2f}(Test)")
    plt.savefig(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--dataset", type=str, default="youtube")
    parser.add_argument("--dataset_path", type=str, default="../datasets/")
    parser.add_argument("--max_dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extract_fn", type=str, default="bert")  # method used to extract features
    parser.add_argument("--output_path", type=str, default="output/")
    args = parser.parse_args()
    train_data, valid_data, test_data = preprocess_data(args)
    train_lf_summary = train_data.lf_summary()
    valid_lf_summary = valid_data.lf_summary()
    test_lf_summary = test_data.lf_summary()
    print("Train Summary: \n", train_lf_summary)
    print("Valid Summary: \n", valid_lf_summary)
    print("Test  Summary: \n", test_lf_summary)
    train_sum = get_LF_quality_summary(train_data)
    valid_sum = get_LF_quality_summary(valid_data)
    test_sum = get_LF_quality_summary(test_data)
    filepath = Path(args.output_path) / args.dataset / "dist.jpg"
    plot_dist(args.dataset, train_sum, valid_sum, test_sum, filepath)


