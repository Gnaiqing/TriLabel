"""
Evaluate the classifier accuracy when revising LF by training classifiers with all golden labels
"""
import argparse
from dataset.load_dataset import load_real_dataset, load_synthetic_dataset
from wrench.dataset import get_dataset_type
from labeller.labeller import get_labeller
from reviser.relief import LFReviser
from pathlib import Path
from torch.utils.data import TensorDataset
import torch
import json
import copy
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from contrastive.mlp import MLP
from utils import evaluate_performance, plot_tsne, plot_results, save_results, evaluate_golden_performance, get_sampler
from main_rlf import update_results, get_feature_encoder


def run_golden_rlf(train_data, valid_data, test_data, args, seed):
    seed_everything(seed, workers=True)
    results = {
        "labeled": [],
        "train_coverage": [],
        "train_covered_acc": [],
        "test_coverage": [],
        "test_covered_acc": [],
        "lm_test": [],
        "em_test": [],
    }
    lf_sum = train_data.lf_summary()
    print("Original LF summary:\n", lf_sum)
    lf_val_sum = valid_data.lf_summary()
    print("Original valid LF summary:\n", lf_val_sum)
    lf_test_sum = test_data.lf_summary()
    print("Original test LF summary:\n", lf_test_sum)
    perf = evaluate_performance(train_data, valid_data, test_data, args, seed=seed)
    golden_perf = evaluate_golden_performance(train_data, valid_data, test_data, args, seed=seed)
    results["em_test_golden"] = golden_perf["em_test"]
    update_results(results, perf, 0)
    indices = np.arange(len(train_data))
    labels = np.array(train_data.labels)
    encoder = get_feature_encoder(train_data, indices, labels, args)
    reviser = LFReviser(train_data, encoder, args.lf_class, args.revision_model_class,
                        valid_data=valid_data, seed=seed)
    reviser.revise_label_functions(indices, labels)
    revised_train_data = reviser.get_revised_dataset(dataset=train_data)
    revised_valid_data = reviser.get_revised_dataset(dataset=valid_data)
    revised_test_data = reviser.get_revised_dataset(dataset=test_data)
    lf_sum = revised_train_data.lf_summary()
    print(f"Revised LF summary:\n", lf_sum)
    perf = evaluate_performance(revised_train_data, revised_valid_data, revised_test_data, args, seed=seed)
    update_results(results, perf, len(train_data))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # device info
    parser.add_argument("--device", type=str, default="cuda:1")
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
    parser.add_argument("--em_batch_size", type=int, default=256)
    parser.add_argument("--em_patience", type=int, default=100)
    parser.add_argument("--em_lr", type=float, default=0.01)
    parser.add_argument("--em_weight_decay", type=float, default=0.0001)
    parser.add_argument("--use_soft_labels", action="store_true")
    # other settings
    parser.add_argument("--labeller", type=str, default="oracle")
    parser.add_argument("--metric", type=str, default="acc")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--tag", type=str, default="GOLD")
    args = parser.parse_args()
    if args.dataset[:3] == "syn":
        train_data, valid_data, test_data = load_synthetic_dataset(args.dataset)
    else:
        train_data, valid_data, test_data = load_real_dataset(args.dataset_path, args.dataset, args.extract_fn)

    np.random.seed(args.seed)
    run_seeds = np.random.randint(1, 100000, args.repeats)

    results_list = []
    for i in range(args.repeats):
        print(f"Start run {i}")
        results = run_golden_rlf(train_data, valid_data, test_data, args, seed=run_seeds[i])
        results_list.append(results)

    save_results(results_list, args.output_path, args.dataset, f"{args.tag}.json")
    plot_results(results_list, args.output_path, args.dataset, f"{args.dataset}_{args.tag}", args.metric)

