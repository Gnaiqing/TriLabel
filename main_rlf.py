import sys
import argparse
import time
from dataset.load_dataset import load_real_dataset, load_synthetic_dataset
from wrench.dataset import get_dataset_type
from labeller.labeller import get_labeller
from pathlib import Path
from torch.utils.data import TensorDataset
import torch
import json
import numpy as np
import pytorch_lightning as pl
from contrastive.mlp import MLP
from sklearn.metrics import accuracy_score
from utils import evaluate_performance, plot_results, save_results, evaluate_golden_performance, get_sampler, \
    get_label_model, get_reviser, ABSTAIN
import copy


def get_contrast_features(data_home, dataset, extract_fn, n_aug=2):
    features = []
    for i in range(n_aug):
        dataset_path = Path(data_home) / dataset
        dataset_class = get_dataset_type(dataset)
        aug_data = dataset_class(path=dataset_path, split=f"train_da{i}")
        aug_data.extract_feature(extract_fn=extract_fn, return_extractor=False, cache_name=extract_fn)
        features.append(aug_data.features)

    features = np.stack(features, axis=1)
    return features  # (n_data * n_views * n_feature)


def build_contrast_dataset(contrast_features, sampled_indices, sampled_labels, mode="all", golden_labels=None):
    """
    Build a dataset for contrastive learning
    :param contrast_features: (n_data * n_views * n_feature)
    :param sampled_indices: indices for sampled data
    :param sampled_labels: labels for sampled data
    :param mode: "all" or "labeled"
    :return:
    """
    if mode == "all":
        labels = - np.ones(contrast_features.shape[0],dtype=int)
        if sampled_indices is not None:
            labels[sampled_indices] = sampled_labels
        contrast_dataset = TensorDataset(torch.tensor(contrast_features), torch.tensor(labels))
    elif mode == "labeled":
        contrast_dataset = TensorDataset(torch.tensor(contrast_features[sampled_indices,:,:]),
                                         torch.tensor(sampled_labels))
    elif mode == "golden":
        contrast_dataset = TensorDataset(torch.tensor(contrast_features), torch.tensor(golden_labels))
    else:
        raise ValueError(f"Mode {mode} not supported for building contrastive dataset")
    return contrast_dataset


def update_results(results, perf, n_labeled, frac_labeled):
    results["n_labeled"].append(n_labeled)
    results["frac_labeled"].append(frac_labeled)
    for key in perf:
        results[key].append(perf[key])


def get_feature_encoder(train_data, sampled_indices, sampled_labels, args):
    # build dataset for contrastive learning
    if args.contrastive_mode is not None:
        contrast_features = get_contrast_features(args.dataset_path, args.dataset, args.extract_fn,
                                                  n_aug=args.n_aug)
        contrast_dataset = build_contrast_dataset(contrast_features, sampled_indices, sampled_labels,
                                                  mode=args.contrastive_mode, golden_labels=train_data.labels)
        dataloader = torch.utils.data.DataLoader(contrast_dataset, batch_size=args.batch_size)
        encoder = MLP(dim_in=contrast_features.shape[-1], dim_out=args.dim_out)
        trainer = pl.Trainer(max_epochs=args.max_epochs, fast_dev_run=False)
        trainer.fit(model=encoder, train_dataloaders=dataloader)
        encoder.eval()
    else:
        encoder = None
    return encoder


def run_rlf(train_data, valid_data, test_data, args, seed):
    """
    Run an active learning pipeline to revise label functions
    """
    start = time.process_time()
    results = {
        "n_labeled": [],
        "frac_labeled": [],
        "train_coverage": [],
        "train_covered_acc": [],
        "revised_frac": [],
        "lm_acc": [],  # label model's accuracy on revised fraction
        "revision_acc": [], # revision model's accuracy on revised fraction
        "em_test": [],
    }
    # set labeller, sampler, label_model, reviser and encoder
    labeller = get_labeller(args.labeller)
    label_model = get_label_model(args.label_model)
    if args.use_valid_labels:
        label_model.fit(dataset_train=train_data, dataset_valid=valid_data)
    else:
        label_model.fit(dataset_train=train_data)
    encoder = None
    reviser = get_reviser(args.revision_model, train_data, valid_data, encoder, args.device, seed)
    sampler = get_sampler(args.sampler, train_data, labeller, label_model, reviser, encoder)
    lm_probs = label_model.predict_proba(train_data)

    # record original stats
    perf = evaluate_performance(train_data, valid_data, test_data, lm_probs, args, seed)
    golden_perf = evaluate_golden_performance(train_data, valid_data, test_data, args, seed)
    update_results(results, perf, n_labeled=0, frac_labeled=0.0)
    results["revised_frac"].append(0)
    results["lm_acc"].append(np.nan)
    results["revision_acc"].append(np.nan)
    results["em_test_golden"] = golden_perf["em_test"]

    if args.verbose:
        lf_sum = train_data.lf_summary()
        print("Original LF summary:\n", lf_sum)
        print("Train coverage: ", perf["train_coverage"])
        print("Train covered acc: ", perf["train_covered_acc"])
        print("Test set acc: ", perf["em_test"])
        print("Golden test set acc: ", golden_perf["em_test"])

    # initialize revised train and valid data
    revised_train_data = copy.copy(train_data)
    revised_valid_data = copy.copy(valid_data)

    if args.sample_budget < 1:
        args.sample_budget = np.ceil(args.sample_budget * len(train_data)).astype(int)
        args.sample_per_iter = np.ceil(args.sample_per_iter * len(train_data)).astype(int)
    else:
        args.sample_budget = int(args.sample_budget)
        args.sample_per_iter = int(args.sample_per_iter)

    while sampler.get_n_sampled() < args.sample_budget:
        n_to_sample = min(args.sample_budget - sampler.get_n_sampled(), args.sample_per_iter)
        sampler.sample_distinct(n=n_to_sample)
        # train revision model using labeled data
        indices, labels = sampler.get_sampled_points()
        reviser.train_revision_model(indices, labels)

        # revise probabilistic labels or LFs
        lm_probs = label_model.predict_proba(revised_train_data)
        lm_preds = label_model.predict(revised_train_data)
        rm_probs = reviser.predict_proba(revised_train_data)
        lm_conf = np.max(lm_probs, axis=1)
        rm_conf = np.max(rm_probs, axis=1)
        revised_indices = rm_conf >= lm_conf
        aggregated_labels = lm_probs.copy()

        if args.revision_type in ["label", "both"]:
            # revise predicted labels based on RM predictions
            aggregated_labels[revised_indices] = rm_probs[revised_indices]

        if args.revision_type in ["LF", "both"]:
            # Let's first try only revise LF on confident points
            rm_preds = reviser.predict(revised_train_data)
            rm_preds[rm_conf < args.revision_threshold] = ABSTAIN
            revised_train_data = reviser.get_revised_dataset(revised_train_data, rm_preds)

            if args.revision_model != "expert-label":
                pred_valid = reviser.predict(revised_valid_data)
                probs_valid = reviser.predict_proba(revised_valid_data)
                conf_valid = np.max(probs_valid, axis=1)
                pred_valid[conf_valid < args.revision_threshold] = ABSTAIN
                revised_valid_data = reviser.get_revised_dataset(revised_valid_data, pred_valid)

            label_model.fit(revised_train_data, revised_valid_data)

        # update stats of sampler
        sampler.update_stats(revised_train_data,
                             label_model=label_model,
                             revision_model=reviser,
                             encoder=encoder)
        perf = evaluate_performance(revised_train_data, revised_valid_data, test_data, aggregated_labels, args, seed)
        n_labeled = sampler.get_n_sampled()
        frac_labeled = n_labeled / len(train_data)
        update_results(results, perf, n_labeled=n_labeled, frac_labeled=frac_labeled)
        # check the LM and RM's performance on revised data
        rm_preds = reviser.predict(revised_train_data)
        labels = np.array(revised_train_data.labels)
        if np.sum(revised_indices) == 0:
            rm_acc = np.nan
            lm_acc = np.nan
        else:
            rm_acc = accuracy_score(labels[revised_indices], rm_preds[revised_indices])
            lm_acc = accuracy_score(labels[revised_indices], lm_preds[revised_indices])
        revised_frac = np.sum(revised_indices) / len(revised_train_data)
        results["revised_frac"].append(revised_frac)
        results["lm_acc"].append(lm_acc)
        results["revision_acc"].append(rm_acc)
        if args.verbose:
            print(f"Sampled fraction: {frac_labeled:.2f}")
            print(f"Revised fraction: {revised_frac:.2f}")
            print(f"Reviser's accuracy on revised part: {rm_acc:.2f}")
            print(f"Label model's accuracy on revised part (before revision): {lm_acc:.2f}")
            print("Train coverage: ", perf["train_coverage"])
            print("Train covered acc: ", perf["train_covered_acc"])
            print("Test set acc: ", perf["em_test"])

    results["time"] = time.process_time() - start
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # device info
    parser.add_argument("--device", type=str, default="cuda:0")
    # dataset
    parser.add_argument("--dataset", type=str, default="youtube")
    parser.add_argument("--dataset_path", type=str, default="../datasets/")
    parser.add_argument("--extract_fn", type=str, default=None)  # method used to extract features
    # contrastive learning. If set to None, use original features.
    parser.add_argument("--contrastive_mode", type=str, default=None)
    # sampler
    parser.add_argument("--sampler", type=str, default="uncertain-joint")
    parser.add_argument("--sample_budget", type=float, default=0.10)  # Total sample budget
    parser.add_argument("--sample_per_iter",type=float, default=0.01)  # sample budget per iteration
    # revision model
    parser.add_argument("--revision_model", type=str, default="dalen")
    parser.add_argument("--revision_type", type=str, default="both", choices=["LF", "label", "both"])
    parser.add_argument("--revision_threshold", type=float, default=0.95)
    # label model and end models
    parser.add_argument("--label_model", type=str, default="metal")
    parser.add_argument("--end_model", type=str, default="mlp")
    parser.add_argument("--em_epochs", type=int, default=100)
    parser.add_argument("--use_soft_labels", action="store_true")
    # other settings
    parser.add_argument("--use_valid_labels", action="store_true")
    parser.add_argument("--labeller", type=str, default="oracle")
    parser.add_argument("--metric", type=str, default="acc")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tag", type=str, default="test")
    parser.add_argument("--load_results", action="store_true")
    # plot settings
    parser.add_argument("--plot_lf", action="store_true")  # plot LF accuracy and coverage over revision process
    parser.add_argument("--plot_tsne", action="store_true")  # plot density plots for samples
    parser.add_argument("--perplexity", type=float, default=20.0)
    args = parser.parse_args()
    if args.sample_budget < 1:
        plot_labeled_frac = True
    else:
        plot_labeled_frac = False
        args.sample_budget = int(args.sample_budget)
        args.sample_per_iter = int(args.sample_per_iter)

    if args.load_results:
        if args.rejection_cost is not None:
            reject_tag = f"{args.rejection_cost:.2f}"
        else:
            reject_tag = "adaptive"
        filepath = Path(args.output_path) / args.dataset / \
                   f"{args.label_model}_{args.end_model}_{args.revision_model}_{args.sampler}_{reject_tag}_{args.tag}.json"
        readfile = open(filepath, "r")
        results = json.load(readfile)
        results_list = results["data"]
        plot_results(results_list, args.output_path, args.dataset, args.dataset,
                     f"{args.label_model}_{args.end_model}_{args.revision_model}_{args.sampler}_{reject_tag}_{args.tag}",
                     plot_labeled_frac)
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
        results = run_rlf(train_data, valid_data, test_data, args, seed=run_seeds[i])
        args.plot_tsne = False  # only plot the first iteration
        results_list.append(results)

    save_results(results_list, args.output_path, args.dataset,
                 f"{args.label_model}_{args.end_model}_{args.revision_model}_{args.sampler}_{args.tag}.json")
