import sys
import argparse
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
from utils import evaluate_performance, plot_results, save_results, evaluate_golden_performance, get_sampler, \
    get_label_model, get_reviser
from sklearn.metrics import accuracy_score
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
    results = {
        "n_labeled": [],
        "frac_labeled": [],
        "train_coverage": [],
        "train_covered_acc": [],
        "rm_coverage": [],
        "rm_covered_acc": [],
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

    # initialize revised train and valid data
    revised_train_data = copy.copy(train_data)
    revised_valid_data = copy.copy(valid_data)

    if args.sample_budget < 1:
        args.sample_budget = np.ceil(args.sample_budget * len(train_data)).astype(int)
        args.sample_per_iter = np.ceil(args.sample_per_iter * len(train_data)).astype(int)

    while sampler.get_n_sampled() < args.sample_budget:
        n_to_sample = min(args.sample_budget - sampler.get_n_sampled(), args.sample_per_iter)
        sampler.sample_distinct(n=n_to_sample)
        indices, labels = sampler.get_sampled_points()
        # estimate current accuracy of label model using valid labels or sampled train labels
        if args.use_valid_labels:
            lm_pred = label_model.predict(revised_valid_data)
            lm_acc_hat = accuracy_score(revised_valid_data.labels, lm_pred)
        else:
            lm_pred = label_model.predict(revised_train_data)
            lm_acc_hat = accuracy_score(labels, lm_pred[indices])

        if args.rejection_cost is None:
            cost = 1 - lm_acc_hat
        else:
            cost = args.rejection_cost

        print(f"Set cost to {cost:.2f}")
        # train revision model
        reviser.train_revision_model(indices, labels, cost=cost)
        y_hat_train = reviser.predict_labels(reviser.train_data, cost)
        if args.revision_type in ["LF", "both"]:
            # revise label functions
            revised_train_data = reviser.get_revised_dataset("train", cost)
            revised_valid_data = reviser.get_revised_dataset("valid", cost)

        perf = evaluate_performance(revised_train_data, revised_valid_data, test_data, args,
                                    rm_predict_labels=y_hat_train,
                                    seed=seed)

        n_labeled = sampler.get_n_sampled()
        frac_labeled = n_labeled / len(train_data)
        update_results(results, perf, n_labeled=n_labeled, frac_labeled=frac_labeled)
        # update stats of label model, sampler and reviser
        label_model.fit(dataset_train=revised_train_data, dataset_valid=revised_valid_data)
        sampler.update_stats(revised_train_data,
                             label_model=label_model,
                             revision_model=reviser,
                             encoder=encoder)

        if args.verbose:
            lf_sum = revised_train_data.lf_summary()
            print(f"Revised LF summary at {n_labeled}({frac_labeled*100:.1f}%):")
            print(lf_sum)
            print("Train coverage: ", perf["train_coverage"])
            print("Train covered acc: ", perf["train_covered_acc"])
            print("Revision model coverage: ", perf["rm_coverage"])
            print("Revision model acc: ", perf["rm_covered_acc"])
            print("Test set acc: ", perf["em_test"])

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # device info
    parser.add_argument("--device", type=str, default="cuda:1")
    # dataset
    parser.add_argument("--dataset", type=str, default="youtube")
    parser.add_argument("--dataset_path", type=str, default="../wrench-1.1/datasets/")
    parser.add_argument("--extract_fn", type=str, default=None)  # method used to extract features
    # contrastive learning. If set to None, use original features.
    parser.add_argument("--contrastive_mode", type=str, default=None)
    # rejection cost. If set to None, use adaptive cost.
    parser.add_argument("--rejection_cost", type=float, default=None)
    # sampler
    parser.add_argument("--sampler", type=str, default="passive")
    parser.add_argument("--sample_budget", type=float, default=0.05)  # Total sample budget
    parser.add_argument("--sample_per_iter",type=float, default=0.01)  # sample budget per iteration
    # revision model
    parser.add_argument("--revision_model", type=str, default="mlp")
    parser.add_argument("--revision_type", type=str, default="both", choices=["LF", "label", "both"])
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
        filepath = Path(args.output_path) / args.dataset / \
                   f"{args.label_model}-{args.end_model}-{args.revision_model}-{args.sampler}_{args.tag}.json"
        readfile = open(filepath, "r")
        results = json.load(readfile)
        results_list = results["data"]
        plot_results(results_list, args.output_path, args.dataset, args.dataset,
                     f"{args.label_model}-{args.end_model}-nashaat_{args.tag}.jpg",
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

    if args.rejection_cost is not None:
        reject_tag = f"{args.rejection_cost:.2f}"
    else:
        reject_tag = "adaptive"

    save_results(results_list, args.output_path, args.dataset,
                 f"{args.label_model}-{args.end_model}-{args.revision_model}-{args.sampler}-{reject_tag}_{args.tag}.json")
    plot_results(results_list, args.output_path, args.dataset, args.dataset,
                 f"{args.label_model}-{args.end_model}-{args.revision_model}-{args.sampler}-{reject_tag}_{args.tag}",
                 plot_labeled_frac)
