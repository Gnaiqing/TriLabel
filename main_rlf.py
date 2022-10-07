import sys
import argparse
from dataset.load_dataset import load_real_dataset, load_synthetic_dataset
from wrench.dataset import get_dataset_type
from sampler.passive import PassiveSampler
from labeller.labeller import get_labeller
from reviser.relief import LFReviser
from pathlib import Path
from torch.utils.data import TensorDataset
import torch
import json
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from contrastive.mlp import MLP
from utils import evaluate_performance, plot_tsne, plot_results, save_results, evaluate_golden_performance


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


def update_results(results, perf, n_labeled):
    results["labeled"].append(n_labeled)
    for key in perf:
        results[key].append(perf[key])


def run_rlf(train_data, valid_data, test_data, args, seed):
    """
    Run an active learning pipeline to revise label functions
    """
    # enable reproducibility for this run
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
    perf = evaluate_performance(train_data, valid_data, test_data, args, seed=seed)
    golden_perf = evaluate_golden_performance(train_data, valid_data, test_data, args, seed=seed)
    results["em_test_golden"] = golden_perf["em_test"]
    update_results(results, perf, 0)
    n_labeled = 0
    if args.plot_tsne:
        plot_tsne(train_data.features, train_data.labels, args.output_path, args.dataset,
                  f"{args.dataset}_l=0_{args.tag}", perplexity=args.perplexity)
    # initialize labeled set
    labeller = get_labeller(args.labeller)
    init_sampler = PassiveSampler(train_data, labeller, seed=seed)
    init_indices, init_labels = init_sampler.sample_distinct(n=args.sample_budget_init)
    # print("Init indices: \n", init_indices[:10])
    n_labeled += args.sample_budget_init
    # build dataset for contrastive learning
    if args.contrastive_mode is not None:
        contrast_features = get_contrast_features(args.dataset_path, args.dataset, args.extract_fn, n_aug=args.n_aug)
        contrast_dataset = build_contrast_dataset(contrast_features, init_indices, init_labels,
                                                  mode=args.contrastive_mode, golden_labels=train_data.labels)
        # train encoder with contrastive loss
        dataloader = torch.utils.data.DataLoader(contrast_dataset, batch_size=args.batch_size)
        encoder = MLP(dim_in=contrast_features.shape[-1], dim_out=args.dim_out)
        trainer = pl.Trainer(max_epochs=args.max_epochs, fast_dev_run=False)
        trainer.fit(model=encoder, train_dataloaders=dataloader)
        encoder.eval()
        # plot tsne
        if args.plot_tsne:
            features = encoder(torch.tensor(train_data.features)).detach().cpu().numpy()
            plot_tsne(features, train_data.labels, args.output_path, args.dataset,
                      f"{args.dataset}_l={n_labeled}_{args.tag}", perplexity=args.perplexity)
    else:
        encoder = None

    # revise dataset
    reviser = LFReviser(train_data, encoder, args.revision_model, seed=seed)
    reviser.train_revision_models(init_indices, init_labels)
    revised_train_data = reviser.revised_dataset
    revised_valid_data = reviser.get_revised_dataset(valid_data)
    revised_test_data = reviser.get_revised_dataset(test_data)
    lf_sum = revised_train_data.lf_summary()
    print(f"Revised LF summary at {n_labeled} labeled:\n", lf_sum)
    perf = evaluate_performance(revised_train_data, revised_valid_data, revised_test_data, args, seed=seed)
    update_results(results, perf, n_labeled)
    # spend remaining labelling budget
    active_sampler = PassiveSampler(revised_train_data, labeller,
                                    sampled_indices=init_indices, sampled_labels=init_labels, seed=seed*2)
    while n_labeled < args.sample_budget:
        n_to_sample = min(args.sample_budget - n_labeled, args.sample_budget_inc)
        active_sampler.sample_distinct(n=n_to_sample)
        n_labeled += n_to_sample
        sampled_indices, sampled_labels = active_sampler.get_sampled_points()
        if args.contrastive_mode is not None:
            contrast_dataset = build_contrast_dataset(contrast_features, sampled_indices, sampled_labels,
                                                      mode=args.contrastive_mode, golden_labels=train_data.labels)
            # train encoder with contrastive loss
            dataloader = torch.utils.data.DataLoader(contrast_dataset, batch_size=args.batch_size)
            encoder = MLP(dim_in=contrast_features.shape[-1], dim_out=args.dim_out)
            trainer = pl.Trainer(max_epochs=args.max_epochs, fast_dev_run=False)
            trainer.fit(model=encoder, train_dataloaders=dataloader)
            encoder.eval()
            # plot tsne
            if args.plot_tsne:
                features = encoder(torch.tensor(train_data.features)).detach().cpu().numpy()
                plot_tsne(features, train_data.labels, args.output_path, args.dataset,
                          f"{args.dataset}_l={n_labeled}_{args.tag}", perplexity=args.perplexity)
        else:
            encoder = None

        # revise dataset
        reviser = LFReviser(revised_train_data, encoder, args.revision_model, seed=seed)
        reviser.train_revision_models(sampled_indices, sampled_labels)
        revised_train_data = reviser.revised_dataset
        revised_valid_data = reviser.get_revised_dataset(revised_valid_data)
        revised_test_data = reviser.get_revised_dataset(revised_test_data)
        lf_sum = revised_train_data.lf_summary()
        print(f"Revised LF summary at {n_labeled} labeled:\n", lf_sum)
        perf = evaluate_performance(revised_train_data, revised_valid_data, revised_test_data, args, seed=seed)
        update_results(results, perf, n_labeled)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # device info
    parser.add_argument("--device", type=str, default="cuda")
    # dataset
    parser.add_argument("--dataset", type=str, default="youtube")
    parser.add_argument("--dataset_path", type=str, default="../wrench-1.1/datasets/")
    parser.add_argument("--extract_fn", type=str, default="bert")  # method used to extract features
    # active learning baseline without using label model
    parser.add_argument("--active_learning_only", action="store_true")
    # contrastive learning
    parser.add_argument("--contrastive_mode", type=str, default=None)
    parser.add_argument("--data_augment", type=str, default="eda")
    parser.add_argument("--n_aug", type=int, default=2)
    parser.add_argument("--dim_out", type=int, default=128)
    parser.add_argument("--batch_size",type=int, default=64)
    parser.add_argument("--max_epochs",type=int, default=10)
    # sampler
    parser.add_argument("--sampler", type=str, nargs="+", default="passive")
    parser.add_argument("--sample_budget", type=int, default=500)  # Total sample budget
    parser.add_argument("--sample_budget_init",type=int, default=100)  # sample budget for initialization
    parser.add_argument("--sample_budget_inc", type=int, default=100)  # increased sample budget per iteration
    # revision model
    parser.add_argument("--revision_method", type=str, default="relief") # "nashaat": only revise labeled points
    parser.add_argument("--revision_model", type=str, default="logistic")
    # label model and end model
    parser.add_argument("--label_model", type=str, default="mv")
    parser.add_argument("--end_model", type=str, default="mlp")
    parser.add_argument("--em_epochs", type=int, default=100)
    parser.add_argument("--em_batch_size", type=int, default=256)
    parser.add_argument("--em_patience", type=int, default=10)
    parser.add_argument("--em_lr", type=float, default=0.01)
    parser.add_argument("--em_weight_decay", type=float, default=0.0001)
    parser.add_argument("--use_soft_labels", action="store_true")
    # other settings
    parser.add_argument("--labeller", type=str, default="oracle")
    parser.add_argument("--metric", type=str, default="acc")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tag", type=str, default="0")
    parser.add_argument("--load_results", type=str, default=None)
    # plot settings
    parser.add_argument("--plot_lf", action="store_true")  # plot LF accuracy and coverage over revision process
    parser.add_argument("--plot_tsne", action="store_true")  # plot density plots for samples
    parser.add_argument("--perplexity", type=float, default=20.0)
    args = parser.parse_args()
    if args.load_results is not None:
        readfile = open(args.load_results, "r")
        results = json.load(readfile)
        results_list = results["data"]
        plot_results(results_list, args.output_path, args.dataset, f"{args.dataset}_{args.tag}", args.metric)
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
        results_list.append(results)

    save_results(results_list, args.output_path, args.dataset, f"{args.tag}.json")
    plot_results(results_list, args.output_path, args.dataset, f"{args.dataset}_{args.tag}", args.metric)
