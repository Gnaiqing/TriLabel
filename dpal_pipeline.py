import sys
from pathlib import Path
import json
import argparse
import time
from dataset.load_dataset import load_real_dataset, load_synthetic_dataset
from labeller.labeller import get_labeller
import numpy as np
from sklearn.metrics import accuracy_score
from utils import evaluate_performance, save_results, evaluate_golden_performance, get_sampler, \
    get_label_model, get_reviser, ABSTAIN, evaluate_al_performance, plot_dpal_results



def update_results(results, n_labeled, frac_labeled,  al_label_acc, al_active_frac,
                   dp_active_acc, al_active_acc, dpal_label_acc, em_test, em_test_al):
    results["n_labeled"].append(n_labeled)
    results["frac_labeled"].append(frac_labeled)
    results["al_label_acc"].append(al_label_acc)
    results["al_active_frac"].append(al_active_frac)
    results["dp_active_acc"].append(dp_active_acc)
    results["al_active_acc"].append(al_active_acc)
    results["dpal_label_acc"].append(dpal_label_acc)
    results["em_test"].append(em_test)
    results["em_test_al"].append(em_test_al)


def run_dpal(train_data, valid_data, test_data, args, seed):
    """
    Run an active learning pipeline to revise label functions
    """
    start = time.process_time()
    results = {
        "n_labeled": [],  # number of expert labeled data
        "frac_labeled": [],  # fraction of expert labeled data
        "al_label_acc": [],  # AL label accuracy
        "al_active_frac": [],   # fraction of data following dp prediction
        "dp_active_acc": [], # DP label accuracy on active region
        "al_active_acc": [],  # AL label accuracy on active region
        "dpal_label_acc": [],  # DPAL label accuracy
        "em_test": [],  # end model's test performance
        "em_test_al": [],  # end model's test accuracy when using active learning
        "dp_label_acc": np.nan, # DP label accuracy
        "em_test_golden": np.nan  # end model's test performance using golden labels
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
    dp_label_acc = perf["train_covered_acc"]
    golden_perf = evaluate_golden_performance(train_data, valid_data, test_data, args, seed)
    results["dp_label_acc"] = perf["train_covered_acc"]
    results["em_test_golden"] = golden_perf["em_test"]
    update_results(results, n_labeled=0, frac_labeled=0.0, al_label_acc=0.0, al_active_frac=0.0,
                   dp_active_acc=perf["train_covered_acc"], al_active_acc=0.0,
                   dpal_label_acc=perf["train_covered_acc"], em_test=perf["em_test"], em_test_al=0.0)

    if args.verbose:
        print("Results at 0 (0%) labeled data:")
        print("DP label accuracy: ", perf["train_covered_acc"])
        print("AL label accuracy: ", 0.0)
        print("AL active fraction: ", 0.0)
        print("DPAL label accuracy: ", perf["train_covered_acc"])
        print("Test set acc: ", perf["em_test"])
        print("Golden test set acc: ", golden_perf["em_test"])

    while sampler.get_n_sampled() < args.sample_budget:
        n_to_sample = min(args.sample_budget - sampler.get_n_sampled(), args.sample_per_iter)
        sampler.sample_distinct(n=n_to_sample)
        # train revision model using labeled data
        indices, labels = sampler.get_sampled_points()
        reviser.train_revision_model(indices, labels)
        ground_truth_labels = np.repeat(ABSTAIN, len(train_data))
        ground_truth_labels[indices] = labels

        # revise probabilistic labels using active learning
        lm_probs = label_model.predict_proba(train_data)
        lm_preds = label_model.predict(train_data)
        lm_conf = np.max(lm_probs, axis=1)

        al_probs = reviser.predict_proba(train_data)
        al_preds = reviser.predict(train_data)
        al_conf = np.max(al_probs, axis=1)

        dp_active_indices = np.nonzero(lm_conf >= al_conf)[0]
        al_active_indices = np.nonzero(lm_conf < al_conf)[0]
        aggregated_labels = al_probs.copy()
        aggregated_labels[dp_active_indices, :] = lm_probs[dp_active_indices, :]

        golden_labels = np.array(train_data.labels)
        al_label_acc = accuracy_score(golden_labels, al_preds)
        al_active_acc = accuracy_score(golden_labels[al_active_indices], al_preds[al_active_indices])
        al_active_frac = len(al_active_indices) / len(train_data)
        dp_active_acc = accuracy_score(golden_labels[dp_active_indices], lm_preds[dp_active_indices])

        perf = evaluate_performance(train_data, valid_data, test_data, aggregated_labels, args, seed)
        al_perf = evaluate_al_performance(train_data, valid_data, test_data, args, seed, ground_truth_labels)
        n_labeled = sampler.get_n_sampled()
        frac_labeled = n_labeled / len(train_data)
        update_results(results, n_labeled=n_labeled, frac_labeled=frac_labeled, al_label_acc=al_label_acc,
                       al_active_frac=al_active_frac, dp_active_acc=dp_active_acc, al_active_acc=al_active_acc,
                       dpal_label_acc=perf["train_covered_acc"], em_test=perf["em_test"], em_test_al=al_perf["em_test"])
        if args.verbose:
            print(f"Results at {n_labeled} ({frac_labeled* 100:.1f}%) labeled data:")
            print("DP label accuracy: ", dp_label_acc)
            print("AL label accuracy: ", al_label_acc)
            print("AL active fraction: ", al_active_frac)
            print("DPAL label accuracy: ", perf["train_covered_acc"])
            print("AL test set acc: ", al_perf["em_test"])
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
    # sampler
    parser.add_argument("--sampler", type=str, default="passive")
    parser.add_argument("--sample_budget", type=float, default=0.10)  # Total sample budget
    parser.add_argument("--sample_per_iter",type=float, default=0.01)  # sample budget per iteration
    # revision model
    parser.add_argument("--revision_model", type=str, default="ensemble")
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
    args = parser.parse_args()
    if args.load_results:
        id_tag = f"dpal_{args.label_model}_{args.end_model}_{args.revision_model}_{args.sampler}_{args.tag}"
        filepath = Path(args.output_path) / args.dataset / f"{id_tag}.json"
        readfile = open(filepath, "r")
        results = json.load(readfile)
        results_list = results["data"]
        plot_labeled_frac = args.sample_budget < 1
        plot_dpal_results(results_list, args.output_path, args.dataset, id_tag, plot_labeled_frac)
        sys.exit(0)

    if args.dataset[:3] == "syn":
        train_data, valid_data, test_data = load_synthetic_dataset(args.dataset)
    else:
        train_data, valid_data, test_data = load_real_dataset(args.dataset_path, args.dataset, args.extract_fn)

    if args.sample_budget < 1:
        plot_labeled_frac = True
        args.sample_budget = np.ceil(args.sample_budget * len(train_data)).astype(int)
        args.sample_per_iter = np.ceil(args.sample_per_iter * len(train_data)).astype(int)
    else:
        plot_labeled_frac = False
        args.sample_budget = int(args.sample_budget)
        args.sample_per_iter = int(args.sample_per_iter)

    np.random.seed(args.seed)
    run_seeds = np.random.randint(1, 100000, args.repeats)

    results_list = []
    for i in range(args.repeats):
        print(f"Start run {i}")
        results = run_dpal(train_data, valid_data, test_data, args, seed=run_seeds[i])
        results_list.append(results)

    id_tag = f"dpal_{args.label_model}_{args.end_model}_{args.revision_model}_{args.sampler}_{args.tag}"

    save_results(results_list, args.output_path, args.dataset,
                 f"{id_tag}.json")
    plot_dpal_results(results_list, args.output_path, args.dataset,id_tag, plot_labeled_frac)


