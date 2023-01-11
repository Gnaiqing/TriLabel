import sys
from pathlib import Path
import json
import argparse
import time
from dataset.load_dataset import load_real_dataset, load_synthetic_dataset
from labeller.labeller import get_labeller
import numpy as np
from calibrator.lm_ensemble import EnsembleCalibrator
from sklearn.metrics import accuracy_score
from utils import evaluate_performance, save_results, evaluate_golden_performance, get_sampler, \
     get_reviser, ABSTAIN, plot_dpal_results, evaluate_label_quality


def update_results(results, n_labeled, frac_labeled,
                   dpal_label_acc, dpal_label_nll, dpal_label_brier,
                   al_label_acc, al_label_nll, al_label_brier,
                   concensus_frac, al_active_frac, dp_active_frac,
                   dpal_test_acc, dpal_test_f1):
    results["n_labeled"].append(n_labeled)
    results["frac_labeled"].append(frac_labeled)

    results["dpal_label_acc"].append(dpal_label_acc)
    results["dpal_label_nll"].append(dpal_label_nll)
    results["dpal_label_brier"].append(dpal_label_brier)

    results["al_label_acc"].append(al_label_acc)
    results["al_label_nll"].append(al_label_nll)
    results["al_label_brier"].append(al_label_brier)

    results["concensus_frac"].append(concensus_frac)
    results["al_active_frac"].append(al_active_frac)
    results["dp_active_frac"].append(dp_active_frac)

    results["dpal_test_acc"].append(dpal_test_acc)
    results["dpal_test_f1"].append(dpal_test_f1)


def run_dpal(train_data, valid_data, test_data, args, seed):
    """
    Run an active learning pipeline to revise label functions
    """
    start = time.process_time()
    results = {
        "n_labeled": [],  # number of expert labeled data
        "frac_labeled": [],  # fraction of expert labeled data
        "dpal_label_acc": [],  # DPAL label accuracy
        "dpal_label_nll": [],  # DPAL label NLL score
        "dpal_label_brier": [],  # DPAL label brier score
        "al_label_acc": [],  # AL label accuracy
        "al_label_nll": [],  # AL label NLL score
        "al_label_brier": [],  # AL label brier score
        "dp_label_acc": np.nan,  # DP label accuracy
        "dp_label_nll": np.nan,  # DP label NLL score
        "dp_label_brier": np.nan,  # DP label brier score
        "concensus_frac": [],  # fraction of data where prediction result follow concensus of DP and AL
        "al_active_frac": [],  # fraction of data where prediction result follow AL but not DP
        "dp_active_frac": [],  # fraction of data where prediction result follow DP but not AL
        "dpal_test_acc": [],  # end model's test accuracy
        "dpal_test_f1": [],  # end model's test f1
        "al_test_acc": [],  # test accuracy using active learning
        "al_test_f1": [],  # test f1 score using active learning
        "golden_test_acc": np.nan,  # end model's test accuracy using golden labels
        "golden_test_f1": np.nan   # end model's test f1 using golden labels
    }
    # set labeller, sampler, label_model, reviser and encoder
    labeller = get_labeller(args.labeller)
    lm_calib = EnsembleCalibrator(train_data, valid_data, args.label_model, args.lm_ensemble_size, args.LF_selected_size,
                       args.bootstrap, args.seed)
    encoder = None
    reviser = get_reviser(args.revision_model, train_data, valid_data, encoder, args.device, seed)
    sampler = get_sampler(args.sampler, train_data, labeller, lm_calib, reviser, encoder, seed=seed)

    lm_probs = lm_calib.predict_proba(train_data)
    acc, nll, brier = evaluate_label_quality(train_data.labels, lm_probs)
    dp_label_acc = acc # record initial label accuracy of DP
    # record original stats
    perf = evaluate_performance(train_data, valid_data, test_data, lm_probs, args, seed)
    golden_perf = evaluate_golden_performance(train_data, valid_data, test_data, args, seed)
    results["dp_label_acc"] = acc
    results["dp_label_nll"] = nll
    results["dp_label_brier"] = brier
    results["golden_test_acc"] = golden_perf["test_acc"]
    results["golden_test_f1"] = golden_perf["test_f1"]
    update_results(results, n_labeled=0, frac_labeled=0.0,
                   dpal_label_acc=acc, dpal_label_nll=nll, dpal_label_brier=brier,
                   al_label_acc=np.nan, al_label_nll=np.nan, al_label_brier=np.nan,
                   concensus_frac=0.0, al_active_frac=0.0, dp_active_frac=1.0,
                   dpal_test_acc=perf["test_acc"], dpal_test_f1=perf["test_f1"])

    if args.verbose:
        print("Results at 0 (0%) labeled data:")
        print("DPAL label accuracy: ", acc)
        print("DP label accuracy: ", acc)
        print("AL label accuracy: ", 0.0)
        print("Concensus frac: ", 0.0)
        print("DP active frac: ", 1.0)
        print("AL active frac: ", 0.0)
        print("Test set acc: ", perf["test_acc"])
        print("Golden test set acc: ", golden_perf["test_acc"])

    while sampler.get_n_sampled() < args.sample_budget:
        n_to_sample = min(args.sample_budget - sampler.get_n_sampled(), args.sample_per_iter)
        sampler.sample_distinct(n=n_to_sample)
        # train revision model using labeled data
        indices, labels = sampler.get_sampled_points()
        reviser.train_revision_model(indices, labels)
        ground_truth_labels = np.repeat(ABSTAIN, len(train_data))
        ground_truth_labels[indices] = labels

        # revise probabilistic labels using active learning
        lm_probs = lm_calib.predict_proba(train_data)
        lm_preds = lm_calib.predict(train_data)
        lm_conf = np.max(lm_probs, axis=1)

        al_probs = reviser.predict_proba(train_data)
        al_preds = reviser.predict(train_data)
        al_conf = np.max(al_probs, axis=1)

        if args.aggregation_method == "confidence":
            dp_active_indices = np.nonzero(lm_conf >= al_conf)[0]
            aggregated_labels = al_probs.copy()
            aggregated_labels[dp_active_indices, :] = lm_probs[dp_active_indices, :]
        elif args.aggregation_method == "average":
            aggregated_labels = (lm_probs + al_probs) / 2
        elif args.aggregation_method == "weighted":
            lm_val_preds = lm_calib.predict(valid_data)
            lm_val_acc = accuracy_score(valid_data.labels, lm_val_preds)
            al_val_preds = reviser.predict(valid_data)
            al_val_acc = accuracy_score(valid_data.labels, al_val_preds)
            aggregated_labels = (lm_probs * lm_val_acc + al_probs * al_val_acc) / (lm_val_acc + al_val_acc)

        elif args.aggregation_method == "bayesian":
            # estimate class distribution using valid labels
            valid_labels = np.array(valid_data.labels)
            class_dist = np.bincount(valid_labels) + 1
            class_dist = class_dist / np.sum(class_dist)
            aggregated_labels = al_probs * lm_probs / (class_dist.reshape(1,-1))
            aggregated_labels = aggregated_labels / (np.sum(aggregated_labels, 1).reshape(-1,1))

        else:
            raise ValueError(f"Aggregation method {args.aggregation_method} not supported.")

        # update labels on labeled subset
        aggregated_labels[indices,:] = 0.0
        aggregated_labels[indices, labels] = 1.0
        golden_labels = np.array(train_data.labels)
        acc, nll, brier = evaluate_label_quality(golden_labels, aggregated_labels)
        al_acc, al_nll, al_brier = evaluate_label_quality(golden_labels, al_probs)

        aggregated_preds = np.argmax(aggregated_labels, axis=1)
        concensus_frac = np.sum((aggregated_preds == lm_preds) & (aggregated_preds == al_preds)) / len(train_data)
        al_active_frac = np.sum((aggregated_preds != lm_preds) & (aggregated_preds == al_preds)) / len(train_data)
        dp_active_frac = np.sum((aggregated_preds == lm_preds) & (aggregated_preds != al_preds)) / len(train_data)

        perf = evaluate_performance(train_data, valid_data, test_data, aggregated_labels, args, seed)
        n_labeled = sampler.get_n_sampled()
        sampler.update_stats(train_data, label_model=lm_calib, revision_model=reviser, encoder=encoder)
        frac_labeled = n_labeled / len(train_data)
        update_results(results, n_labeled=n_labeled, frac_labeled=frac_labeled,
                       dpal_label_acc=acc, dpal_label_nll=nll, dpal_label_brier=brier,
                       al_label_acc=al_acc, al_label_nll=al_nll, al_label_brier=al_brier,
                       concensus_frac=concensus_frac, al_active_frac=al_active_frac, dp_active_frac=dp_active_frac,
                       dpal_test_acc=perf["test_acc"], dpal_test_f1=perf["test_f1"])

        if args.verbose:
            print(f"Results at {n_labeled} ({frac_labeled* 100:.1f}%) labeled data:")
            print("DPAL label accuracy: ", acc)
            print("DP label accuracy: ", dp_label_acc)
            print("AL label accuracy: ", al_acc)
            print("Concensus frac: ", concensus_frac)
            print("DP active frac: ", dp_active_frac)
            print("AL active frac: ", al_active_frac)
            print("Test set acc: ", perf["test_acc"])

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
    parser.add_argument("--sampler", type=str, default="uncertain-joint")
    parser.add_argument("--sample_budget", type=float, default=0.05)  # Total sample budget
    parser.add_argument("--sample_per_iter",type=float, default=0.01)  # sample budget per iteration
    # uncertainty estimation for label model and active learning model
    parser.add_argument("--lm_ensemble_size", type=int, default=10)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--LF_selected_size", type=str, default="auto")
    parser.add_argument("--revision_model", type=str, default="ensemble")
    # DPAL aggregation method
    parser.add_argument("--aggregation_method", type=str, choices=["bayesian", "average", "weighted", "confidence"],
                        default="bayesian")
    # label model and end models
    parser.add_argument("--label_model", type=str, default="metal")
    parser.add_argument("--end_model", type=str, default="mlp")
    parser.add_argument("--em_epochs", type=int, default=100)
    parser.add_argument("--use_soft_labels", action="store_true")
    # other settings
    parser.add_argument("--use_valid_labels", action="store_true")
    parser.add_argument("--labeller", type=str, default="oracle")
    parser.add_argument("--metric", type=str, default="f1_macro")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tag", type=str, default="test")
    parser.add_argument("--load_results", action="store_true")
    args = parser.parse_args()
    calibration_tag = ""
    if args.lm_ensemble_size > 1:
        calibration_tag += "E"
    if args.LF_selected_size != "all":
        calibration_tag += "F"
    if args.bootstrap:
        calibration_tag += "B"

    if args.load_results:
        id_tag = f"dpal_{args.label_model}_{args.end_model}_{args.aggregation_method}_{args.sampler}_{args.tag}"
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

    M = np.array(train_data.weak_labels).shape[1]
    if args.LF_selected_size.isnumeric():
        args.LF_selected_size = int(args.LF_selected_size)
    elif args.LF_selected_size == "all":
        args.LF_selected_size = np.array(train_data.weak_labels).shape[1]
    elif args.LF_selected_size == "auto":
        if M <= 20:
            fs_list = list(range(3, M + 1, 2))
        elif M <= 50:
            fs_list = list(range(3, M + 1, 5))
        else:
            fs_list = list(range(3, M + 1, 10))
        if M not in fs_list:
            fs_list.append(M)
        min_brier = 1.0
        selected_fs = 1
        for fs in fs_list:
            calibrator = EnsembleCalibrator(train_data, valid_data, args.label_model, args.lm_ensemble_size, fs,
                                            args.bootstrap, args.seed)
            valid_probs = calibrator.predict_proba(valid_data)
            acc, nll, brier = evaluate_label_quality(valid_data.labels, valid_probs)
            if brier < min_brier:
                min_brier = brier
                selected_fs = fs

        args.LF_selected_size = selected_fs

    np.random.seed(args.seed)
    run_seeds = np.random.randint(1, 100000, args.repeats)

    results_list = []
    for i in range(args.repeats):
        print(f"Start run {i}")
        results = run_dpal(train_data, valid_data, test_data, args, seed=run_seeds[i])
        results_list.append(results)

    id_tag = f"dpal_{args.label_model}_{args.end_model}_{args.aggregation_method}_{args.sampler}_{calibration_tag}_{args.tag}"

    save_results(results_list, args.output_path, args.dataset, f"{id_tag}.json")


