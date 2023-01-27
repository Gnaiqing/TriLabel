import argparse
import time
from labeller.labeller import get_labeller
import numpy as np
import copy
from utils import evaluate_end_model, save_results, get_sampler, get_label_model, update_results, \
    preprocess_data, evaluate_label_quality


def run_nashaat(train_data, valid_data, test_data, args, seed):
    """
    Run nashaat pipeline to clean noisy labels
    """
    start = time.process_time()
    results = {
        "n_labeled": [],
        "frac_labeled": [],
        "label_acc": [],
        "label_nll": [],
        "label_brier": [],
        "label_coverage": [],
        "test_acc": [],
        "test_f1": [],
        "golden_test_acc": np.nan,
        "golden_test_f1": np.nan,
    }
    label_model = get_label_model(args.label_model)
    covered_train_data = train_data.get_covered_subset()
    label_model.fit(dataset_train=covered_train_data,
                    dataset_valid=valid_data)
    lm_train_probs = label_model.predict_proba(covered_train_data)
    lm_train_preds = np.argmax(lm_train_probs, axis=1)
    label_acc, label_nll, label_brier = evaluate_label_quality(covered_train_data.labels, lm_train_probs)
    label_coverage = len(covered_train_data) / len(train_data)
    if args.use_soft_labels:
        pred_train_labels = lm_train_probs
    else:
        pred_train_labels = lm_train_preds

    perf, _ = evaluate_end_model(covered_train_data, pred_train_labels, valid_data, test_data, args, seed)
    update_results(results, n_labeled=0, frac_labeled=0.0,
                   label_acc=label_acc, label_nll=label_nll, label_brier=label_brier, label_coverage=label_coverage,
                   test_acc=perf["test_acc"], test_f1=perf["test_f1"])
    golden_perf, _ = evaluate_end_model(train_data, train_data.labels, valid_data, test_data, args, seed)
    results["golden_test_acc"] = golden_perf["test_acc"]
    results["golden_test_f1"] = golden_perf["test_f1"]

    labeller = get_labeller(args.labeller)
    revised_train_data = copy.copy(train_data)
    sampler = get_sampler(args.sampler, revised_train_data, labeller, label_model=label_model)

    n_labeled = 0
    while n_labeled < args.sample_budget:
        n_to_sample = min(args.sample_budget - sampler.get_n_sampled(), args.sample_per_iter)
        sampler.sample_distinct(n=n_to_sample)
        indices, labels = sampler.get_sampled_points()
        revised_weak_labels = np.array(revised_train_data.weak_labels)
        revised_weak_labels[indices, :] = labels.reshape((-1, 1))
        revised_train_data.weak_labels = revised_weak_labels.tolist()
        covered_train_data = revised_train_data.get_covered_subset()
        label_model = get_label_model(args.label_model)
        label_model.fit(dataset_train=covered_train_data, dataset_valid=valid_data)
        lm_train_probs = label_model.predict_proba(covered_train_data)
        lm_train_preds = np.argmax(lm_train_probs, axis=1)
        label_acc, label_nll, label_brier = evaluate_label_quality(covered_train_data.labels, lm_train_probs)
        label_coverage = len(covered_train_data) / len(train_data)
        if args.use_soft_labels:
            pred_train_labels = lm_train_probs
        else:
            pred_train_labels = lm_train_preds
        perf, _ = evaluate_end_model(covered_train_data, pred_train_labels, valid_data, test_data, args, seed)
        n_labeled = sampler.get_n_sampled()
        frac_labeled = n_labeled / len(train_data)
        update_results(results, n_labeled=n_labeled, frac_labeled=frac_labeled,
                       label_acc=label_acc, label_nll=label_nll, label_brier=label_brier, label_coverage=label_coverage,
                       test_acc=perf["test_acc"], test_f1=perf["test_f1"])
        sampler.update_stats(train_data=revised_train_data, label_model=label_model)

    end = time.process_time()
    results["time"] = end - start
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # device info
    parser.add_argument("--device", type=str, default="cuda:0")
    # dataset
    parser.add_argument("--dataset", type=str, default="youtube")
    parser.add_argument("--dataset_path", type=str, default="../datasets/")
    parser.add_argument("--extract_fn", type=str, default="bert")  # method used to extract features
    parser.add_argument("--max_dim", type=int, default=300)
    # sampler
    parser.add_argument("--sampler", type=str, default="uncertain-lm")
    parser.add_argument("--sample_budget", type=float, default=0.05)  # Total sample budget
    parser.add_argument("--sample_per_iter",type=float, default=0.01)  # sample budget per iteration
    # label model and end models
    parser.add_argument("--label_model", type=str, default="metal")
    parser.add_argument("--end_model", type=str, default="mlp")
    parser.add_argument("--em_epochs", type=int, default=100)
    parser.add_argument("--use_soft_labels", action="store_true")
    # other settings
    parser.add_argument("--labeller", type=str, default="oracle")
    parser.add_argument("--metric", type=str, default="f1_macro")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tag", type=str, default="test")
    args = parser.parse_args()

    train_data, valid_data, test_data = preprocess_data(args)
    np.random.seed(args.seed)
    run_seeds = np.random.randint(1, 100000, args.repeats)

    results_list = []
    for i in range(args.repeats):
        print(f"Start run {i}")
        results = run_nashaat(train_data, valid_data, test_data, args, seed=run_seeds[i])
        results_list.append(results)

    id_tag = f"nashaat_{args.label_model}_{args.sampler}_{args.tag}"
    save_results(results_list, args.output_path, args.dataset,
                 f"{id_tag}.json")
