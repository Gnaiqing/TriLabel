# pipeline for active learning using uncertain sampling
import time
import argparse
from labeller.labeller import get_labeller
from scipy.stats import entropy
import numpy as np
from utils import save_results, evaluate_end_model, get_sampler, update_results, preprocess_data


def run_active_learning(train_data, valid_data, test_data, args, seed):
    start = time.process_time()
    results = {
        "n_labeled": [],  # number of expert labeled data
        "frac_labeled": [],  # fraction of expert labeled data
        "label_acc": [],
        "label_nll": [],
        "label_brier": [],
        "label_coverage": [],
        "test_acc": [],  # end model's test accuracy when using active learning
        "test_f1": [],  # end model's test f1 score (macro) using active learning
        "golden_test_acc": np.nan,  # end model's test accuracy using golden labels
        "golden_test_f1": np.nan,  # end model's test f1 score (macro) using golden labels
    }
    golden_perf, _ = evaluate_end_model(pred_train_data=train_data,
                                        pred_train_labels=train_data.labels,
                                        valid_data=valid_data,
                                        test_data=test_data,
                                        args=args,
                                        seed=seed)

    update_results(results, n_labeled=0, frac_labeled=0.0,
                   label_acc=np.nan, label_nll = np.nan, label_brier=np.nan, label_coverage=0,
                   test_acc=np.nan, test_f1=np.nan)
    results["golden_test_acc"] = golden_perf["test_acc"]
    results["golden_test_f1"] = golden_perf["test_f1"]

    n_sampled = 0
    labeller = get_labeller(args.labeller)
    sampler = get_sampler("passive", train_data, labeller, seed=seed)
    while n_sampled < args.sample_budget:
        n_to_sample = min(args.sample_budget - n_sampled, args.sample_per_iter)
        if n_sampled == 0:
            sampler.sample_distinct(n_to_sample)
        else:
            indices = []
            uncertainty = entropy(train_probs, axis=1)
            uncertainty_order = np.argsort(uncertainty)[::-1]
            for idx in uncertainty_order:
                if not sampler.sampled[idx]:
                    indices.append(idx)
                if len(indices) >= n_to_sample:
                    break

            sampler.label_selected_indices(indices)

        indices, labels = sampler.get_sampled_points()
        covered_train_data = train_data.create_subset(indices)
        perf, end_model = evaluate_end_model(covered_train_data, labels, valid_data, test_data, args, seed)
        train_probs = end_model.predict_proba(train_data)
        n_sampled = sampler.get_n_sampled()
        frac_sampled = n_sampled / len(train_data)
        update_results(results, n_labeled=n_sampled, frac_labeled=frac_sampled,
                       label_acc=1.0, label_nll=0.0, label_brier=0.0, label_coverage=frac_sampled,
                       test_acc=perf["test_acc"], test_f1=perf["test_f1"])

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
    parser.add_argument("--sample_budget", type=float, default=0.05)  # Total sample budget
    parser.add_argument("--sample_per_iter", type=float, default=0.01)  # sample budget per iteration
    # end model
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
        print(f"Start run {i} of uncertainty sampling (AL)")
        results = run_active_learning(train_data, valid_data, test_data, args, seed=run_seeds[i])
        results_list.append(results)

    id_tag = f"al_{args.end_model}_{args.tag}"
    save_results(results_list, args.output_path, args.dataset, f"{id_tag}.json")
