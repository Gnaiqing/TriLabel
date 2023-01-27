import argparse
import time
from labeller.labeller import get_labeller
import numpy as np
from utils import evaluate_end_model, save_results, get_sampler, get_label_model, update_results, get_filter_probs, \
    preprocess_data, evaluate_label_quality, get_reviser, ABSTAIN, select_thresholds, estimate_cov_acc_tradeoff


def run_dpal_boost(train_data, valid_data, test_data, args, seed):
    start = time.process_time()
    results = {
        "n_labeled": [],  # number of expert labeled data
        "frac_labeled": [],  # fraction of expert labeled data
        "label_acc": [],  # DPAL label accuracy
        "label_nll": [],  # DPAL label NLL score
        "label_brier": [],  # DPAL label brier score
        "label_coverage": [],  # covered fraction of training data
        "test_acc": [],  # end model's test accuracy
        "test_f1": [],  # end model's test f1
        "golden_test_acc": np.nan,  # end model's test accuracy using golden labels
        "golden_test_f1": np.nan  # end model's test f1 using golden labels
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
    encoder = None
    reviser = get_reviser(args.revision_model, train_data, valid_data, encoder, args.device, seed)
    sampler = get_sampler(args.sampler, train_data, labeller, label_model=label_model, revision_model=reviser)
    n_labeled = 0
    while n_labeled < args.sample_budget:
        n_to_sample = min(args.sample_budget - sampler.get_n_sampled(), args.sample_per_iter)
        sampler.sample_distinct(n_to_sample)
        indices, labels = sampler.get_sampled_points()
        reviser.train_revision_model(indices, labels)
        cov_acc_dict = estimate_cov_acc_tradeoff(train_data, valid_data, label_model, reviser)
        candidate_theta_list = []
        cov_list = []
        acc_list = []
        for i in range(len(cov_acc_dict["lm_theta"])):
            for j in range(len(cov_acc_dict["rm_theta"])):
                acc = cov_acc_dict["acc"][i,j]
                cov = cov_acc_dict["cov"][i,j]
                if args.desired_label_acc is not None and acc < args.desired_label_acc:
                    continue
                if args.desired_label_cov is not None and cov < args.desired_label_cov:
                    continue

                idx = 0
                while idx < len(cov_list) and cov_list[idx] <= cov:
                    idx += 1

                if idx < len(cov_list) and acc_list[idx] >= acc:
                    # there exist a combination that have higher/same accuracy and higher coverage
                    continue

                if idx > 0 and cov_list[idx-1] == cov and acc_list[idx-1] >= acc:
                    # there exist a combination that have higher/same accuracy and same coverage
                    continue

                cov_list.insert(idx, cov)
                acc_list.insert(idx, acc)
                candidate_theta_list.insert(idx, (cov_acc_dict["lm_theta"][i], cov_acc_dict["rm_theta"][j]))
                while idx > 0 and acc_list[idx-1] <= acc:
                    cov_list.pop(idx-1)
                    acc_list.pop(idx-1)
                    candidate_theta_list.pop(idx-1)
                    idx -= 1

        selected_theta, best_f1 = select_thresholds(train_data, valid_data, candidate_theta_list, label_model, reviser,
                                                    args, seed)

        train_act_probs, train_act_indices = get_filter_probs(train_data, label_model, reviser,
                                                              selected_theta[0], selected_theta[1])
        pred_train_data = train_data.create_subset(train_act_indices)
        label_acc, label_nll, label_brier = evaluate_label_quality(pred_train_data.labels, train_act_probs)
        label_coverage = len(pred_train_data) / len(train_data)
        if args.use_soft_labels:
            pred_train_labels = train_act_probs
        else:
            pred_train_labels = train_act_probs.argmax(axis=1)
        perf, _ = evaluate_end_model(pred_train_data, pred_train_labels, valid_data, test_data, args, seed)
        n_labeled = sampler.get_n_sampled()
        frac_labeled = n_labeled / len(train_data)
        update_results(results, n_labeled=n_labeled, frac_labeled=frac_labeled,
                       label_acc=label_acc, label_nll=label_nll, label_brier=label_brier, label_coverage=label_coverage,
                       test_acc=perf["test_acc"], test_f1=perf["test_f1"])
        sampler.update_stats(revision_model=reviser)

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
    parser.add_argument("--max_dim", type=int, default=300)  # dimension reduction using PCA
    # User provided constraints
    parser.add_argument("--sample_budget", type=float, default=0.05)  # Total sample budget
    parser.add_argument("--sample_per_iter", type=float, default=0.01)  # sample budget per iteration
    parser.add_argument("--desired_label_acc", type=float, default=None)
    parser.add_argument("--desired_label_cov", type=float, default=None)
    # active learning sample strategy
    parser.add_argument("--sampler", type=str, default="uncertain-rm")
    # active learning model
    parser.add_argument("--revision_model", type=str, default="mlp")
    # label model and end models
    parser.add_argument("--label_model", type=str, default="metal")
    parser.add_argument("--end_model", type=str, default="mlp")
    parser.add_argument("--em_epochs", type=int, default=100)
    parser.add_argument("--use_soft_labels", action="store_true")
    # other settings
    parser.add_argument("--theta_explore_strategy", type=str, default="step")
    parser.add_argument("--theta_explore_num", type=int, default=10)
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
        results = run_dpal_boost(train_data, valid_data, test_data, args, seed=run_seeds[i])
        results_list.append(results)

    id_tag = f"dpal-boost_{args.label_model}_{args.end_model}_{args.sampler}_{args.tag}"
    save_results(results_list, args.output_path, args.dataset, f"{id_tag}.json")