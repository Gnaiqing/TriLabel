import argparse
import time
from labeller.labeller import get_labeller
import numpy as np
from sklearn.metrics import accuracy_score
from utils import evaluate_end_model, save_results, get_sampler, get_label_model, update_results, \
    preprocess_data, evaluate_label_quality, get_reviser


def run_dpal_ensemble(train_data, valid_data, test_data, args, seed):
    start = time.process_time()
    results = {
        "n_labeled": [],  # number of expert labeled data
        "frac_labeled": [],  # fraction of expert labeled data
        "label_acc": [],  # DPAL label accuracy
        "label_nll": [],  # DPAL label NLL score
        "label_brier": [],  # DPAL label brier score
        "label_coverage": [], # covered fraction of training data
        "concensus_frac": [],  # fraction of data where prediction result follow concensus of DP and AL
        "al_active_frac": [],  # fraction of data where prediction result follow AL but not DP
        "dp_active_frac": [],  # fraction of data where prediction result follow DP but not AL
        "test_acc": [],  # end model's test accuracy
        "test_f1": [],  # end model's test f1
        "golden_test_acc": np.nan,  # end model's test accuracy using golden labels
        "golden_test_f1": np.nan   # end model's test f1 using golden labels
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
                   concensus_frac=0.0, al_active_frac=0.0, dp_active_frac=1.0,
                   test_acc=perf["test_acc"], test_f1=perf["test_f1"])
    golden_perf, _ = evaluate_end_model(train_data, train_data.labels, valid_data, test_data, args, seed)
    results["golden_test_acc"] = golden_perf["test_acc"]
    results["golden_test_f1"] = golden_perf["test_f1"]

    labeller = get_labeller(args.labeller)

    encoder = None
    reviser = get_reviser(args.revision_model, train_data, valid_data, encoder, args.device, seed)
    sampler = get_sampler(args.sampler, train_data, labeller, label_model, reviser, encoder, seed=seed)

    n_labeled = 0
    while n_labeled < args.sample_budget:
        n_to_sample = min(args.sample_budget - sampler.get_n_sampled(), args.sample_per_iter)
        sampler.sample_distinct(n=n_to_sample)
        # train revision model (active learning model) using labeled data
        indices, labels = sampler.get_sampled_points()
        reviser.train_revision_model(indices, labels)

        # revise probabilistic labels using active learning
        lm_probs = label_model.predict_proba(train_data)
        lm_preds = np.argmax(lm_probs, axis=1)
        lm_conf = np.max(lm_probs, axis=1)

        al_probs = reviser.predict_proba(train_data)
        al_preds = np.argmax(al_probs, axis=1)
        al_conf = np.max(al_probs, axis=1)

        if args.aggregation_method == "confidence":
            dp_active_indices = np.nonzero(lm_conf >= al_conf)[0]
            aggregated_labels = al_probs.copy()
            aggregated_labels[dp_active_indices, :] = lm_probs[dp_active_indices, :]
        elif args.aggregation_method == "average":
            aggregated_labels = (lm_probs + al_probs) / 2
        elif args.aggregation_method == "weighted":
            lm_val_preds = label_model.predict(valid_data)
            lm_val_acc = accuracy_score(valid_data.labels, lm_val_preds)
            al_val_preds = reviser.predict(valid_data)
            al_val_acc = accuracy_score(valid_data.labels, al_val_preds)
            aggregated_labels = (lm_probs * lm_val_acc + al_probs * al_val_acc) / (lm_val_acc + al_val_acc)
        elif args.aggregation_method == "bayesian":
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
        label_acc, label_nll, label_brier = evaluate_label_quality(train_data.labels, aggregated_labels)
        aggregated_preds = np.argmax(aggregated_labels, axis=1)
        concensus_frac = np.sum((aggregated_preds == lm_preds) & (aggregated_preds == al_preds)) / len(train_data)
        al_active_frac = np.sum((aggregated_preds != lm_preds) & (aggregated_preds == al_preds)) / len(train_data)
        dp_active_frac = np.sum((aggregated_preds == lm_preds) & (aggregated_preds != al_preds)) / len(train_data)

        if args.use_soft_labels:
            pred_train_labels = aggregated_labels
        else:
            pred_train_labels = aggregated_preds
        perf, _ = evaluate_end_model(train_data, pred_train_labels, valid_data, test_data, args, seed)
        n_labeled = sampler.get_n_sampled()
        frac_labeled = n_labeled / len(train_data)
        sampler.update_stats(train_data, revision_model=reviser)
        update_results(results, n_labeled=n_labeled, frac_labeled=frac_labeled,
                       label_acc=label_acc, label_nll=label_nll, label_brier=label_brier, label_coverage=1.0,
                       concensus_frac=concensus_frac, al_active_frac=al_active_frac, dp_active_frac=dp_active_frac,
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
    parser.add_argument("--sampler", type=str, default="passive")
    parser.add_argument("--sample_budget", type=float, default=0.05)  # Total sample budget
    parser.add_argument("--sample_per_iter",type=float, default=0.01)  # sample budget per iteration
    # active learning model
    parser.add_argument("--revision_model", type=str, default="mlp")
    # aggregation method
    parser.add_argument("--aggregation_method", type=str, choices=["bayesian", "average", "weighted", "confidence"],
                        default="bayesian")
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
        results = run_dpal_ensemble(train_data, valid_data, test_data, args, seed=run_seeds[i])
        results_list.append(results)

    id_tag = f"dpal_{args.label_model}_{args.end_model}_{args.aggregation_method}_{args.sampler}_{args.tag}"
    save_results(results_list, args.output_path, args.dataset, f"{id_tag}.json")


