import argparse
import time
from labeller.labeller import get_labeller
import numpy as np
from sklearn.metrics import accuracy_score
from calibrator.lm_ensemble import EnsembleCalibrator
from utils import evaluate_end_model, save_results, get_sampler, get_label_model, update_results, get_filter_probs, \
    preprocess_data, evaluate_label_quality, get_reviser, select_thresholds, estimate_cov_acc_tradeoff,\
    plot_performance, calibrate_lm_threshold


def run_trilabel(train_data, valid_data, test_data, args, seed):
    start = time.process_time()
    if args.evaluate:
        # evaluate label quality
        results = {
            "n_labeled": [],  # number of expert labeled data
            "n_sampled": [],  # number of expert labeled data + pseudo-labeled data
            "frac_labeled": [],  # fraction of expert labeled data
            "sampled_acc": [],  # accuracy of sampled data
            "label_acc": [],  # label accuracy
            "label_nll": [],  # label NLL score
            "label_brier": [],  # label brier score
            "label_coverage": [],  # covered fraction of training data
            "dp_coverage": [],  # fraction of training data covered by DP
            "al_coverage": [],  # fraction of training data covered by AL
            "al_acc": [],  # accuracy of AL model
            "test_acc": [],  # end model's test accuracy
            "test_f1": [],  # end model's test f1
            "golden_test_acc": np.nan,  # end model's test accuracy using golden labels
            "golden_test_f1": np.nan  # end model's test f1 using golden labels
        }
    else:
        results = {}

    if args.record_runtime:
        last_timestamp = start
        lm_time = 0   # time used for training label model
        al_time = 0   # time used for sampling and training AL model
        thres_time = 0  # time used for threshold selection
        perf_time = 0  # time used for estimating coverage-accuracy tradeoff

    train_act_probs, train_act_indices = None, None
    M = train_data.n_lf
    covered_train_data = train_data.get_covered_subset()
    if args.calibration is None:
        label_model = get_label_model(args.label_model, seed=seed)
        label_model.fit(dataset_train=covered_train_data,
                        dataset_valid=valid_data)
    elif args.calibration == "EN":
        label_model = EnsembleCalibrator(train_data, valid_data, args.label_model,
                                         ensemble_size=10, feature_size=M, bootstrap=False, seed=seed)
    elif args.calibration == "EN+FS":
        feature_size = np.floor(M * 0.8).astype(int)
        label_model = EnsembleCalibrator(train_data, valid_data, args.label_model,
                                         ensemble_size=10, feature_size=feature_size,bootstrap=False, seed=seed)
    else:
        feature_size = np.floor(M * 0.8).astype(int)
        label_model = EnsembleCalibrator(train_data, valid_data, args.label_model,
                                         ensemble_size=10, feature_size=feature_size, bootstrap=True, seed=seed)



    if args.record_runtime:
        cur_timestamp = time.process_time()
        lm_time += cur_timestamp - last_timestamp
        last_timestamp = cur_timestamp

    if args.evaluate:
        lm_train_probs = label_model.predict_proba(covered_train_data)
        lm_train_preds = np.argmax(lm_train_probs, axis=1)
        if args.use_soft_labels:
            pred_train_labels = lm_train_probs
        else:
            pred_train_labels = lm_train_preds
        label_acc, label_nll, label_brier = evaluate_label_quality(covered_train_data.labels, lm_train_probs)
        label_coverage = len(covered_train_data) / len(train_data)
        perf, _ = evaluate_end_model(covered_train_data, pred_train_labels, valid_data, test_data, args, seed)
        update_results(results, n_labeled=0, frac_labeled=0.0,n_sampled=0, sampled_acc=np.nan,
                       label_acc=label_acc, label_nll=label_nll, label_brier=label_brier,
                       label_coverage=label_coverage,dp_coverage=label_coverage, al_coverage=0.0,
                       al_acc=np.nan, test_acc=perf["test_acc"], test_f1=perf["test_f1"])
        golden_perf, _ = evaluate_end_model(train_data, train_data.labels, valid_data, test_data, args, seed)
        results["golden_test_acc"] = golden_perf["test_acc"]
        results["golden_test_f1"] = golden_perf["test_f1"]

    labeller = get_labeller(args.labeller)
    encoder = None
    reviser = get_reviser(args.revision_model, train_data, valid_data, encoder, args.device, seed)

    if args.sampler in ["tri-pl", "tri-random", "tri-pl+random"]:
        lm_threshold = calibrate_lm_threshold(valid_data, label_model)
        valid_class_dist = np.bincount(valid_data.labels) / len(train_data)
        init_method = args.sampler[4:]
        sampler = get_sampler(args.sampler, train_data, labeller, label_model=label_model, revision_model=reviser,
                              valid_class_dist=valid_class_dist, init_method=init_method, lm_threshold=lm_threshold)
    else:
        sampler = get_sampler(args.sampler, train_data, labeller, label_model=label_model, revision_model=reviser)
    n_labeled = 0
    while n_labeled < args.sample_budget:
        n_to_sample = min(args.sample_budget - sampler.get_n_sampled(), args.sample_per_iter)
        sampler.sample_distinct(n_to_sample)
        indices, labels = sampler.get_sampled_points()
        n_labeled = sampler.get_n_sampled()
        frac_labeled = n_labeled / len(train_data)
        reviser.train_revision_model(indices, labels)
        sampler.update_stats()
        if args.record_runtime:
            cur_timestamp = time.process_time()
            al_time += cur_timestamp - last_timestamp
            last_timestamp = cur_timestamp

        cov_acc_dict = estimate_cov_acc_tradeoff(train_data, valid_data, label_model, reviser)
        candidate_theta_list = []
        cov_list = []
        acc_list = []
        epsilon_list = []
        for i in range(len(cov_acc_dict["lm_theta"])):
            for j in range(len(cov_acc_dict["rm_theta"])):

                acc = cov_acc_dict["acc"][i,j]
                epsilon = cov_acc_dict["epsilon"][i,j]
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
                epsilon_list.insert(idx, epsilon)
                candidate_theta_list.insert(idx, (cov_acc_dict["lm_theta"][i], cov_acc_dict["rm_theta"][j]))
                while idx > 0 and acc_list[idx-1] <= acc:
                    cov_list.pop(idx-1)
                    acc_list.pop(idx-1)
                    candidate_theta_list.pop(idx-1)
                    idx -= 1

        if args.record_runtime:
            cur_timestamp = time.process_time()
            perf_time += cur_timestamp - last_timestamp
            last_timestamp = cur_timestamp

        if len(candidate_theta_list) == 0:
            # No configuration can satisfy user provided constraints at current label budget
            print(f"Constraints cannot be satisfied at label budget {n_labeled}. Increasing sample size.")
            if args.evaluate:
                indices, labels = sampler.get_sampled_points()
                sampled_acc = accuracy_score(np.array(train_data.labels)[indices], labels)
                update_results(results, n_labeled=n_labeled, frac_labeled=frac_labeled, n_sampled=len(indices),
                               sampled_acc=sampled_acc, label_acc=np.nan, label_nll=np.nan, label_brier=np.nan,
                               label_coverage=np.nan, dp_coverage=np.nan, al_coverage=np.nan,
                               al_acc=np.nan, test_acc=np.nan, test_f1=np.nan)
            continue


        if args.optimize_target == "f1":
            selected_theta, valid_f1, test_f1 = select_thresholds(train_data, valid_data, test_data,
                                                                   candidate_theta_list,
                                                                   label_model, reviser, args, seed)

            if args.plot_performance:
                theta_indices = np.nonzero(valid_f1 > 0)[0]
                evaluated_valid_f1 = valid_f1[theta_indices]
                evaluated_test_f1 = test_f1[theta_indices]
                evaluated_cov = np.array(cov_list)[theta_indices]
                evaluated_acc = np.array(acc_list)[theta_indices]
                evaluated_epsilon = np.array(epsilon_list)[theta_indices]
                gt_acc = np.zeros_like(evaluated_acc)
                for i in range(len(theta_indices)):
                    lm_theta, rm_theta = candidate_theta_list[theta_indices[i]]
                    train_probs, train_act_indices = get_filter_probs(train_data, label_model, reviser, lm_theta, rm_theta)
                    train_preds = np.argmax(train_probs, axis=1)
                    train_labels = np.array(train_data.labels)[train_act_indices]
                    gt_acc[i] = accuracy_score(train_labels, train_preds)

                plot_performance(args.dataset, evaluated_cov, evaluated_acc, evaluated_epsilon, gt_acc,
                                 evaluated_valid_f1, evaluated_test_f1, n_labeled, args.output_path)
        elif args.optimize_target == "accuracy":
            selected_theta = candidate_theta_list[0]  # select the one with highest label accuracy
        else:
            selected_theta = candidate_theta_list[-1]  # select the one with highest label coverage

        train_act_probs, train_act_indices = get_filter_probs(train_data, label_model, reviser,
                                                              selected_theta[0], selected_theta[1])
        if args.record_runtime:
            cur_timestamp = time.process_time()
            thres_time += cur_timestamp - last_timestamp
            last_timestamp = cur_timestamp

        if args.evaluate:
            indices, labels = sampler.get_sampled_points()
            sampled_acc = accuracy_score(np.array(train_data.labels)[indices], labels)
            lm_train_conf = np.max(lm_train_probs, axis=1)
            dp_coverage = np.sum(lm_train_conf >= selected_theta[0]) / len(train_data)
            al_coverage = len(train_act_indices) / len(train_data) - dp_coverage
            al_preds = reviser.predict(train_data)
            al_acc = accuracy_score(train_data.labels, al_preds)
            pred_train_data = train_data.create_subset(train_act_indices)
            label_acc, label_nll, label_brier = evaluate_label_quality(pred_train_data.labels, train_act_probs)
            label_coverage = len(pred_train_data) / len(train_data)
            if args.use_soft_labels:
                pred_train_labels = train_act_probs
            else:
                pred_train_labels = train_act_probs.argmax(axis=1)
            perf, _ = evaluate_end_model(pred_train_data, pred_train_labels, valid_data, test_data, args, seed)
            update_results(results, n_labeled=n_labeled, frac_labeled=frac_labeled,n_sampled=len(indices),
                           sampled_acc=sampled_acc, label_acc=label_acc, label_nll=label_nll, label_brier=label_brier,
                           label_coverage=label_coverage, dp_coverage=dp_coverage, al_coverage=al_coverage,
                           al_acc=al_acc, test_acc=perf["test_acc"], test_f1=perf["test_f1"])

    end = time.process_time()
    results["time"] = end - start
    if args.record_runtime:
        results["lm_time"] = lm_time
        results["al_time"] = al_time
        results["perf_time"] = perf_time
        results["thres_time"] = thres_time

    return train_act_probs, train_act_indices, results


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
    parser.add_argument("--sample_budget", type=float, default=300)  # Total sample budget
    parser.add_argument("--desired_label_acc", type=float, default=None)  # Desired Label Accuracy
    parser.add_argument("--desired_label_cov", type=float, default=None)  # Desired Label Coverage
    parser.add_argument("--optimize_target", type=str, choices=["accuracy", "coverage", "f1"], default="f1")
    # active learning sample strategy and model
    parser.add_argument("--sampler", type=str, default="uncertain-rm")
    parser.add_argument("--sample_per_iter", type=float, default=100)  # Sample per iteration (batch size)
    parser.add_argument("--revision_model", type=str, default="mlp")
    # label model and end models
    parser.add_argument("--label_model", type=str, default="metal")
    parser.add_argument("--end_model", type=str, default="mlp")
    parser.add_argument("--em_epochs", type=int, default=100)
    parser.add_argument("--use_soft_labels", action="store_true")
    # other settings
    parser.add_argument("--calibration", type=str, choices=["EN", "EN+FS", "EN+FS+BS"], default=None)
    parser.add_argument("--evaluate", action="store_true")  # evaluate label quality and performance using golden labels
    parser.add_argument("--record_runtime", action="store_true")  # record runtime of every component
    parser.add_argument("--plot_performance", action="store_true")  # draw performance plot
    parser.add_argument("--theta_explore_strategy", type=str, default="step")
    parser.add_argument("--theta_explore_num", type=int, default=10)  # maximum number of evaluated theta thresholds

    parser.add_argument("--labeller", type=str, default="oracle")
    parser.add_argument("--metric", type=str, default="f1_macro")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tag", type=str, default="test")
    args = parser.parse_args()
    if args.record_runtime:
        args.evaluate = False  # No evaluation of label quality when recording pipeline runtime

    train_data, valid_data, test_data = preprocess_data(args)

    if args.sample_budget <= 1:
        args.sample_budget = np.ceil(args.sample_budget * len(train_data)).astype(int)
        args.sample_per_iter = np.ceil(args.sample_per_iter * len(train_data)).astype(int)
    else:
        args.sample_budget = int(args.sample_budget)
        args.sample_per_iter = int(args.sample_per_iter)

    np.random.seed(args.seed)
    run_seeds = np.random.randint(1, 100000, args.repeats)
    results_list = []
    for i in range(args.repeats):
        print(f"Start run {i}")
        _, _, results = run_trilabel(train_data, valid_data, test_data, args, seed=run_seeds[i])
        results_list.append(results)

    id_tag = f"trilabel_{args.label_model}_{args.end_model}_{args.sampler}_{args.tag}"
    save_results(results_list, args.output_path, args.dataset, f"{id_tag}.json")
