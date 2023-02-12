import numpy as np
from wrench.labelmodel import Snorkel, DawidSkene, MajorityVoting, MeTaL
from baselines.active_weasul.label_model import LabelModel
from wrench.endmodel import EndClassifierModel, Cosine
from dataset.load_dataset import load_real_dataset, load_synthetic_dataset
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
import random
import torch
from pathlib import Path
import json

ABSTAIN = -1


def preprocess_data(args):
    if args.dataset[:3] == "syn":
        train_data, valid_data, test_data = load_synthetic_dataset(args.dataset)
    else:
        train_data, valid_data, test_data = load_real_dataset(args.dataset_path, args.dataset, args.extract_fn)

    if args.max_dim is not None and train_data.features.shape[1] > args.max_dim:
        # use truncated SVD to reduce feature dimensions
        pca = PCA(n_components=args.max_dim, random_state=args.seed)
        pca.fit(train_data.features)
        train_data.features = pca.transform(train_data.features)
        valid_data.features = pca.transform(valid_data.features)
        test_data.features = pca.transform(test_data.features)

    return train_data, valid_data, test_data


def get_label_model(model_type, **kwargs):
    if model_type == "snorkel":
        label_model = Snorkel(lr=0.01, l2=0.0, n_epochs=100, **kwargs)
    elif model_type == "ds":
        label_model = DawidSkene()
    elif model_type == "mv":
        label_model = MajorityVoting()
    elif model_type == "metal":
        label_model = MeTaL(lr=0.01, n_epochs=100, **kwargs)
    elif model_type == "aw":
        if "penalty_strength" in kwargs:
            label_model = LabelModel(n_epochs=100, lr=0.01, active_learning=True,
                                     penalty_strength=kwargs["penalty_strength"])
        else:
            label_model = LabelModel(n_epochs=100, lr=0.01, active_learning=True)
    else:
        raise ValueError(f"label model {model_type} not supported.")
    return label_model


def get_end_model(model_type):
    if model_type == "mlp":
        end_model = EndClassifierModel(
            backbone="MLP",
            batch_size=4096,
            test_batch_size=4096,
            optimizer="Adam",
            optimizer_lr=1e-2,
            optimizer_weight_decay=1e-5
        )
    elif model_type == "bert":
        end_model = EndClassifierModel(
            batch_size=32,
            real_batch_size=32,
            test_batch_size=32,
            backbone="BERT",
            backbone_model_name="bert-base-cased",
            backbone_max_tokens=128,
            backbone_fine_tune_layers=-1,
            optimizer="AdamW",
            optimizer_lr=5e-5,
            optimizer_weight_decay=0.0
        )
    elif model_type == "roberta":
        end_model = EndClassifierModel(
            batch_size=32,
            real_batch_size=32,
            test_batch_size=32,
            backbone="BERT",
            backbone_model_name="roberta-base",
            backbone_max_tokens=128,
            backbone_fine_tune_layers=-1,
            optimizer="AdamW",
            optimizer_lr=5e-5,
            optimizer_weight_decay=0.0
        )
    else:
        raise ValueError(f"end model {model_type} not implemented.")
    return end_model


def get_sampler(sampler_type, train_data, labeller, label_model=None, revision_model=None, encoder=None, seed=None, **kwargs):
    from sampler import PassiveSampler, UncertaintySampler, MaxKLSampler, CoreSetSampler, \
         DALSampler, ClusterMarginSampler, BadgeSampler, TriSampler
    if sampler_type == "passive":
        return PassiveSampler(train_data, labeller, label_model, revision_model, encoder, seed=seed, **kwargs)
    elif sampler_type == "uncertain-lm":
        return UncertaintySampler(train_data, labeller, label_model, revision_model, encoder, seed=seed,
                                  uncertain_type="lm", **kwargs)
    elif sampler_type == "uncertain-rm":
        return UncertaintySampler(train_data, labeller, label_model, revision_model, encoder, seed=seed,
                                  uncertain_type="rm", **kwargs)
    elif sampler_type == "uncertain-joint":
        return UncertaintySampler(train_data, labeller, label_model, revision_model, encoder, seed=seed,
                                  uncertain_type="joint", **kwargs)
    elif sampler_type == "dal":
        return DALSampler(train_data, labeller, label_model, revision_model, encoder, seed=seed, **kwargs)
    elif sampler_type == "cluster-margin-lm":
        return ClusterMarginSampler(train_data, labeller, label_model, revision_model, encoder, seed=seed,
                                    uncertain_type="lm", **kwargs)
    elif sampler_type == "cluster-margin-rm":
        return ClusterMarginSampler(train_data, labeller, label_model, revision_model, encoder, seed=seed,
                                    uncertain_type="rm", **kwargs)
    elif sampler_type == "cluster-margin-joint":
        return ClusterMarginSampler(train_data, labeller, label_model, revision_model, encoder, seed=seed,
                                    uncertain_type="joint", **kwargs)
    elif sampler_type == "maxkl":
        return MaxKLSampler(train_data, labeller,label_model, revision_model, encoder, seed=seed, **kwargs)
    elif sampler_type == "badge":
        return BadgeSampler(train_data, labeller, label_model, revision_model, encoder, seed=seed, **kwargs)
    elif sampler_type == "coreset":
        return CoreSetSampler(train_data, labeller, label_model, revision_model, encoder, seed=seed, **kwargs)
    elif sampler_type in ["tri-pl", "tri-random", "tri-pl+random"]:
        return TriSampler(train_data, labeller, label_model, revision_model, encoder, seed=seed,
                              uncertain_type="rm", print_result=True, **kwargs)
    else:
        raise ValueError(f"sampler {sampler_type} not implemented.")


def get_reviser(reviser_type, train_data, valid_data, encoder,  device, seed):
    from reviser import EnsembleReviser, ExpertLabelReviser, MCDropoutReviser, \
        MLPReviser, MLPTempReviser
    if reviser_type == "mlp":
        return MLPReviser(train_data, encoder, device, valid_data, seed)
    elif reviser_type == "mlp-temp":
        return MLPTempReviser(train_data, encoder, device, valid_data, seed)
    elif reviser_type == "ensemble":
        return EnsembleReviser(train_data, encoder, device, valid_data, seed)
    elif reviser_type == "expert-label":
        return ExpertLabelReviser(train_data, encoder, device, valid_data, seed)
    elif reviser_type == "mc-dropout":
        return MCDropoutReviser(train_data, encoder, device, valid_data, seed)
    else:
        raise ValueError(f"reviser {reviser_type} not implemented.")


def evaluate_label_quality(labels, probs):
    N = len(labels)
    n_class = probs.shape[1]
    labels = np.array(labels)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(labels, preds)
    # to prevent inf values, use laplace smoothing before evaluting nll
    smoothed_probs = (probs + 1e-6) / (1 + 1e-6 * probs.shape[1])
    nll = - np.log(smoothed_probs[np.arange(N), labels]).mean().item()
    brier_score = (1 - 2 * probs[np.arange(N), labels] + np.sum(probs ** 2, axis=1)) / n_class
    brier_score = brier_score.mean().item()
    return acc, nll, brier_score


def evaluate_end_model(pred_train_data, pred_train_labels, valid_data, test_data, args, seed):
    """
    Evaluate end model's performance with predicted labels
    """
    seed_everything(seed, workers=True)
    end_model = get_end_model(args.end_model)
    n_steps = int(np.ceil(len(pred_train_data) / end_model.hyperparas["batch_size"])) * args.em_epochs
    evaluation_step = int(np.ceil(len(pred_train_data) / end_model.hyperparas["batch_size"]))  # evaluate every epoch
    if valid_data is not None:
        end_model.fit(
            dataset_train=pred_train_data,
            y_train=pred_train_labels,
            dataset_valid=valid_data,
            y_valid=valid_data.labels,
            metric=args.metric,
            device=args.device,
            verbose=args.verbose,
            n_steps=n_steps,
            evaluation_step=evaluation_step
        )
    else:
        end_model.fit(
            dataset_train=pred_train_data,
            y_train=pred_train_labels,
            metric=args.metric,
            device=args.device,
            verbose=args.verbose,
            n_steps=n_steps,
            evaluation_step=evaluation_step
        )
    test_acc = end_model.test(test_data, "acc", device=args.device)
    test_f1 = end_model.test(test_data, "f1_macro", device=args.device)
    perf = {
        "test_acc": test_acc,
        "test_f1": test_f1
    }
    return perf, end_model


def plot_performance(dataset, cov, est_acc, est_epsilon, gt_acc, est_f1, test_f1, sample_size, output_path):
    """
    Plot performance estimation that shows the tradeoff between accuracy and coverage
    """
    plt.rcParams.update({'font.size': 14})
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(cov, est_acc, label="Estimated Label Acc")
    est_acc_lb = np.clip(est_acc-est_epsilon, 0, 1)
    est_acc_ub = np.clip(est_acc+est_epsilon, 0, 1)
    ax1.fill_between(cov, est_acc_lb, est_acc_ub, alpha=0.1)
    ax1.plot(cov, gt_acc, label="GT Label Acc")
    ax2.plot(cov, est_f1, 'go', label="Valid F1 score")
    ax2.plot(cov, test_f1, 'k^', label="Test F1 score")
    ax1.set_xlabel("Label Coverage")
    ax1.set_ylabel("Label Accuracy")
    ax2.set_ylabel("F1 score")
    # ax1.legend(loc=0)
    # ax2.legend(loc=0)
    ax1.set_title(f"{dataset}")
    fig.tight_layout()
    figpath = Path(output_path) / dataset / f"performance_plot_{sample_size}.jpg"
    plt.savefig(figpath)


def get_filter_probs(dataset, label_model, reviser, lm_theta, rm_theta):
    """
    Get the prediction result with filtering
    """
    lm_probs = label_model.predict_proba(dataset)
    rm_probs = reviser.predict_proba(dataset)
    lm_conf = np.max(lm_probs, axis=1)
    rm_conf = np.max(rm_probs, axis=1)
    lm_act = lm_conf >= lm_theta
    rm_act = (~lm_act) & (rm_conf >= rm_theta)
    act = lm_act | rm_act
    filter_probs = lm_probs
    filter_probs[rm_act,:] = rm_probs[rm_act, :]
    filter_probs = filter_probs[act, :]
    active_indices = np.nonzero(act)[0]
    return filter_probs, active_indices


def estimate_cov_acc_tradeoff(train_data, valid_data, label_model, reviser, max_theta_size=10, alpha=0.05):
    """
    Estimate the coverage-accuracy tradeoff using validation set.
    max_theta_size: discrete size for thresholds
    alpha: risk that label accuracy fall below lower bound
    """

    lm_probs = label_model.predict_proba(train_data)
    lm_conf = np.trunc(np.max(lm_probs, axis=1) * 1000) / 1000
    lm_conf_thres = np.unique(lm_conf)
    if len(lm_conf_thres) > max_theta_size:
        K = int(len(lm_conf_thres) / max_theta_size)
        lm_conf_thres = lm_conf_thres[::K]

    lm_conf_thres = np.append(lm_conf_thres, [1.1])

    rm_probs = reviser.predict_proba(train_data)
    rm_conf = np.trunc(np.max(rm_probs, axis=1) * 1000) / 1000
    rm_conf_thres = np.unique(rm_conf)
    if len(rm_conf_thres) > max_theta_size:
        K = int(len(rm_conf_thres) / max_theta_size)
        rm_conf_thres = rm_conf_thres[::K]

    rm_conf_thres = np.append(rm_conf_thres, [1.1])
    acc_mat = np.zeros((len(lm_conf_thres), len(rm_conf_thres)))
    epsilon_mat = np.zeros((len(lm_conf_thres), len(rm_conf_thres)))
    cov_mat = np.zeros_like(acc_mat)
    for i in range(len(lm_conf_thres)):
        for j in range(len(rm_conf_thres)):
            lm_theta = lm_conf_thres[i]
            rm_theta = rm_conf_thres[j]
            _, train_act_indices = get_filter_probs(train_data, label_model, reviser, lm_theta, rm_theta)
            coverage = len(train_act_indices) / len(train_data)
            valid_act_probs, valid_act_indices = get_filter_probs(valid_data, label_model, reviser, lm_theta, rm_theta)
            if len(valid_act_indices) > 0:
                valid_act_labels = np.array(valid_data.labels)[valid_act_indices]
                acc, nll, brier = evaluate_label_quality(valid_act_labels, valid_act_probs)
                n_valid = len(valid_act_indices)
                epsilon = np.sqrt(np.log(alpha/2) / (-2*n_valid))

            else:
                acc = 0.0
                epsilon = np.nan

            acc_mat[i,j] = acc
            epsilon_mat[i,j] = epsilon
            cov_mat[i,j] = coverage

    perf = {
        "lm_theta": lm_conf_thres,
        "rm_theta": rm_conf_thres,
        "acc": acc_mat,
        "epsilon": epsilon_mat,  # half length of confidence interval for acc
        "cov": cov_mat
    }
    return perf


def select_thresholds(train_data, valid_data, test_data, candidate_theta_list, label_model, reviser, args, seed):
    """
    Select the best combination of thresholds using performance on validation set
    """
    def get_f1(theta_idx):
        lm_theta, rm_theta = candidate_theta_list[theta_idx]
        train_act_probs, train_act_indices = get_filter_probs(train_data, label_model, reviser, lm_theta, rm_theta)
        pred_train_data = train_data.create_subset(train_act_indices)
        if args.use_soft_labels:
            pred_train_labels = train_act_probs
        else:
            pred_train_labels = train_act_probs.argmax(axis=1)
        valid_perf, _ = evaluate_end_model(pred_train_data, pred_train_labels, valid_data, valid_data, args, seed)
        if args.plot_performance:
            test_perf, _ = evaluate_end_model(pred_train_data, pred_train_labels, valid_data, test_data, args, seed)
            return valid_perf["test_f1"], test_perf["test_f1"]
        else:
            return valid_perf["test_f1"], np.nan

    best_f1 = 0
    selected_theta = None
    valid_f1_list = np.repeat(-1.0, len(candidate_theta_list))
    test_f1_list = np.repeat(-1.0, len(candidate_theta_list))
    if args.theta_explore_strategy in ["exhaustive", "random", "step"]:
        indices = np.arange(len(candidate_theta_list))
        if args.theta_explore_strategy == "random":
            selected_indices = np.random.choice(indices, size=args.theta_explore_num, replace=False)
        elif args.theta_explore_strategy == "step":
            step = np.ceil(len(candidate_theta_list) / args.theta_explore_num).astype(int)
            selected_indices = indices[::step]
        else:
            selected_indices = indices
        # make sure the first and last element exist in selected indices
        if 0 not in selected_indices:
            selected_indices = np.concatenate(([0], selected_indices))
        if indices[-1] not in selected_indices:
            selected_indices = np.concatenate((selected_indices, [indices[-1]]))

        for i in selected_indices:
            valid_f1, test_f1 = get_f1(i)
            lm_theta, rm_theta = candidate_theta_list[i]
            valid_f1_list[i] = valid_f1
            test_f1_list[i] = test_f1
            if valid_f1 > best_f1:
                best_f1 = valid_f1
                selected_theta = (lm_theta, rm_theta)

    # elif args.theta_explore_strategy == "ternery":
    #     l = 0
    #     r = len(candidate_theta_list)
    #     while r - l > 2:
    #         m1 = l + (r-l) // 3
    #         m2 = r - (r-l) // 3
    #         f1_1 = get_valid_f1(m1, f1_buffer)
    #         f1_buffer[m1] = f1_1
    #         f1_2 = get_valid_f1(m2, f1_buffer)
    #         f1_buffer[m2] = f1_2
    #         if f1_1 > f1_2:
    #             r = m2
    #         elif f1_1 < f1_2:
    #             l = m1
    #         else:
    #             l = m1
    #             r = m2
    #     best_f1 = get_valid_f1(l, f1_buffer)
    #     selected_theta = candidate_theta_list[l]

    return selected_theta, valid_f1_list, test_f1_list


def calibrate_lm_threshold(valid_data, label_model, desired_acc=0.95):
    lb = 0.5
    rb = 1.0
    lm_probs = label_model.predict_proba(valid_data)
    lm_preds = label_model.predict(valid_data)
    lm_conf = np.max(lm_probs, axis=1)
    while rb - lb > 0.01:
        mid = (lb + rb) / 2
        selected_indices = np.nonzero(lm_conf > mid)[0]
        if len(selected_indices) == 0:
            rb = mid
        else:
            acc = accuracy_score(np.array(valid_data.labels)[selected_indices], lm_preds[selected_indices])
            n_valid = len(selected_indices)
            # epsilon = np.sqrt(np.log(alpha) / (-2 * n_valid))
            if acc >= desired_acc:
                rb = mid
            else:
                lb = mid
    return rb


def save_results(results_list, output_path, dataset, filename):
    filepath = Path(output_path) / dataset / filename
    dirname = Path(output_path) / dataset
    Path(dirname).mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as write_file:
        json.dump({"data": results_list}, write_file, indent=4)


def update_results(results_dict, **kwargs):
    for key in kwargs:
        results_dict[key].append(kwargs[key])


def plot_tsne(features, labels, labeled_indices, title):
    if len(features) > 1000:
        # random sampling
        indices = np.random.choice(len(features), 1000, replace=False)
        features = features[indices, :]
        labels = np.array(labels)[indices]
    else:
        indices = np.arange(len(features))

    pca = PCA(n_components=2)
    sc = StandardScaler()
    features = sc.fit_transform(features)
    features = pca.fit_transform(features)
    colors = []
    color_list = ["blue", "orange", "yellow", "cyan"]
    for i in range(len(indices)):
        if indices[i] in labeled_indices:
            colors.append(color_list[labels[i]])
        else:
            colors.append("grey")

    plt.figure()
    plt.scatter(features[:,0], features[:,1], c=colors)
    plt.title(title)
    plt.show()