import numpy as np
from wrench.labelmodel import Snorkel, DawidSkene, MajorityVoting, MeTaL
from active_weasul.label_model import LabelModel
from wrench.endmodel import EndClassifierModel, LogRegModel, Cosine
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy.random import default_rng
from pytorch_lightning import seed_everything
from pathlib import Path
from scipy.stats import entropy
import os
import json

ABSTAIN = -1


def get_label_model(model_type, **kwargs):
    if model_type == "snorkel":
        label_model = Snorkel(lr=0.01, l2=0.0, n_epochs=100)
    elif model_type == "ds":
        label_model = DawidSkene()
    elif model_type == "mv":
        label_model = MajorityVoting()
    elif model_type == "metal":
        label_model = MeTaL(lr=0.01, n_epochs=100)
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
            batch_size=512,
            test_batch_size=512,
            optimizer="Adam",
            optimizer_lr=1e-2,
            optimizer_weight_decay=1e-5
        )
    elif model_type == "logistic":
        end_model = LogRegModel(
            lr=1e-2,
            batch_size=512,
            test_batch_size=512
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
    elif model_type == "cosine-bert":
        end_model = Cosine(
            batch_size=32,
            real_batch_size=32,  # for accumulative gradient update
            test_batch_size=32,
            lamda=0.1,
            backbone='BERT',
            backbone_model_name='bert-base-cased',
            backbone_max_tokens=128,
            backbone_fine_tune_layers=-1,  # fine  tune all
            optimizer='AdamW',
            optimizer_lr=5e-5,
            optimizer_weight_decay=1e-4,
    )
    elif model_type == "cosine-roberta":
        end_model = Cosine(
            batch_size=32,
            real_batch_size=32,  # for accumulative gradient update
            test_batch_size=32,
            lamda=0.1,
            backbone='BERT',
            backbone_model_name='roberta-base',
            backbone_max_tokens=128,
            backbone_fine_tune_layers=-1,  # fine  tune all
            optimizer='AdamW',
            optimizer_lr=5e-5,
            optimizer_weight_decay=1e-4,
        )
    else:
        raise ValueError(f"end model {model_type} not implemented.")
    return end_model


def get_sampler(sampler_type, train_data, labeller, label_model, revision_model, encoder, seed=None):
    from sampler import PassiveSampler, UncertaintySampler, MaxKLSampler, \
        RmUncertaintySampler, DALSampler, DisagreementSampler, AbstainSampler, JointUncertaintySampler
    if sampler_type == "passive":
        return PassiveSampler(train_data, labeller, label_model, revision_model, encoder, seed=seed)
    elif sampler_type == "uncertain":
        return UncertaintySampler(train_data, labeller, label_model, revision_model, encoder, seed=seed)
    elif sampler_type == "uncertain-rm":
        return RmUncertaintySampler(train_data, labeller, label_model, revision_model, encoder, seed=seed)
    elif sampler_type == "uncertain-joint":
        return JointUncertaintySampler(train_data, labeller, label_model, revision_model, encoder, seed=seed)
    elif sampler_type == "dal":
        return DALSampler(train_data, labeller, label_model, revision_model, encoder, seed=seed)
    elif sampler_type == "abstain":
        return AbstainSampler(train_data, labeller, label_model, revision_model, encoder, seed=seed)
    elif sampler_type == "disagreement":
        return DisagreementSampler(train_data, labeller, label_model, revision_model, encoder, seed=seed)
    elif sampler_type == "maxkl":
        return MaxKLSampler(train_data, labeller,label_model, revision_model, encoder, seed=seed)
    else:
        raise ValueError(f"sampler {sampler_type} not implemented.")


def get_reviser(reviser_type, train_data, valid_data, encoder,  device, seed):
    from reviser import EnsembleReviser, ExpertLabelReviser, MCDropoutReviser, \
        MLPReviser, MLPTempReviser, DalenReviser
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
    elif reviser_type == "dalen":
        return DalenReviser(train_data, encoder, device, valid_data, seed)
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


def evaluate_performance(train_data, valid_data, test_data, aggregated_soft_labels, args, seed):
    """
    Evaluate the end model's performance after revising labels
    """
    seed_everything(seed, workers=True)
    nan_pos = np.isnan(aggregated_soft_labels) # replace nan with uniform distribution
    aggregated_soft_labels[nan_pos] = 1 / train_data.n_class
    label_acc, label_nll, label_brier = evaluate_label_quality(train_data.labels, aggregated_soft_labels)
    aggregated_hard_labels = np.argmax(aggregated_soft_labels, axis=1)
    end_model = get_end_model(args.end_model)
    n_steps = int(np.ceil(len(train_data) / end_model.hyperparas["batch_size"])) * args.em_epochs
    evaluation_step = int(np.ceil(len(train_data) / end_model.hyperparas["batch_size"]))  # evaluate every epoch
    if args.use_soft_labels:
        aggregated_labels = aggregated_soft_labels
    else:
        aggregated_labels = aggregated_hard_labels

    if args.use_valid_labels:
        end_model.fit(
            dataset_train=train_data,
            y_train=aggregated_labels,
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
            dataset_train=train_data,
            y_train=aggregated_labels,
            metric=args.metric,
            device=args.device,
            verbose=args.verbose,
            n_steps=n_steps,
            evaluation_step=evaluation_step
        )
    test_acc = end_model.test(test_data, "acc", device=args.device)
    test_f1 = end_model.test(test_data, "f1_macro", device=args.device)
    perf = {
        "label_acc": label_acc,
        "label_nll": label_nll,
        "label_brier": label_brier,
        "test_acc": test_acc,
        "test_f1": test_f1
    }
    return perf


def evaluate_golden_performance(train_data, valid_data, test_data, args, seed):
    seed_everything(seed, workers=True)  # reproducibility for LM & EM
    end_model = get_end_model(args.end_model)
    n_steps = int(np.ceil(len(train_data) / end_model.hyperparas["batch_size"])) * args.em_epochs
    evaluation_step = int(np.ceil(len(train_data) / end_model.hyperparas["batch_size"]))  # evaluate every epoch
    end_model.fit(
        dataset_train=train_data,
        y_train=train_data.labels,
        dataset_valid=valid_data,
        y_valid=valid_data.labels,
        metric=args.metric,
        device=args.device,
        verbose=args.verbose,
        n_steps=n_steps,
        evaluation_step=evaluation_step
    )

    test_acc = end_model.test(test_data, "acc", device=args.device)
    test_f1 = end_model.test(test_data, "f1_macro", device=args.device)
    perf = {
        "label_acc": 1.0,
        "label_nll": 0.0,
        "label_brier": 0.0,
        "test_acc": test_acc,
        "test_f1": test_f1
    }
    return perf


def evaluate_al_performance(train_data, valid_data, test_data, args, seed,
                            ground_truth_labels=None):
    # evaluate the performance of active learning with sampled ground truth labels
    seed_everything(seed, workers=True)
    if ground_truth_labels is None:
        return None

    end_model = get_end_model(args.end_model)
    labeled_indices = np.nonzero(ground_truth_labels != ABSTAIN)[0]
    labeled_train_data = train_data.create_subset(labeled_indices.tolist())
    n_steps = int(np.ceil(len(labeled_train_data) / end_model.hyperparas["batch_size"])) * args.em_epochs
    evaluation_step = int(np.ceil(len(labeled_train_data) / end_model.hyperparas["batch_size"]))  # evaluate every epoch
    if args.use_valid_labels:
        end_model.fit(
            dataset_train=labeled_train_data,
            y_train=labeled_train_data.labels,
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
            dataset_train=labeled_train_data,
            y_train=labeled_train_data.labels,
            metric=args.metric,
            device=args.device,
            verbose=args.verbose,
            n_steps=n_steps,
            evaluation_step=evaluation_step
        )

    train_probs = end_model.predict_proba(train_data, device=args.device)
    uncertainty = entropy(train_probs, axis=1)
    uncertainty_order = np.argsort(uncertainty)[::-1]
    label_acc, label_nll, label_brier = evaluate_label_quality(train_data.labels, train_probs)
    test_acc = end_model.test(test_data, "acc", device=args.device)
    test_f1 = end_model.test(test_data, "f1_macro", device=args.device)
    perf = {
        "label_acc": label_acc,
        "label_nll": label_nll,
        "label_brier": label_brier,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "uncertainty_order": uncertainty_order
    }
    return perf


def evaluate_pl_performance(train_data, valid_data, test_data, args, seed,
                            ground_truth_labels=None, threshold=0.8):
    # evaluate the performance of semi-supervised learning using pseudo labelling
    seed_everything(seed, workers=True)
    if ground_truth_labels is None:
        return None

    end_model = get_end_model(args.end_model)
    labeled_indices = np.nonzero(ground_truth_labels != ABSTAIN)[0]
    labeled_train_data = train_data.create_subset(labeled_indices.tolist())
    n_steps = int(np.ceil(len(labeled_train_data) / end_model.hyperparas["batch_size"])) * args.em_epochs
    evaluation_step = int(np.ceil(len(labeled_train_data) / end_model.hyperparas["batch_size"]))  # evaluate every epoch
    if args.use_valid_labels:
        end_model.fit(
            dataset_train=labeled_train_data,
            y_train=labeled_train_data.labels,
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
            dataset_train=labeled_train_data,
            y_train=labeled_train_data.labels,
            metric=args.metric,
            device=args.device,
            verbose=args.verbose,
            n_steps=n_steps,
            evaluation_step=evaluation_step
        )

    train_probs = end_model.predict_proba(train_data, device=args.device)
    pseudo_labels = np.argmax(train_probs, axis=1)
    pseudo_labels[labeled_indices] = np.array(labeled_train_data.labels) # for labeled data, use provided labels
    train_conf = np.max(train_probs, axis=1)
    pl_indices = np.nonzero((train_conf >= threshold) | (ground_truth_labels != ABSTAIN))[0]
    pl_labels = pseudo_labels[pl_indices]
    pl_train_data = train_data.create_subset(pl_indices.tolist())

    end_model = get_end_model(args.end_model)
    n_steps = int(np.ceil(len(pl_train_data) / end_model.hyperparas["batch_size"])) * args.em_epochs
    evaluation_step = int(np.ceil(len(pl_train_data) / end_model.hyperparas["batch_size"]))  # evaluate every epoch
    if args.use_valid_labels:
        end_model.fit(
            dataset_train=pl_train_data,
            y_train=pl_labels,
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
            dataset_train=pl_train_data,
            y_train=pl_labels,
            metric=args.metric,
            device=args.device,
            verbose=args.verbose,
            n_steps=n_steps,
            evaluation_step=evaluation_step
        )
    train_probs = end_model.predict_proba(train_data, device=args.device)
    label_acc, label_nll, label_brier = evaluate_label_quality(train_data.labels, train_probs)
    test_acc = end_model.test(test_data, "acc", device=args.device)
    test_f1 = end_model.test(test_data, "f1_macro", device=args.device)
    perf = {
        "label_acc": label_acc,
        "label_nll": label_nll,
        "label_brier": label_brier,
        "test_acc": test_acc,
        "test_f1": test_f1,
    }
    return perf


def evaluate_aw_performance(train_data, valid_data, test_data, args, seed,
                            ground_truth_labels=None):
    """
    Evaluate Active WeaSuL's performance with ground truth labels
    """
    seed_everything(seed, workers=True)
    label_model = get_label_model("aw", penalty_strength=args.penalty_strength)
    label_model.fit(dataset_train=train_data, dataset_valid=valid_data, ground_truth_labels=ground_truth_labels)
    aggregated_soft_labels = label_model.predict_proba(train_data)
    nan_pos = np.isnan(aggregated_soft_labels)
    aggregated_soft_labels[nan_pos] = 1 / train_data.n_class
    label_acc, label_nll, label_brier = evaluate_label_quality(train_data.labels, aggregated_soft_labels)
    aggregated_hard_labels = np.argmax(aggregated_soft_labels, axis=1)
    end_model = get_end_model(args.end_model)
    n_steps = int(np.ceil(len(train_data) / end_model.hyperparas["batch_size"])) * args.em_epochs
    evaluation_step = int(np.ceil(len(train_data) / end_model.hyperparas["batch_size"]))  # evaluate every epoch
    if args.use_soft_labels:
        aggregated_labels = aggregated_soft_labels
    else:
        aggregated_labels = aggregated_hard_labels

    if args.use_valid_labels:
        end_model.fit(
            dataset_train=train_data,
            y_train=aggregated_labels,
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
            dataset_train=train_data,
            y_train=aggregated_labels,
            metric=args.metric,
            device=args.device,
            verbose=args.verbose,
            n_steps=n_steps,
            evaluation_step=evaluation_step
        )
    test_acc = end_model.test(test_data, "acc", device=args.device)
    test_f1 = end_model.test(test_data, "f1_macro", device=args.device)
    perf = {
        "label_acc": label_acc,
        "label_nll": label_nll,
        "label_brier": label_brier,
        "test_acc": test_acc,
        "test_f1": test_f1
    }
    return perf


def plot_tsne(features, labels, figure_path, dataset, title, perplexity=5.0, pca_reduction=False):
    """
    Plot t-sne plot of transformed features
    :return:
    """
    if len(features) > 1000:
        rng = default_rng(seed=0)
        indices = rng.choice(len(features), 1000, replace=False)
        features = features[indices, :]
        labels = np.array(labels)[indices]

    if pca_reduction:
        pca = PCA(n_components=128)
        sc = StandardScaler()
        features = sc.fit_transform(features)
        features = pca.fit_transform(features)

    X = TSNE(perplexity=perplexity, init="pca", learning_rate="auto").fit_transform(features)
    color_map = {
        0: "red",
        1: "green",
        2: "yellow",
        3: "blue",
        4: "cyan",
        5: "purple",
        6: "orange",
        7: "lime",
        8: "violet"
    }
    color = [color_map[i] for i in labels]
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=color)
    plt.title(title)
    dirname = Path(figure_path) / dataset
    Path(dirname).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(dirname, f"tsne_{title}.jpg")
    plt.savefig(filename)


def save_results(results_list, output_path, dataset, filename):
    filepath = Path(output_path) / dataset / filename
    dirname = Path(output_path) / dataset
    Path(dirname).mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as write_file:
        json.dump({"data": results_list}, write_file, indent=4)


def plot_dpal_results(results_list, figure_path, dataset, nametag, plot_labeled_frac=False):
    """
    Plot pipeline results (train label accuracy and test set performance)
    :param results:
    :return:
    """
    # first plot LM's coverage and covered accuracy on train set
    n_run = len(results_list)

    res = {
        "n_labeled": [],  # number of expert labeled data
        "frac_labeled": [],  # fraction of expert labeled data
        "al_label_acc": [],  # AL label accuracy
        "dp_label_acc": [],  # DP label accuracy
        "dpal_label_acc": [],  # DPAL label accuracy
        "dpal_test_acc": [],  # end model's test accuracy
        "dpal_test_f1": [],  # end model's test f1 score
        "al_test_acc": [],  # active learning model's test accuracy
        "al_test_f1": [],  # active learning model's test f1
        "golden_test_acc": [],  # end model's test accuracy using golden labels
        "golden_test_f1": [],   # end model's test f1 using golden labels
    }
    for i in range(n_run):
        for key in res:
            res[key].append(results_list[i][key])

    for key in res:
        res[key] = np.array(res[key])

    if plot_labeled_frac:
        x = results_list[0]["frac_labeled"]
    else:
        x = results_list[0]["n_labeled"]

    # plot train label accuracy
    fig, ax = plt.subplots()
    y = res["dp_label_acc"].mean()
    y_stderr = res["dp_label_acc"].std() / np.sqrt(n_run)
    ax.plot(x, np.repeat(y, len(x)), label="DP label accuracy", c="b")
    ax.fill_between(x, np.repeat(y-1.96 * y_stderr, len(x)), np.repeat(y+1.96 * y_stderr, len(x)),alpha=.1, color="b")

    y = res["al_label_acc"].mean(axis=0)
    y_stderr = res["al_label_acc"].std(axis=0) / np.sqrt(n_run)
    ax.plot(x, y, label="AL label accuracy", c="g")
    ax.fill_between(x, y - 1.96 * y_stderr, y + 1.96 * y_stderr, alpha=.1, color="g")

    y = res["dpal_label_acc"].mean(axis=0)
    y_stderr = res["dpal_label_acc"].std(axis=0) / np.sqrt(n_run)
    ax.plot(x, y, label="DPAL label accuracy", c="r")
    ax.fill_between(x, y - 1.96 * y_stderr, y + 1.96 * y_stderr, alpha=.1, color="r")

    ax.set_xlabel("label budget")
    ax.set_ylabel("train label accuracy")
    ax.set_title(dataset+" Train Label Accuracy")
    ax.legend()
    dirname = Path(figure_path) / dataset
    Path(dirname).mkdir(parents=True, exist_ok=True)
    train_fig_path = os.path.join(dirname, nametag + "_TrainLabel.jpg")
    fig.savefig(train_fig_path)

    fig2, ax2 = plt.subplots()
    y = res["test_acc"][:,0].mean()
    y_stderr = res["test_acc"][:,0].std() / np.sqrt(n_run)
    ax2.plot(x, np.repeat(y, len(x)), label="DP test accuracy", c="b")
    ax2.fill_between(x, np.repeat(y - 1.96 * y_stderr, len(x)), np.repeat(y + 1.96 * y_stderr, len(x)), alpha=.1,
                    color="b")

    y = res["em_test_al"].mean(axis=0)
    y_stderr = res["em_test_al"].std(axis=0) / np.sqrt(n_run)
    ax2.plot(x, y, label="AL test accuracy", c="g")
    ax2.fill_between(x, y - 1.96 * y_stderr, y + 1.96 * y_stderr, alpha=.1, color="g")

    y = res["em_test"].mean(axis=0)
    y_stderr = res["em_test"].std(axis=0) / np.sqrt(n_run)
    ax2.plot(x, y, label="DPAL test accuracy", c="r")
    ax2.fill_between(x, y - 1.96 * y_stderr, y + 1.96 * y_stderr, alpha=.1, color="r")

    y = res["em_test_golden"].mean()
    ax2.axhline(y, color='k', linestyle='--', label="Golden test accuracy")
    ax2.set_xlabel("label budget")
    ax2.set_ylabel("end model test accuracy")
    ax2.set_title(dataset+" Test Accuracy")
    ax2.legend()
    test_fig_path = os.path.join(dirname, nametag + "_TestAcc.jpg")
    fig2.savefig(test_fig_path)