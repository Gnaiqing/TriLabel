import numpy as np
from wrench.labelmodel import Snorkel, DawidSkene, MajorityVoting, MeTaL
from label_model.label_model import LabelModel
from wrench.endmodel import EndClassifierModel, LogRegModel, Cosine
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy.random import default_rng
from pytorch_lightning import seed_everything
from pathlib import Path
import os
import json
import torch

ABSTAIN = -1


def get_label_model(model_type, **kwargs):
    if model_type == "snorkel":
        label_model = Snorkel(lr=0.001, l2=0.0, n_epochs=1000)
    elif model_type == "ds":
        label_model = DawidSkene()
    elif model_type == "mv":
        label_model = MajorityVoting()
    elif model_type == "metal":
        label_model = MeTaL(lr=0.001, n_epochs=1000)
    elif model_type == "aw":
        if "penalty_strength" in kwargs:
            label_model = LabelModel(n_epochs=1000, lr=0.001, active_learning=False,
                                     penalty_strength=kwargs["penalty_strength"])
        else:
            label_model = LabelModel(n_epochs=1000, lr=0.001, active_learning=False)
    else:
        raise ValueError(f"label model {model_type} not supported.")
    return label_model


def get_lf(lf_class, seed=None):
    if lf_class == "logistic":
        clf = LogisticRegression(random_state=seed, max_iter=500)
    elif lf_class == "linear-svm":
        clf = SVC(kernel="linear", probability=True, random_state=seed)
    elif lf_class == "dt":
        clf = DecisionTreeClassifier(random_state=seed)
    else:
        raise ValueError(f"LF Class {lf_class} not supported yet.")
    return clf


def get_revision_model_kwargs(revision_model_class):
    if revision_model_class == "voting":
        kwargs = {
            "base_classifiers": [("logistic", {"max_iter": 500}),
                                 ("linear-svm", {}),
                                 ("decision-tree", {})]
        }
    elif revision_model_class == "logistic":
        kwargs = {"max_iter": 500}
    else:
        kwargs = {}
    return kwargs


def get_revision_model(revision_model_class, seed=None, **kwargs):
    if revision_model_class == "logistic":
        clf = LogisticRegression(random_state=seed, **kwargs)
    elif revision_model_class == "linear-svm":
        clf = SVC(kernel="linear", probability=True, random_state=seed, **kwargs)
    elif revision_model_class == "decision-tree":
        clf = DecisionTreeClassifier(random_state=seed, **kwargs)
    elif revision_model_class == "random-forest":
        clf = RandomForestClassifier(random_state=seed, **kwargs)
    elif revision_model_class == "voting":
        estimators = []
        for (base_classifier, base_classifier_args) in kwargs["base_classifiers"]:
            baseclf = get_revision_model(base_classifier, seed=seed, **base_classifier_args)
            estimators.append((base_classifier, baseclf))
        clf = VotingClassifier(estimators=estimators)
    elif revision_model_class == "expert-label":
        clf = None
    else:
        raise ValueError(f"Revision model {revision_model_class} not supported yet.")
    return clf


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


def get_sampler(sampler_type, train_data, labeller, **kwargs):
    from sampler.passive import PassiveSampler
    from sampler.lfcov import LFCovSampler
    from sampler.uncertain import UncertaintySampler
    from sampler.maxkl import MaxKLSampler
    if sampler_type == "passive":
        return PassiveSampler(train_data, labeller, **kwargs)
    elif sampler_type == "lfcov":
        return LFCovSampler(train_data, labeller, **kwargs)
    elif sampler_type == "uncertain":
        return UncertaintySampler(train_data, labeller, kwargs["label_model"])
    elif sampler_type == "maxkl":
        if "penalty_strength" in kwargs:
            return MaxKLSampler(train_data, labeller, kwargs["label_model"], penalty_strength=kwargs["penalty_strength"])
        else:
            return MaxKLSampler(train_data, labeller, kwargs["label_model"])
    else:
        raise ValueError(f"sampler {sampler_type} not implemented.")


def score(y_true, y_pred, metric):
    if metric == "acc":
        score = accuracy_score(y_true, y_pred)
    else:
        raise ValueError(f"Metric {metric} not supported")
    return score


def evaluate_performance(train_data, valid_data, test_data, args, seed,
                         ground_truth_labels=None,
                         rm_predict_labels=None):
    """
    Evaluate the performance of weak supervision pipeline
    """
    seed_everything(seed, workers=True)  # reproducibility for LM & EM
    covered_train_data = train_data.get_covered_subset()

    if args.label_model == "aw":
        if hasattr(args, "penalty_strength"):
            label_model = get_label_model(args.label_model, penalty_strength=args.penalty_strength)
        else:
            label_model = get_label_model(args.label_model)
        # active WeaSuL model use ground truth labels to tune parameters
        if ground_truth_labels is not None:
            label_model.active_learning = True
        if args.use_valid_labels:
            label_model.fit(dataset_train=train_data,
                            dataset_valid=valid_data,
                            y_valid=valid_data.labels,
                            ground_truth_labels=ground_truth_labels)
        else:
            label_model.fit(
                dataset_train=train_data,
                ground_truth_labels=ground_truth_labels
            )
    else:
        label_model = get_label_model(args.label_model)
        if args.use_valid_labels:
            label_model.fit(dataset_train=covered_train_data,
                            dataset_valid=valid_data,
                            y_valid=valid_data.labels)
        else:
            # use majority voting to estimate valid labels
            mv = get_label_model("mv")
            mv.fit(dataset_train=covered_train_data)
            covered_valid_data = valid_data.get_covered_subset()
            pred_valid_labels = mv.predict(covered_valid_data)
            label_model.fit(dataset_train=covered_train_data,
                            dataset_valid=covered_valid_data,
                            y_valid=pred_valid_labels)

    if args.revision_type == "pre" or rm_predict_labels is None:
        # use the predicted label of label model
        train_coverage = len(covered_train_data) / len(train_data)
        pred_train_labels = label_model.predict(covered_train_data)
        train_covered_acc = score(covered_train_data.labels, pred_train_labels, "acc")
        end_model = get_end_model(args.end_model)
        if args.use_soft_labels:
            aggregated_labels = label_model.predict_proba(covered_train_data)
        else:
            aggregated_labels = label_model.predict(covered_train_data)
    else:
        # use labels predicted by revision model to correct labels predicted by LM
        weak_labels = np.array(train_data.weak_labels)
        covered_mask = np.any(weak_labels != ABSTAIN, axis=1)
        pred_mask = rm_predict_labels != ABSTAIN
        pred_indices = np.nonzero(pred_mask)[0]
        covered_indices = np.nonzero(covered_mask | pred_mask)[0]
        covered_train_data = train_data.create_subset(covered_indices)
        _, pred_pos, covered_pos = np.intersect1d(pred_indices, covered_indices, return_indices=True)
        pred_train_labels = label_model.predict(covered_train_data)
        pred_train_labels[covered_pos] = rm_predict_labels[pred_mask]
        train_coverage = len(covered_train_data) / len(train_data)
        train_covered_acc = score(covered_train_data.labels, pred_train_labels, "acc")
        end_model = get_end_model(args.end_model)
        if args.use_soft_labels:
            aggregated_labels = label_model.predict_proba(covered_train_data)
            aggregated_labels[covered_pos, :] = 0.0
            aggregated_labels[covered_pos, rm_predict_labels[pred_mask]] = 1.0
        else:
            aggregated_labels = pred_train_labels

    n_steps = int(np.ceil(len(covered_train_data) / end_model.hyperparas["batch_size"])) * args.em_epochs
    evaluation_step = int(np.ceil(len(covered_train_data) / end_model.hyperparas["batch_size"])) # evaluate every epoch

    if args.use_valid_labels:
        end_model.fit(
            dataset_train=covered_train_data,
            y_train=aggregated_labels,
            dataset_valid=valid_data,
            y_valid=valid_data.labels,
            metric=args.metric,
            device=args.device,
            verbose=True,
            n_steps=n_steps,
            evaluation_step=evaluation_step
        )

    else:
        covered_valid_data = valid_data.get_covered_subset()
        pred_valid_labels = label_model.predict(covered_valid_data)
        end_model.fit(
            dataset_train=covered_train_data,
            y_train=aggregated_labels,
            dataset_valid=covered_valid_data,
            y_valid=pred_valid_labels,
            metric=args.metric,
            device=args.device,
            verbose=True,
            n_steps=n_steps,
            evaluation_step=evaluation_step
        )

    em_test = end_model.test(test_data, args.metric, device=args.device)
    perf = {
        "train_coverage": train_coverage,
        "train_covered_acc": train_covered_acc,
        "em_test": em_test
    }
    if rm_predict_labels is not None:
        # record the coverage and accuracy of RM predict labels
        rm_coverage = np.sum(rm_predict_labels != ABSTAIN) / len(rm_predict_labels)
        rm_accuracy = np.sum(rm_predict_labels == np.array(train_data.labels)) / np.sum(rm_predict_labels != ABSTAIN)
        perf["rm_coverage"] = rm_coverage
        perf["rm_covered_acc"] = rm_accuracy
    else:
        perf["rm_coverage"] = np.nan
        perf["rm_covered_acc"] = np.nan

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
        verbose=True,
        n_steps=n_steps,
        evaluation_step=evaluation_step
    )
    em_test = end_model.test(test_data, args.metric, device=args.device)
    perf = {
        "train_coverage": 1.0,
        "train_covered_acc": 1.0,
        "em_test": em_test,
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


def plot_LF_activation(dataset, lf_idx, figure_path, dataset_name, title,
                       encoder=None, perplexity=5.0, pca_reduction=False):
    """
    Plot the region where an LF is activated. See where it makes correct/wrong predictions.
    :param dataset: dataset to inspect
    :param lf_idx: LF to inspect
    :param title: title of plot
    :return:
    """
    L = np.array(dataset.weak_labels)
    active_mask = L[:, lf_idx] != ABSTAIN
    if encoder is None:
        features = dataset.features[active_mask, :]
    else:
        features = encoder(torch.tensor(dataset.features[active_mask, :])).detach().cpu().numpy()

    labels = np.array(dataset.labels)[active_mask]
    weak_labels = L[active_mask, lf_idx]
    y = labels == weak_labels
    plot_tsne(features, y, figure_path, dataset_name, title, perplexity=perplexity, pca_reduction=pca_reduction)


def plot_results(results_list, figure_path, dataset, title, filename, plot_labeled_frac=False):
    """
    Plot pipeline results
    :param results:
    :return:
    """
    # first plot LM's coverage and covered accuracy on train set
    n_run = len(results_list)
    fig, ax = plt.subplots()
    res = {
        "train_coverage": [],
        "train_covered_acc": [],
        "em_test": [],
        "em_test_golden": [],
        "rm_coverage": [],
        "rm_covered_acc": []
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

    y = res["train_coverage"].mean(axis=0)
    y_stderr = res["train_coverage"].std(axis=0) / np.sqrt(n_run)
    ax.plot(x, y, label="Train label coverage", c="b")
    ax.fill_between(x, y - 1.96 * y_stderr, y + 1.96 * y_stderr, alpha=.1, color="b")

    y = res["train_covered_acc"].mean(axis=0)
    y_stderr = res["train_covered_acc"].std(axis=0) / np.sqrt(n_run)
    ax.plot(x, y, label="Train label accuracy", c="r")
    ax.fill_between(x, y - 1.96 * y_stderr, y + 1.96 * y_stderr, alpha=.1, color="r")

    ax.axhline(y=res["em_test_golden"].mean(), color='k', linestyle='--')
    y = res["em_test"].mean(axis=0)
    y_stderr = res["em_test"].std(axis=0) / np.sqrt(n_run)
    ax.plot(x, y, label="Test set accuracy (EM)", c="g")
    ax.fill_between(x, y - 1.96 * y_stderr, y + 1.96 * y_stderr, alpha=.1, color="g")

    # if not np.isnan(res["rm_coverage"]).all():
    #     y = res["rm_coverage"].mean(axis=0)
    #     y_stderr = res["rm_coverage"].std(axis=0) / np.sqrt(n_run)
    #     ax.plot(x, y, label="Revision model coverage", c="orange")
    #     ax.fill_between(x, y - 1.96 * y_stderr, y + 1.96 * y_stderr, alpha=.1, color="orange")
    #
    #     y = res["rm_covered_acc"].mean(axis=0)
    #     y_stderr = res["rm_covered_acc"].std(axis=0) / np.sqrt(n_run)
    #     ax.plot(x, y, label="Revision model accuracy", c="cyan")
    #     ax.fill_between(x, y - 1.96 * y_stderr, y + 1.96 * y_stderr, alpha=.1, color="cyan")

    ax.set_xlabel("label budget")
    ax.set_title(title)
    ax.legend()
    dirname = Path(figure_path) / dataset
    Path(dirname).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(dirname, filename)
    fig.savefig(filename)
