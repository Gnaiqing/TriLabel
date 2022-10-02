import numpy as np
from wrench.labelmodel import Snorkel, DawidSkene, MajorityVoting, ActiveWeasulModel
from wrench.endmodel import EndClassifierModel
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

ABSTAIN = -1


def get_label_model(model_type):
    if model_type == "snorkel":
        label_model = Snorkel(lr=0.01, l2=0.0, n_epochs=100)
    elif model_type == "ds":
        label_model = DawidSkene()
    elif model_type == "mv":
        label_model = MajorityVoting()
    elif model_type == "aw-metal":
        label_model = ActiveWeasulModel(active_learning=False, penalty_strength=0.0)
    else:
        raise ValueError(f"label model {model_type} not supported.")
    return label_model


def get_end_model(model_type):
    if model_type == "mlp":
        end_model = EndClassifierModel(
            backbone="MLP",
            batch_size=256,
            test_batch_size=512,
            n_steps=5000,
            optimizer="Adam",
            optimizer_lr=0.01,
            optimizer_weight_decay=0.0001)
    elif model_type == "bert":
        end_model = EndClassifierModel(
            batch_size=32,
            real_batch_size=32,
            test_batch_size=512,
            n_steps=1000,
            backbone="BERT",
            backbone_model_name="bert-base-cased",
            backbone_max_tokens=128,
            backbone_fine_tune_layers=-1,
            optimizer="AdamW",
            optimizer_lr=0.00005,
            optimizer_weight_decay=0.0
            )
    else:
        raise ValueError(f"end model {model_type} not implemented.")
    return end_model


def score(y_true, y_pred, metric):
    if metric == "acc":
        score = accuracy_score(y_true, y_pred)
    else:
        raise ValueError(f"Metric {metric} not supported")
    return score


def evaluate_performance(train_data, valid_data, test_data, args, seed):
    seed_everything(seed, workers=True)  # reproducibility for LM & EM
    covered_train_data = train_data.get_covered_subset()
    covered_valid_data = valid_data.get_covered_subset()
    label_model = get_label_model(args.label_model)
    label_model.fit(dataset_train=covered_train_data)
    train_coverage = len(covered_train_data) / len(train_data)
    pred_train_labels = label_model.predict(covered_train_data)
    pred_valid_labels = label_model.predict(covered_valid_data)
    train_covered_acc = score(covered_train_data.labels, pred_train_labels, "acc")

    covered_test_data = test_data.get_covered_subset()
    test_coverage = len(covered_test_data) / len(test_data)
    pred_test_labels = label_model.predict(covered_test_data)
    test_covered_acc = score(covered_test_data.labels, pred_test_labels, "acc")

    lm_test = label_model.test(test_data, args.metric, test_data.labels)
    if args.end_model is not None:
        end_model = get_end_model(args.end_model)
        if args.use_soft_labels:
            aggregated_labels = label_model.predict_proba(covered_train_data)
        else:
            aggregated_labels = label_model.predict(covered_train_data)
        end_model.fit(
            dataset_train=covered_train_data,
            y_train=aggregated_labels,
            dataset_valid=covered_valid_data,
            y_valid=pred_valid_labels,
            evaluation_step=100,
            metric=args.metric,
            patience=500,
            device=args.device,
            verbose=True
        )
        em_test = end_model.test(test_data, args.metric, device=args.device)

    else:
        em_test = np.nan

    perf = {
        "train_coverage": train_coverage,
        "train_covered_acc": train_covered_acc,
        "test_coverage": test_coverage,
        "test_covered_acc": test_covered_acc,
        "lm_test": lm_test,
        "em_test": em_test
    }
    return perf


def evaluate_golden_performance(train_data, valid_data, test_data, args, seed):
    assert args.end_model is not None
    seed_everything(seed, workers=True)  # reproducibility for LM & EM
    end_model = get_end_model(args.end_model)
    end_model.fit(
        dataset_train=train_data,
        y_train=train_data.labels,
        dataset_valid=valid_data,
        evaluation_step=100,
        metric=args.metric,
        patience=500,
        device=args.device,
        verbose=True
    )
    em_test = end_model.test(test_data, args.metric, device=args.device)
    perf = {
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
        features = features[indices,:]
        labels = np.array(labels)[indices]

    if pca_reduction:
        pca = PCA(n_components=128)
        sc = StandardScaler()
        features = sc.fit_transform(features)
        features = pca.fit_transform(features)

    X = TSNE(perplexity=perplexity, init="pca", learning_rate="auto").fit_transform(features)
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=labels)
    plt.title(title)
    dirname = Path(figure_path) / dataset
    Path(dirname).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(dirname, f"tsne_{title}.jpg")
    plt.savefig(filename)


def save_results(results_list, output_path, dataset, filename):
    filepath = Path(output_path) / dataset / filename

    with open(filepath, "w") as write_file:
        json.dump({"data": results_list}, write_file, indent=4)


def plot_results(results_list, figure_path, dataset, title, metric):
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
        "lm_test": [],
        "em_test": [],
        "em_test_golden": []
    }

    for i in range(n_run):
        for key in res:
            res[key].append(results_list[i][key])

    for key in res:
        res[key] = np.array(res[key])

    x = results_list[0]["labeled"]
    y = res["train_coverage"].mean(axis=0)
    y_stderr = res["train_coverage"].std(axis=0) / np.sqrt(n_run)
    ax.plot(x, y, label="train_coverage", c="b")
    ax.fill_between(x, y-1.96*y_stderr, y+1.96*y_stderr, alpha=.1, color="b")

    y = res["train_covered_acc"].mean(axis=0)
    y_stderr = res["train_covered_acc"].std(axis=0) / np.sqrt(n_run)
    ax.plot(x, y, label="train_covered_acc", c="r")
    ax.fill_between(x, y - 1.96 * y_stderr, y + 1.96 * y_stderr, alpha=.1, color="r")

    ax.set_xlabel("label budget")
    ax.set_ylabel("accuracy/coverage")
    ax.set_title(title)
    ax.legend()
    dirname = Path(figure_path) / dataset
    Path(dirname).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(dirname, f"result_train_{title}.jpg")
    fig.savefig(filename)

    # then plot LM and EM's performance on test set
    fig, ax = plt.subplots()
    ax.axhline(y=res["em_test_golden"].mean(), color='k', linestyle='--')
    y = res["lm_test"].mean(axis=0)
    y_stderr = res["lm_test"].std(axis=0) / np.sqrt(n_run)
    ax.plot(x, y, label="Label Model Accuracy", c="r")
    ax.fill_between(x, y - 1.96 * y_stderr, y + 1.96 * y_stderr, alpha=.1, color="r")

    y = res["em_test"].mean(axis=0)
    y_stderr = res["em_test"].std(axis=0) / np.sqrt(n_run)
    ax.plot(x, y, label="End Model Accuracy", c="b")
    ax.fill_between(x, y - 1.96 * y_stderr, y + 1.96 * y_stderr, alpha=.1, color="b")

    ax.set_xlabel("label budget")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend()
    filename = os.path.join(dirname, f"result_test_{title}.jpg")
    fig.savefig(filename)





