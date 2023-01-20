import sys
import copy
from pathlib import Path
import argparse
from dataset.load_dataset import load_real_dataset, load_synthetic_dataset
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
from utils import save_results, get_label_model, ABSTAIN
from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt


def plot_conf_histogram(thresholds, accs, confs, dataset, output_path, tag):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()
    ax.stairs(accs, thresholds, fill=True, color="blue", alpha=0.2, label="True Accuracy")
    ax.stairs(confs, thresholds, fill=True, color="red", alpha=0.2, label="Confidence")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(dataset)
    ax.legend(loc="upper left")
    filepath = Path(output_path) / dataset / f"confhist_{tag}.jpg"
    dirname = Path(output_path) / dataset
    Path(dirname).mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath)


def evaluate_uncertainty_estimation(train_data, valid_data, label_model_type, feature_size, ensemble_size, bootstrap,
                                    repeats, seeds):
    exp_results = {
        "acc": [],
        "nll": [],
        "ece": [],
        "brier": [],
        "entropy": [],
        "binned_info": []
    }
    for k in range(repeats):
        seed_everything(seeds[k], workers=True)
        weak_labels = np.array(train_data.weak_labels)
        valid_weak_labels = np.array(valid_data.weak_labels)
        N, M = weak_labels.shape
        probs = []
        for i in range(ensemble_size):
            if bootstrap:
                selected_indices = np.random.choice(N, N, replace=True)
            else:
                selected_indices = np.random.permutation(N)
            if feature_size < M:
                selected_features = np.random.choice(M, feature_size, replace=False)
            else:
                selected_features = np.arange(M)

            bagging_weak_labels = weak_labels[selected_indices,:]
            bagging_weak_labels = bagging_weak_labels[:, selected_features]
            bagging_valid_weak_labels = valid_weak_labels[:, selected_features]
            bagging_train_data = copy.copy(train_data)
            bagging_train_data.weak_labels = bagging_weak_labels.tolist()
            bagging_valid_data = copy.copy(valid_data)
            bagging_valid_data.weak_labels = bagging_valid_weak_labels.tolist()
            label_model = get_label_model(label_model_type)
            label_model.fit(bagging_train_data, bagging_valid_data)
            bagging_probs = label_model.predict_proba(weak_labels[:, selected_features])
            probs.append(bagging_probs.tolist())

        probs = np.array(probs).mean(axis=0)
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        labels = np.array(train_data.labels)
        acc = accuracy_score(labels, preds)
        nll = - np.log(probs[np.arange(N), labels]).mean()
        brier_score = (1 - 2 * probs[np.arange(N), labels] + np.sum(probs ** 2, axis=1)) / train_data.n_class
        brier_score = brier_score.mean()
        ece = 0
        conf_thres = np.linspace(0.5, 1.0, 11)
        bin_sizes = []
        binned_accs = []
        binned_confs = []
        for i in range(len(conf_thres) - 1):
            mask = (confs <= conf_thres[i+1]) & (confs > conf_thres[i])
            bin_size = np.sum(mask).item()
            if bin_size > 0:
                binned_acc = accuracy_score(labels[mask], preds[mask]).item()
                binned_conf = np.mean(confs[mask]).item()
            else:
                binned_acc = 0.0
                binned_conf = 0.0

            binned_accs.append(binned_acc)
            binned_confs.append(binned_conf)
            bin_sizes.append(bin_size)
            ece += np.abs(binned_acc - binned_conf) * bin_size / N

        ent = entropy(probs, axis=1)
        ent = np.mean(ent)
        results = {
            "acc": acc.item(),
            "nll": nll.item(),
            "ece": ece.item(),
            "brier": brier_score.item(),
            "entropy": ent.item(),
            "binned_info": {
                "thresholds": conf_thres.tolist(),
                "sizes": bin_sizes,
                "accs": binned_accs,
                "confs": binned_confs
            }
        }
        for key in exp_results:
            exp_results[key].append(results[key])

    return exp_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="youtube")
    parser.add_argument("--dataset_path", type=str, default="../datasets/")
    parser.add_argument("--extract_fn", type=str, default="bert")  # method used to extract features
    parser.add_argument("--label_model", type=str, default="metal")
    # other settings
    parser.add_argument("--ensemble_size", type=int, default=10)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--feature_size", type=int, default=None)
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tag", type=str, default="test")
    args = parser.parse_args()

    if args.dataset[:3] == "syn":
        train_data, valid_data, test_data = load_synthetic_dataset(args.dataset)
    else:
        train_data, valid_data, test_data = load_real_dataset(args.dataset_path, args.dataset, args.extract_fn)

    np.random.seed(args.seed)
    run_seeds = np.random.randint(1, 100000, args.repeats)
    if args.feature_size is None:
        args.feature_size = np.array(train_data.weak_labels).shape[1]

    id_tag = f"calib_lm={args.label_model}_en={args.ensemble_size}_bs={args.bootstrap}_ft={args.feature_size}_{args.tag}"

    res = evaluate_uncertainty_estimation(train_data, valid_data, args.label_model, args.feature_size,
                                          args.ensemble_size, args.bootstrap, args.repeats, run_seeds)

    bin_sizes = np.zeros_like(res["binned_info"][0]["sizes"])
    bin_confs = np.zeros_like(bin_sizes, dtype=float)
    bin_accs = np.zeros_like(bin_sizes, dtype=float)
    for k in range(args.repeats):
        bin_sizes += np.array(res["binned_info"][k]["sizes"])
        bin_accs += np.array(res["binned_info"][k]["sizes"]) * np.array(res["binned_info"][k]["accs"])
        bin_confs += np.array(res["binned_info"][k]["sizes"]) * np.array(res["binned_info"][k]["confs"])

    bin_accs = np.divide(bin_accs, bin_sizes, out=np.zeros_like(bin_accs), where=bin_sizes != 0)
    bin_confs = np.divide(bin_confs, bin_sizes, out=np.zeros_like(bin_confs), where=bin_sizes != 0)
    plot_conf_histogram(res["binned_info"][0]["thresholds"],
                        bin_accs,
                        bin_confs,
                        args.dataset,
                        args.output_path,
                        id_tag)

    save_results(res, args.output_path, args.dataset, f"{id_tag}.json")
    if args.verbose:
        acc_list = np.array(res["acc"])
        nll_list = np.array(res["nll"])
        brier_list = np.array(res["brier"])
        ece_list = np.array(res["ece"])
        print(f"Accuracy: {acc_list.mean():.3f}({acc_list.std():.3f})")
        print(f"NLL: {nll_list.mean():.3f}({nll_list.std():.3f})")
        print(f"Brier Score: {brier_list.mean():.3f}({brier_list.std():.3f})")
        print(f"ECE: {ece_list.mean():.3f}({ece_list.std():.3f})")
