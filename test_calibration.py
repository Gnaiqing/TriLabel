from pathlib import Path
import argparse
from dataset.load_dataset import load_real_dataset, load_synthetic_dataset
import numpy as np
from utils import save_results, get_label_model, ABSTAIN
from uncertain_estimation import evaluate_uncertainty_estimation
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_calib_results(errors, nlls, briers, x, x_label, dataset, output_path, tag):
    fig, ax = plt.subplots()
    if x_label == "ensemble":
        ax.plot(x, errors, color="r", label="error")
        ax.plot(x, nlls, color="g", label="nll")
        ax.plot(x, briers, color="b", label="brier")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel(x_label)
        ax.set_ylabel("Error/NLL/Brier")
        ax.set_title(dataset)
        ax.legend()
        filepath = Path(output_path) / dataset / f"Ensemble_{tag}.jpg"
        dirname = Path(output_path) / dataset
        Path(dirname).mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath)
    elif x_label == "LF_size":
        ax.plot(x, errors, color="r", label="error")
        ax.plot(x, nlls, color="g", label="nll")
        ax.plot(x, briers, color="b", label="brier")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel(x_label)
        ax.set_ylabel("Error/NLL/Brier")
        ax.set_title(dataset)
        ax.legend()
        filepath = Path(output_path) / dataset / f"LF_{tag}.jpg"
        dirname = Path(output_path) / dataset
        Path(dirname).mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath)
    else:
        x = np.arange(2)
        width=0.2
        ax.bar(x-width, errors, width, label="error")
        ax.bar(x, nlls, width, label="nll")
        ax.bar(x+width, briers, width, label="brier")
        ax.set_xticks(x, ["No Bootstrap", "Bootstrap"])
        ax.legend()
        ax.set_title(dataset)
        ax.set_ylabel("Error/NLL/Brier")
        filepath = Path(output_path) / dataset / f"Bootstrap_{tag}.jpg"
        dirname = Path(output_path) / dataset
        Path(dirname).mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="youtube")
    parser.add_argument("--dataset_path", type=str, default="../datasets/")
    parser.add_argument("--extract_fn", type=str, default=None)  # method used to extract features
    parser.add_argument("--label_model", type=str, default="snorkel")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_type", type=str, choices=["ensemble", "bootstrap", "LF_size"])
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--tag", type=str, default="test")
    parser.add_argument("--save_results", action="store_true")
    args = parser.parse_args()
    if args.dataset[:3] == "syn":
        train_data, valid_data, test_data = load_synthetic_dataset(args.dataset)
    else:
        train_data, valid_data, test_data = load_real_dataset(args.dataset_path, args.dataset, args.extract_fn)

    np.random.seed(args.seed)
    run_seeds = np.random.randint(1, 100000, args.repeats)
    weak_labels = np.array(train_data.weak_labels)
    N, M = weak_labels.shape
    errs = []
    nlls = []
    briers = []
    if args.exp_type == "ensemble":
        ensemble_size_list = list(range(1, 20, 2))
        for K in ensemble_size_list:
            res = evaluate_uncertainty_estimation(train_data, valid_data,
                                                  label_model_type=args.label_model,
                                                  feature_size=M,
                                                  ensemble_size=K,
                                                  bootstrap=False,
                                                  repeats=args.repeats,
                                                  seeds=run_seeds)
            id_tag = f"calib_lm={args.label_model}_en={K}_bs=False_ft={M}_{args.tag}"
            if args.save_results:
                save_results(res, args.output_path, args.dataset, f"{id_tag}.json")

            errs.append(1 - np.mean(res["acc"]).item())
            nlls.append(np.mean(res["nll"]).item())
            briers.append(np.mean(res["brier"]).item())
        plot_calib_results(errs, nlls, briers, ensemble_size_list, "ensemble", args.dataset, args.output_path, id_tag)
    elif args.exp_type == 'bootstrap':
        K = 10  # fix ensemble size to 10
        for bs in [False, True]:
            res = evaluate_uncertainty_estimation(train_data, valid_data,
                                                  label_model_type=args.label_model,
                                                  feature_size=M,
                                                  ensemble_size=K,
                                                  bootstrap=bs,
                                                  repeats=args.repeats,
                                                  seeds=run_seeds)
            id_tag = f"calib_lm={args.label_model}_en={K}_bs=False_ft={M}_{args.tag}"
            if args.save_results:
                save_results(res, args.output_path, args.dataset, f"{id_tag}.json")

            errs.append(1 - np.mean(res["acc"]).item())
            nlls.append(np.mean(res["nll"]).item())
            briers.append(np.mean(res["brier"]).item())

        plot_calib_results(errs, nlls, briers, [False, True], "bootstrap", args.dataset, args.output_path, id_tag)
    else:
        K = 10  # fix ensemble size to 10
        if M <= 20:
            fs_list = list(range(3, M+1, 2))
        elif M <= 50:
            fs_list = list(range(3, M+1, 5))
        else:
            fs_list = list(range(3, M+1, 10))

        for fs in fs_list:
            res = evaluate_uncertainty_estimation(train_data, valid_data,
                                                  label_model_type=args.label_model,
                                                  feature_size=fs,
                                                  ensemble_size=K,
                                                  bootstrap=False,
                                                  repeats=args.repeats,
                                                  seeds=run_seeds)
            id_tag = f"calib_lm={args.label_model}_en={K}_bs=False_ft={fs}_{args.tag}"
            if args.save_results:
                save_results(res, args.output_path, args.dataset, f"{id_tag}.json")

            errs.append(1 - np.mean(res["acc"]).item())
            nlls.append(np.mean(res["nll"]).item())
            briers.append(np.mean(res["brier"]).item())

        plot_calib_results(errs, nlls, briers, fs_list, "LF_size", args.dataset, args.output_path, id_tag)





