import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
import os
import matplotlib.pyplot as plt


method_dict = {
    "dpal" : "ReLieF",
    "al": "Active Learning",
    "aw": "Active WeaSuL",
    "nashaat": "Nashaat",
    "pl": "Semi Supervised"
}

calibration_dict = {
    "": "No calibration",
    "E": "EN",
    "EF": "EN+FS",
    "EFB": "EN+FS+BS"
}

def get_results(results_list, method):
    n_run = len(results_list)
    res = {}
    for i in range(n_run):
        if i == 0:
            for key in results_list[i]:
                res[key] = []
                res[key].append(results_list[i][key])
        else:
            for key in results_list[i]:
                res[key].append(results_list[i][key])

    for key in res:
        res[key] = np.array(res[key])

    print_res = {}
    print_res["frac_labeled"] = res["frac_labeled"][0,:]
    if method in ["al", "aw", "dpal"]:
        for metric in ["label_acc", "label_brier", "test_acc", "test_f1"]:
            print_res[metric] = res[f"{method}_{metric}"].mean(axis=0)
            print_res[f"{metric}_std"] = res[f"{method}_{metric}"].std(axis=0)
    else:
        for metric in ["label_acc", "label_brier", "test_acc", "test_f1"]:
            print_res[metric] = res[metric].mean(axis=0)
            print_res[f"{metric}_std"] = res[metric].std(axis=0)
    return print_res


def plot_results(dataset, results_map, metric, output_path, tag):
    fig, ax = plt.subplots()
    for method in results_map:
        x = results_map[method]["frac_labeled"]
        y = results_map[method][metric]
        y_stderr = results_map[method][f"{metric}_std"] / np.sqrt(10)
        ax.plot(x, y, label=method)
        ax.fill_between(x, y - 1.96 * y_stderr, y + 1.96 * y_stderr, alpha=.1)

    ax.legend()
    ax.set_xlabel("Label Budget")
    ax.set_ylabel(metric)
    ax.set_title(dataset)
    figpath = Path(output_path) / dataset / f"{metric}_{tag}.jpg"
    fig.savefig(figpath)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, nargs="+", default=["bank-marketing", "PhishingWebsites",
                        "census","youtube", "imdb", "yelp", "trec", "agnews"])
    parser.add_argument("--method", type=str, nargs="+", default=["pl", "al", "nashaat", "aw", "dpal"])
    parser.add_argument("--ablation", type=str, default=None, choices=["sampler", "aggregation", "calibration"])
    parser.add_argument("--label_model", type=str, default="metal")
    parser.add_argument("--end_model", type=str, default="mlp")
    parser.add_argument("--aggregation", type=str, default="bayesian")
    parser.add_argument("--sampler", type=str, default="uncertain-joint")
    parser.add_argument("--tag", type=str, default="07")
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--plot_results", action="store_true")
    parser.add_argument("--print_results", action="store_true")
    args = parser.parse_args()
    pd.set_option('display.precision', 3)

    for dataset in args.dataset:
        results_map = {}
        if args.ablation is None:
            for method in args.method:
                if method == "dpal":
                    id_tag = f"{method}_{args.label_model}_{args.end_model}_{args.aggregation}_{args.sampler}_E_{args.tag}"
                elif method == "nashaat":
                    id_tag = f"{method}_{args.label_model}_uncertain_{args.tag}"
                else:
                    id_tag = f"{method}_{args.end_model}_{args.tag}"

                filepath = Path(args.output_path) / dataset / f"{id_tag}.json"
                if os.path.exists(filepath):
                    readfile = open(filepath, "r")
                    results = json.load(readfile)
                    results_list = results["data"]
                    res = get_results(results_list, method)
                    df_res = pd.DataFrame(res)
                    method_name = method_dict[method]
                    results_map[method_name] = res
                    if args.print_results:
                        print(f"Dataset: {dataset}, Method: {method_name}")
                        print(df_res)
                else:
                    print(f"Not find output file for {method} method on {dataset} dataset.")

            if args.plot_results:
                for metric in ["label_brier", "label_acc", "test_acc", "test_f1"]:
                    plot_results(dataset, results_map, metric, args.output_path, args.tag)

        elif args.ablation == "aggregation":  # Ablation study for aggregation method
            method = "dpal"
            for aggregation in ["bayesian", "average", "confidence", "weighted"]:
                id_tag = f"{method}_{args.label_model}_{args.end_model}_{aggregation}_{args.sampler}_EF_{args.tag}"
                filepath = Path(args.output_path) / dataset / f"{id_tag}.json"
                if os.path.exists(filepath):
                    readfile = open(filepath, "r")
                    results = json.load(readfile)
                    results_list = results["data"]
                    res = get_results(results_list, method)
                    df_res = pd.DataFrame(res)
                    results_map[aggregation] = res
                    if args.print_results:
                        print(f"Dataset: {dataset}, Aggregation: {aggregation}")
                        print(df_res)
                else:
                    print(f"Not find output file for {aggregation} method on {dataset} dataset.")

            if args.plot_results:
                for metric in ["label_brier", "label_acc", "test_acc", "test_f1"]:
                    plot_results(dataset, results_map, metric, args.output_path, args.tag + "_agg")

        elif args.ablation == "sampler":
            method = "dpal"
            for sampler in ["passive", "uncertain", "uncertain-rm" ,"dal", "uncertain-joint"]:
                id_tag = f"{method}_{args.label_model}_{args.end_model}_{args.aggregation}_{sampler}_EF_{args.tag}"
                filepath = Path(args.output_path) / dataset / f"{id_tag}.json"
                if os.path.exists(filepath):
                    readfile = open(filepath, "r")
                    results = json.load(readfile)
                    results_list = results["data"]
                    res = get_results(results_list, method)
                    df_res = pd.DataFrame(res)
                    results_map[sampler] = res
                    if args.print_results:
                        print(f"Dataset: {dataset}, Sampler: {sampler}")
                        print(df_res)
                else:
                    print(f"Not find output file for {sampler} method on {dataset} dataset.")

            if args.plot_results:
                for metric in ["label_brier", "label_acc", "test_acc", "test_f1"]:
                    plot_results(dataset, results_map, metric, args.output_path, args.tag + "_sampler")

        elif args.ablation == "calibration":
            method = "dpal"
            for calibration_tag in ["", "E", "EF", "EFB"]:
                id_tag = f"{method}_{args.label_model}_{args.end_model}_{args.aggregation}_{args.sampler}_{calibration_tag}_{args.tag}"
                filepath = Path(args.output_path) / dataset / f"{id_tag}.json"
                if os.path.exists(filepath):
                    readfile = open(filepath, "r")
                    results = json.load(readfile)
                    results_list = results["data"]
                    res = get_results(results_list, method)
                    df_res = pd.DataFrame(res)
                    calib_name = calibration_dict[calibration_tag]
                    results_map[calib_name] = res

                    if args.print_results:
                        print(f"Dataset: {dataset}, Calibration: {calibration_tag}")
                        print(df_res)
                else:
                    print(f"Not find output file for Calibration {calibration_tag} on {dataset} dataset.")

            if args.plot_results:
                for metric in ["label_brier", "label_acc", "test_acc", "test_f1"]:
                    plot_results(dataset, results_map, metric, args.output_path, args.tag + "_calib")






