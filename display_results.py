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
    print_res["time"] = np.repeat(res["time"].mean(axis=0), res["frac_labeled"].shape[1])
    print_res["frac_labeled"] = res["frac_labeled"][0,:]
    if method in ["al", "aw", "dpal"]:
        for metric in ["label_brier", "test_f1"]:
            print_res[metric] = res[f"{method}_{metric}"].mean(axis=0)
            print_res[f"{metric}_stderr"] = res[f"{method}_{metric}"].std(axis=0) / np.sqrt(len(res[f"{method}_{metric}"]))
    else:
        for metric in ["label_brier", "test_f1"]:
            print_res[metric] = res[metric].mean(axis=0)
            print_res[f"{metric}_stderr"] = res[metric].std(axis=0) / np.sqrt(len(res[metric]))
    return print_res


def plot_results(dataset, results_map, output_path, tag):
    plt.rcParams.update({'font.size': 14})
    fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)
    for method in results_map:
        x = results_map[method]["frac_labeled"]
        y1 = results_map[method]["label_brier"]
        y1_stderr = results_map[method][f"label_brier_stderr"]
        ax1.plot(x, y1, label=method)
        ax1.fill_between(x, y1 - 1.96 * y1_stderr, y1 + 1.96 * y1_stderr, alpha=.1)
        y2 = results_map[method]["test_f1"]
        y2_stderr = results_map[method]["test_f1_stderr"]
        ax2.plot(x,y2, label=method)
        ax2.fill_between(x, y2-1.96*y2_stderr, y2 + 1.96 * y2_stderr, alpha=0.1)

    # ax1.legend()
    ax2.set_xlabel("Label Budget")
    ax1.set_ylabel("brier score")
    ax2.set_ylabel("Test F1")
    ax1.set_title(dataset)
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    figpath = Path(output_path) / dataset / f"{dataset}_perf_{tag}.jpg"
    fig.savefig(figpath)
    plt.tight_layout()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, nargs="+", default=["bank-marketing", "PhishingWebsites",
                        "census","youtube", "imdb", "yelp", "sms", "trec"])
    parser.add_argument("--method", type=str, nargs="+", default=["pl", "al", "nashaat", "dpal", "aw"])
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
                    id_tag = f"{method}_{args.label_model}_{args.end_model}_{args.aggregation}_{args.sampler}_EF_{args.tag}"
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
                plot_results(dataset, results_map, args.output_path, args.tag)

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
                plot_results(dataset, results_map, args.output_path, args.tag + "_agg")

        elif args.ablation == "sampler":
            method = "dpal"
            for sampler in ["passive", "uncertain-joint", "cluster-margin", "coreset", "badge"]:
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
                plot_results(dataset, results_map, args.output_path, args.tag + "_sampler")

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
                plot_results(dataset, results_map, args.output_path, args.tag + "_calib")






