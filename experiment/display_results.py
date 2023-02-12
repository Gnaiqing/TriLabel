import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


method_dict = {
    "trilabel": "TriLabel",
    "al": "Active Learning",
    "aw": "Active WeaSuL",
    "nashaat": "Revise LF",
    "pl": "Semi Supervised"
}


def get_results(results_list):
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
    print_res["n_labeled"] = res["n_labeled"][0,:]
    print_res["frac_labeled"] = res["frac_labeled"][0,:]
    print_res["golden_test_f1"] = res["golden_test_f1"].mean()

    for metric in ["label_acc", "label_coverage", "test_f1"]:
        print_res[metric] = res[metric].mean(axis=0)
        print_res[f"{metric}_stderr"] = res[metric].std(axis=0) / np.sqrt(len(res[metric]))
    return print_res


# def get_results(results_list):
#     n_run = len(results_list)
#     res = {}
#     for i in range(n_run):
#         if i == 0:
#             for key in results_list[i]:
#                 res[key] = []
#                 if isinstance(results_list[i][key], list):
#                     res[key].append(results_list[i][key][-1])
#                 else:
#                     res[key].append(results_list[i][key])
#         else:
#             for key in results_list[i]:
#                 if isinstance(results_list[i][key], list):
#                     res[key].append(results_list[i][key][-1])
#                 else:
#                     res[key].append(results_list[i][key])
#
#     for key in res:
#         res[key] = np.array(res[key])
#
#     print_res = {}
#     print_res["time"] = res["time"].mean()
#     print_res["n_labeled"] = res["n_labeled"].mean()
#     print_res["frac_labeled"] = res["frac_labeled"].mean()
#     print_res["golden_test_f1"] = res["golden_test_f1"].mean()
#
#     for metric in ["label_acc", "label_coverage", "test_f1"]:
#         print_res[metric] = res[metric].mean()
#         print_res[f"{metric}_stderr"] = res[metric].std() / np.sqrt(len(res[metric]))
#     return print_res


def plot_results(dataset, results_map, metric, output_path, tag):
    plt.rcParams.update({'font.size': 14})
    # plot label accuracy and coverage
    fig, ax = plt.subplots()
    for method in results_map:
        x = results_map[method]["n_labeled"]
        y1 = results_map[method][metric]
        y1_stderr = results_map[method][f"{metric}_stderr"]
        ax.plot(x, y1, label=method)
        ax.fill_between(x, y1 - 1.96 * y1_stderr, y1 + 1.96 * y1_stderr, alpha=.1)
        if method == "TriLabel" and metric == "test_f1":
            y_gold = results_map[method]["golden_test_f1"]
            ax.axhline(y=y_gold, ls="--",c="k", label="Golden Label")

    ax.set_xlabel("Label Budget")
    ax.set_ylabel(metric)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax.legend()
    ax.set_title(dataset)
    figpath = Path(output_path) / dataset / f"{dataset}_{metric}_{tag}.jpg"
    fig.savefig(figpath)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, nargs="+", default=["youtube", "sms", "imdb", "yelp", "PhishingWebsites",
                                                                   "bank-marketing", "census", "tennis"])
    parser.add_argument("--method", type=str, nargs="+", default=["al", "nashaat", "trilabel"])
    parser.add_argument("--exp_type", type=str, default="baseline", choices=["baseline", "sampler", "active_frac"])
    parser.add_argument("--label_model", type=str, default="metal")
    parser.add_argument("--end_model", type=str, default="mlp")
    parser.add_argument("--sampler", type=str, nargs="+", default=["passive",  "uncertain-rm", "tri-pl+random"])
    parser.add_argument("--tag", type=str, default="00")
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--plot_results", action="store_true")
    parser.add_argument("--print_results", action="store_true")
    args = parser.parse_args()
    pd.set_option('display.precision', 3)
    pd.set_option('display.max_columns', None)

    for dataset in args.dataset:
        results_map = {}
        if args.exp_type == "baseline":
            for method in args.method:
                if method == "trilabel":
                    id_tag = f"trilabel_{args.label_model}_{args.end_model}_uncertain-rm_{args.tag}"
                elif method == "nashaat":
                    id_tag = f"{method}_{args.label_model}_uncertain-lm_{args.tag}"
                else:
                    id_tag = f"{method}_{args.end_model}_{args.tag}"

                filepath = Path(args.output_path) / dataset / f"{id_tag}.json"
                if os.path.exists(filepath):
                    readfile = open(filepath, "r")
                    results = json.load(readfile)
                    results_list = results["data"]
                    res = get_results(results_list)
                    df_res = pd.DataFrame(res)
                    method_name = method_dict[method]
                    results_map[method_name] = res
                    if args.print_results:
                        print(f"Dataset: {dataset}, Method: {method_name}")
                        print(df_res)
                else:
                    print(f"Not find output file for {method} method on {dataset} dataset.")

            if args.plot_results:
                for metric in ["label_acc", "label_coverage", "test_f1"]:
                    plot_results(dataset, results_map, metric, args.output_path, args.tag)

        elif args.exp_type == "sampler":
            for sampler in args.sampler:
                id_tag = f"trilabel_{args.label_model}_{args.end_model}_{sampler}_{args.tag}"
                filepath = Path(args.output_path) / dataset / f"{id_tag}.json"
                if os.path.exists(filepath):
                    readfile = open(filepath, "r")
                    results = json.load(readfile)
                    results_list = results["data"]
                    res = get_results(results_list)
                    df_res = pd.DataFrame(res)
                    results_map[sampler] = res
                    if args.print_results:
                        print(f"Dataset: {dataset}, Sampler: {sampler}")
                        print(df_res)
                else:
                    print(f"Not find output file for {sampler} method on {dataset} dataset.")

            if args.plot_results:
                for metric in ["label_acc", "label_coverage", "test_f1"]:
                    plot_results(dataset, results_map, metric, args.output_path, args.tag+ "_sampler")

        elif args.exp_type == "active_frac":
            id_tag = f"trilabel_{args.label_model}_{args.end_model}_uncertain-rm_{args.tag}"
            filepath = Path(args.output_path) / dataset / f"{id_tag}.json"
            if os.path.exists(filepath):
                readfile = open(filepath, "r")
                results = json.load(readfile)
                results_list = results["data"]
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

                dp_active_frac = np.mean(res["dp_coverage"], axis=0)
                dp_active_frac_stderr = np.std(res["dp_coverage"], axis=0) / np.sqrt(len(res["dp_coverage"]))
                al_active_frac = np.mean(res["al_coverage"], axis=0)
                al_active_frac_stderr = np.std(res["al_coverage"], axis=0) / np.sqrt(len(res["al_coverage"]))
                n_labeled = res["n_labeled"][0,:]
                plt.rcParams.update({'font.size': 14})
                plt.figure()
                plt.plot(n_labeled, dp_active_frac, label="DP Active")
                plt.fill_between(n_labeled, dp_active_frac - 1.96 * dp_active_frac_stderr,
                                 dp_active_frac + 1.96 * dp_active_frac_stderr, alpha=.1)
                plt.plot(n_labeled, al_active_frac, label="AL Active")
                plt.fill_between(n_labeled, al_active_frac - 1.96 * al_active_frac_stderr,
                                 al_active_frac + 1.96 * al_active_frac_stderr, alpha=.1)
                plt.xlabel("Label Budget")
                plt.ylabel("Active Fraction")
                # plt.legend()
                plt.title(dataset)
                figpath = Path(args.output_path) / dataset / f"{dataset}_frac_{args.tag}.jpg"
                plt.savefig(figpath)






