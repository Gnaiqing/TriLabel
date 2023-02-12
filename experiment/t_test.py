# Apply t-test to test whether the samplers outperform each other
import argparse
from pathlib import Path

import pandas as pd

from experiment.display_results import get_results
import os
import json
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, nargs="+", default=["youtube", "sms", "imdb", "yelp", "PhishingWebsites",
                                                                   "bank-marketing", "census", "tennis", "basketball"])
    parser.add_argument("--sampler", type=str, nargs="+", default=["passive", "uncertain-rm","tri-pl+random"])
    parser.add_argument("--tag", type=str, default="02")
    parser.add_argument("--output_path", type=str, default="output/")
    args = parser.parse_args()

    for dataset in args.dataset:
        print(f"Evaluating dataset {dataset}")
        f1_mean = []
        f1_stderr = []
        for sampler in args.sampler:
            id_tag = f"trilabel_metal_mlp_{sampler}_{args.tag}"
            filepath = Path(args.output_path) / dataset / f"{id_tag}.json"
            if os.path.exists(filepath):
                readfile = open(filepath, "r")
                results = json.load(readfile)
                results_list = results["data"]
                res = get_results(results_list)
                f1_mean.append(res["test_f1"][-1])
                f1_stderr.append(res["test_f1_stderr"][-1])
            else:
                print(f"Sampler {sampler} not found for dataset {dataset}")

        df = pd.DataFrame({"f1_mean": f1_mean, "f1_stderr": f1_stderr}, index=args.sampler)
        best_idx = np.argmax(f1_mean)
        candidate_sampler = []
        n_sampler = len(args.sampler)
        T = np.zeros(n_sampler, dtype=float)

        thres = 1.860  # for one-sided independent two-sample test with confidence 95%
        for i in range(n_sampler):
            T[i] = (f1_mean[best_idx] - f1_mean[i]) / np.sqrt(f1_stderr[best_idx]**2 + f1_stderr[i]**2)
            if T[i] < thres:
                candidate_sampler.append(args.sampler[i])

        print(df)
        print("Best sampler: ", args.sampler[best_idx])
        print("Comparable samplers: ", candidate_sampler)

