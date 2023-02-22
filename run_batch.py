import os
import argparse

bert_embedding_datasets = ["youtube", "sms", "imdb", "yelp"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, nargs="+", default=["youtube", "sms", "imdb", "yelp", "PhishingWebsites",
                                                                   "bank-marketing", "census", "tennis"])
    parser.add_argument("--method", type=str, nargs="+", default=["al", "aw", "nashaat", "trilabel"])
    parser.add_argument("--sampler", type=str, nargs="+", default=["uncertain-rm"])
    parser.add_argument("--use_soft_labels", type=bool, default=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--tag", type=str, default="test")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--sample_budget", type=float, default=300)
    parser.add_argument("--sample_per_iter", type=float, default=100)
    parser.add_argument("--desired_label_acc", type=float, default=None)
    parser.add_argument("--desired_label_cov", type=float, default=None)
    parser.add_argument("--plot_performance", action="store_true")
    parser.add_argument("--record_runtime", action="store_true")
    parser.add_argument("--theta_explore_strategy", type=str, default="step")
    parser.add_argument("--theta_explore_num", type=int, default=10)
    parser.add_argument("--optimize_target", type=str, choices=["accuracy", "coverage", "f1"], default="f1")
    parser.add_argument("--calibration", type=str, choices=["EN", "EN+FS", "EN+FS+BS"], default=None)

    args = parser.parse_args()
    tag = args.tag
    for dataset in args.dataset:
        ext = f" --sample_budget {args.sample_budget} --sample_per_iter {args.sample_per_iter} --repeats {args.repeats}"
        if dataset in bert_embedding_datasets:
            ext += " --extract_fn bert"
        if args.use_soft_labels:
            ext += " --use_soft_labels"
        if args.verbose:
            ext += " --verbose"
        if args.record_runtime:
            ext += " --record_runtime"

        for method in args.method:
            if method == "al":
                cmd = f"python baselines/uncertain_sampling.py --dataset {dataset} {ext} --tag {tag}"
                print(cmd)
                os.system(cmd)
            elif method == "aw":
                cmd = f"python baselines/run_active_weasul.py --dataset {dataset} {ext} --tag {tag}"
                print(cmd)
                os.system(cmd)
            elif method == "nashaat":
                cmd = f"python baselines/run_nashaat.py --dataset {dataset} {ext} --tag {tag}"
                print(cmd)
                os.system(cmd)
            elif method == "pl":
                cmd = f"python baselines/pseudo_labelling.py --dataset {dataset} {ext} --tag {tag}"
                print(cmd)
                os.system(cmd)
            elif method == "trilabel":
                if args.record_runtime:
                    trilabel_ext = " --record_runtime"
                else:
                    trilabel_ext = " --evaluate"

                if args.desired_label_acc is not None:
                    trilabel_ext += f" --desired_label_acc {args.desired_label_acc}"

                if args.desired_label_cov is not None:
                    trilabel_ext += f" --desired_label_cov {args.desired_label_cov}"

                if args.plot_performance:
                    trilabel_ext += f" --plot_performance"

                if args.optimize_target == "f1":
                    trilabel_ext += f" --theta_explore_strategy {args.theta_explore_strategy}"
                    trilabel_ext += f" --theta_explore_num {args.theta_explore_num}"
                else:
                    trilabel_ext += f" --optimize_target {args.optimize_target}"

                if args.calibration is not None:
                    trilabel_ext += f" --calibration {args.calibration}"

                for sampler in args.sampler:
                    cmd = f"python trilabel.py --dataset {dataset} {ext} {trilabel_ext} " \
                          f"--sampler {sampler} --tag {tag}"
                    print(cmd)
                    os.system(cmd)

