import argparse
from utils import compare_em_performance, compare_baseline_performance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="spambase")
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--exp_type", type=str, choices=["baseline", "sampler", "reviser", "cost"])
    parser.add_argument("--label_model", type=str, default="metal")
    parser.add_argument("--end_model", type=str, default="mlp")
    parser.add_argument("--tag", type=str, default="00")
    parser.add_argument("--plot_sample_frac", action="store_true")
    args = parser.parse_args()

    sampler_list = "passive uncertain uncertain-rm dal abstain disagreement".split(" ")
    reviser_list = "mlp mlp-temp expert-label ensemble cs-hinge cs-sigmoid".split(" ")
    cost_list = "adaptive 0.10 0.20 0.30 0.40".split(" ")
    if args.exp_type == "baseline":
        compare_baseline_performance(args.output_path, args.dataset, args.tag, args.plot_sample_frac)
    elif args.exp_type == "sampler":
        compare_em_performance(args.output_path, args.dataset, args.label_model, args.end_model, ["mlp"],
                               sampler_list, ["adaptive"], args.tag, args.plot_sample_frac)
    elif args.exp_type == "reviser":
        compare_em_performance(args.output_path, args.dataset, args.label_model, args.end_model, reviser_list,
                               ["passive"], ["adaptive"], args.tag, args.plot_sample_frac)
    else:
        compare_em_performance(args.output_path, args.dataset, args.label_model, args.end_model,
                               ["mlp"], ["passive"], cost_list, args.tag, args.plot_sample_frac)
