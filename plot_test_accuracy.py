import argparse
from utils import compare_em_performance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="youtube")
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--revision_model_list", type=str,
                        default="mlp mlp-temp expert-label dropout")
    parser.add_argument("--label_model", type=str, default="metal")
    parser.add_argument("--end_model", type=str, default="mlp")
    parser.add_argument("--tag", type=str, default="00")
    parser.add_argument("--plot_sample_frac", action="store_true")
    args = parser.parse_args()
    args.revision_model_list = args.revision_model_list.split(" ")
    compare_em_performance(args.output_path, args.dataset, args.label_model, args.end_model, args.revision_model_list,
                           args.tag, args.plot_sample_frac)
