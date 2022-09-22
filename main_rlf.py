import argparse
from dataset.load_dataset import load_real_dataset, load_synthetic_dataset


def run_rlf(train_data, valid_data, test_data, args, seeds):
    """
    Run an active learning pipeline to revise label functions
    """




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # device info
    parser.add_argument("--device", type=str, default="cuda")
    # dataset
    parser.add_argument("--dataset", type=str, default="youtube")
    parser.add_argument("--dataset_path", type=str, default="../wrench-1.1/datasets/")
    parser.add_argument("--extract_fn", type=str, default="bert")  # method used to extract features
    # contrastive learning
    parser.add_argument("--contrastive_loss", type=str, default="scl")
    parser.add_argument("--data_augment", type=str, default="eda")
    # sampler
    parser.add_argument("--sampler", type=str, nargs="+", default="passive")
    parser.add_argument("--sample_budget", type=int, default=500)  # Total sample budget
    parser.add_argument("--sample_budget_init",type=int, default=50)  # sample budget for initialization
    parser.add_argument("--sample_budget_inc", type=int, default=50)  # increased sample budget per iteration
    # revision model
    parser.add_argument("--revision_method", type=str, default="relief")  # "nashaat": only revise labeled points
    parser.add_argument("--revision_model", type=str, default="linear-svm")
    # label model and end model
    parser.add_argument("--label_model", type=str, default="snorkel")
    parser.add_argument("--end_model", type=str, default=None)
    parser.add_argument("--use_soft_labels", action="store_true")
    # other settings
    parser.add_argument("--labeller", type=str, default="oracle")
    parser.add_argument("--metric", type=str, default="acc")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tag", type=str, default=None)
    # plot settings
    parser.add_argument("--plot_lf", action="store_true")  # plot LF accuracy and coverage over revision process
    parser.add_argument("--plot_sample", action="store_true")  # plot density plots for samples
    args = parser.parse_args()
    if args.dataset[:3] == "syn":
        train_data, valid_data, test_data = load_synthetic_dataset(args.dataset)
    else:
        train_data, valid_data, test_data = load_real_dataset(args.dataset_path, args.dataset, args.extract_fn)
