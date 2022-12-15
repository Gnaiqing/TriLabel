import os

dataset_list = [
    "spambase",
    "mushroom",
    "bank-marketing",
    "PhishingWebsites",
    "Bioresponse",
    "census",
    # "trec",
    # "semeval",
    # "tennis",
]

text_datasets = [
    "trec",
    "semeval"
]

tag = "02"
device = "cuda:0"
repeats = 10
penalty = 1000.0

for dataset in dataset_list:
    em = "mlp"
    sampler = "maxkl"
    rm = "ensemble"  # use ensemble reviser when comparing AL methods
    if em == "mlp":
        n_epochs = 100
    else:
        n_epochs = 5
    if dataset in text_datasets:
        ext = " --extract_fn bert"
    else:
        ext = ""
    cmd = f"python aw_pipeline.py --device {device} --dataset {dataset} {ext} " \
          f"--end_model {em} --em_epochs {n_epochs} --sampler {sampler} --penalty_strength {penalty} " \
          f"--use_valid_labels --use_soft_labels --repeats {repeats} --tag {tag} --sample_budget 100 --sample_per_iter 10"
    print(cmd)
    os.system(cmd)