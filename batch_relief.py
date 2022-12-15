import os

dataset_list = [
    "spambase",
    "mushroom",
    "bank-marketing",
    "PhishingWebsites",
    "Bioresponse",
    "census",
    "trec",
    "semeval",
    "tennis",
]

text_datasets = [
    "trec",
    "semeval"
]

tag = "02"
device = "cuda:0"
repeats = 10

for dataset in dataset_list:
    lm = "metal"
    em = "mlp"
    sampler = "uncertain-rm"
    rm = "ensemble"  # use ensemble reviser when comparing AL methods
    if em == "mlp":
        n_epochs = 100
    else:
        n_epochs = 5
    if dataset in text_datasets:
        ext = " --extract_fn bert"
    else:
        ext = ""
    cmd = f"python main_rlf.py --device {device} --dataset {dataset} {ext} --label_model {lm} " \
          f"--end_model {em} --em_epochs {n_epochs} --sampler {sampler} --revision_model {rm} " \
          f"--use_valid_labels --use_soft_labels --repeats {repeats} --sample_budget 100 --sample_per_iter 10 --tag {tag}"
    print(cmd)
    os.system(cmd)