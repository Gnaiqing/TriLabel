import os

dataset_list = [
    "spambase",
    # "mushroom",
    # "bank-marketing",
    # "PhishingWebsites",
    # "Bioresponse",
    # "census",
    # "trec",
    # "semeval",
    # "tennis",
]

text_datasets = [
    "youtube",
    "trec",
    "yelp"
    "semeval"
]

label_model_list = [
    "mv",
    "metal"
]


revision_model_list = [
    "expert-label",
    "mlp",
    "mlp-temp",
    "mc-dropout",
    "cs-hinge",
    "cs-sigmoid",
    "ensemble"
]

tag = "00"
device = "cuda:0"
repeats = 10
debug_mode = False

for dataset in dataset_list:
    for lm in label_model_list:
        for rm in revision_model_list:
            sampler = "passive" # use passive sampler when comparing revisers
            if dataset in text_datasets:
                ext = " --extract_fn bert"
            else:
                ext = ""
            cmd = f"python main_rlf.py --device {device} --dataset {dataset} {ext} --label_model {lm} " \
                  f"--end_model mlp --em_epochs 100 --sampler {sampler} --revision_model {rm} " \
                  f"--use_valid_labels --repeats {repeats} --tag {tag}"
            print(cmd)
            os.system(cmd)