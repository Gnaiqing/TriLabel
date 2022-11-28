import os

dataset_list = [
    # "youtube",
    # "trec",
    # "census",
    # "yelp",
    # "basketball",
    # "tennis",
    "spambase",
    "mushroom",
    "bank-marketing",
    "PhishingWebsites",
    "Bioresponse"
]

text_datasets = [
    "youtube",
    "trec",
    "yelp"
]

tabular_datasets = [
    "census",
    "spambase",
    "mushroom",
    "bank-marketing",
    "PhishingWebsites",
    "Bioresponse"
]

image_datasets = [
    "basketball",
    "tennis"
]

label_model_list = [
    "mv",
    "metal"
]

end_model_list = [
    "mlp"
]

sampler_list = [
    "passive"
]

revision_model_list = [
    "expert-label",
    "mlp",
    "mlp-temp",
    "dropout"
]

tag = "00"
device = "cuda:1"
repeats = 10
debug_mode = False

for dataset in dataset_list:
    for lm in label_model_list:
        for em in end_model_list:
            for sampler in sampler_list:
                for rm in revision_model_list:
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
                          f"--use_valid_labels --repeats {repeats} --tag {tag}"
                    print(cmd)
                    os.system(cmd)
                    if debug_mode:
                        exit(0)