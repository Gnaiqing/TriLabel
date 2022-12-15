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
    "yelp",
    "semeval"
]

tag = "00"
device = "cuda:0"
repeats = 10

for dataset in dataset_list:
    if dataset in text_datasets:
        ext = " --extract_fn bert"
    else:
        ext = ""

    cmd = f"python main_rlf.py --device {device} --dataset {dataset} {ext} --label_model metal " \
          f"--end_model mlp --em_epochs 100 --sampler passive --revision_model mlp " \
          f"--use_valid_labels --repeats {repeats} --tag {tag}"
    print(cmd)
    os.system(cmd)

    cmd = f"python main_rlf.py --device {device} --dataset {dataset} {ext} --label_model metal " \
          f"--end_model mlp --em_epochs 100 --sampler passive --revision_model mlp " \
          f"--use_valid_labels --repeats {repeats} --tag {tag}-LF --revision_type LF"
    print(cmd)
    os.system(cmd)

    cmd = f"python main_rlf.py --device {device} --dataset {dataset} {ext} --label_model metal " \
          f"--end_model mlp --em_epochs 100 --sampler passive --revision_model mlp " \
          f"--use_valid_labels --repeats {repeats} --tag {tag}-label --revision_type label"
    print(cmd)
    os.system(cmd)