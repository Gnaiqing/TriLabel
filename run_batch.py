import os

dataset_list = [
    ## text datasets
    "youtube",
    "sms",
    "imdb",
    "yelp",
    ## tabular datasets
    # "PhishingWebsites",
    # "bank-marketing",
    # "census",
    ## image datasets
    # "tennis",
    # "basketball",
    ## multiclass datasets
    # "trec",
    # "agnews",
    ## relation datasets
    # "spouse",
    # "cdr",
    # "semeval",
    # "chemprot"
]

method_list = [
    "al",
    "aw",
    "nashaat",
    "pl",
    "dpal-ensemble"
]

bert_embedding_datasets = ["youtube", "sms", "imdb", "yelp", "trec", "agnews", "spouse", "cdr", "semeval", "chemprot"]
tag = "07"
test_mode = True
use_soft_labels = True
verbose = False
for dataset in dataset_list:
    ext = ""
    if dataset in bert_embedding_datasets:
        ext += " --extract_fn bert"
    if use_soft_labels:
        ext += " --use_soft_labels"
    if verbose:
        ext += " --verbose"

    for method in method_list:
        if method == "al":
            cmd = f"python baselines/uncertain_sampling.py --dataset {dataset} {ext} --tag {tag}"
        elif method == "aw":
            cmd = f"python baselines/run_active_weasul.py --dataset {dataset} {ext} --tag {tag}"
        elif method == "nashaat":
            cmd = f"python baselines/run_nashaat.py --dataset {dataset} {ext} --tag {tag}"
        elif method == "pl":
            cmd = f"python baselines/pseudo_labelling.py --dataset {dataset} {ext} --tag {tag}"
        elif method == "dpal-ensemble":
            cmd = f"python baselines/dpal_ensemble.py --dataset {dataset} {ext} --tag {tag}"

        print(cmd)
        os.system(cmd)

    if test_mode:
        break