import os

dataset_list = [
    ## text datasets
    # "youtube",
    "sms",
    # "imdb",
    # "yelp",
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

bert_embedding_datasets = ["youtube", "sms", "imdb", "yelp", "trec", "agnews", "spouse", "cdr", "semeval", "chemprot"]
tag = "07"
test_mode = False
for dataset in dataset_list:
    if dataset in bert_embedding_datasets:
        ext = " --extract_fn bert"
    else:
        ext = ""
    if tag == "08":
        ext += " --max_dim 300"

    cmd = f"python nashaat_pipeline.py --dataset {dataset} {ext} --label_model metal " \
          f"--use_valid_labels --use_soft_labels --tag {tag}"
    print(cmd)
    os.system(cmd)

    if test_mode:
        break
