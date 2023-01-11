import os

dataset_list = [
    ## text datasets
    "youtube",
    "imdb",
    "yelp",
    # ## tabular datasets
    "PhishingWebsites",
    "bank-marketing",
    "census",
    ## image datasets
    # "tennis",
    # "basketball",
    ## multiclass datasets
    "trec",
    "agnews"
]

bert_embedding_datasets = ["youtube", "imdb", "yelp", "trec", "agnews"]
tag = "07"
test_mode = False
for dataset in dataset_list:
    if dataset in bert_embedding_datasets:
        ext = " --extract_fn bert"
    else:
        ext = ""

    cmd = f"python nashaat_pipeline.py --dataset {dataset} {ext} --label_model metal " \
          f"--use_valid_labels --use_soft_labels --tag {tag}"
    print(cmd)
    os.system(cmd)

    if test_mode:
        break
