import os

dataset_list = [
    ## text datasets
    "youtube",
    "imdb",
    "yelp",
    ## tabular datasets
    "PhishingWebsites",
    "bank-marketing",
    "census",
    ## image datasets
    "tennis",
    "basketball",
    ## multiclass datasets
    "trec",
    "agnews"
]

bert_embedding_datasets = ["youtube", "imdb", "yelp", "trec", "agnews"]
label_model_list = ["snorkel", "mv"]
tag = "06"
test_mode = False
for dataset in dataset_list:
    for lm in label_model_list:
        if dataset in bert_embedding_datasets:
            ext = " --extract_fn bert"
        else:
            ext = ""

        # evaluate the effect of ensemble size, bootstrap and select features
        for exp_type in ["ensemble", "bootstrap", "LF_size"]:
            cmd = f"python test_calibration.py --dataset {dataset} {ext} --label_model {lm} --exp_type {exp_type} --tag {tag}"
            print(cmd)
            os.system(cmd)

    if test_mode:
        break
