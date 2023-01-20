import os

dataset_list = [
    ## text datasets
    "youtube",
    "sms",
    "imdb",
    "yelp",
    ## tabular datasets
    "PhishingWebsites",
    "bank-marketing",
    "census",
    ## image datasets
    # "tennis",
    # "basketball",
    ## multiclass datasets
    "trec",
    "agnews",
    ## relation datasets
    # "spouse",
    # "cdr",
    # "semeval",
    # "chemprot"
]

bert_embedding_datasets = ["youtube", "sms", "imdb", "yelp", "trec", "agnews", "spouse", "cdr", "semeval", "chemprot"]
label_model_list = ["metal"]
tag = "07"
test_mode = False
for dataset in dataset_list:
    for lm in label_model_list:
        if dataset in bert_embedding_datasets:
            ext = " --extract_fn bert"
        else:
            ext = ""

        if tag == "08":
            ext += " --max_dim 300"
        # No calibration
        cmd = f"python dpal_pipeline.py --dataset {dataset} {ext} --lm_ensemble_size 1 --LF_selected_size all " \
              f"--label_model {lm} --use_valid_labels --use_soft_labels --tag {tag}"
        print(cmd)
        os.system(cmd)

        # Use ensemble calibration
        cmd = f"python dpal_pipeline.py --dataset {dataset} {ext} --lm_ensemble_size 10 --LF_selected_size all " \
              f"--label_model {lm} --use_valid_labels --use_soft_labels --tag {tag}"
        print(cmd)
        os.system(cmd)

        # Ensemble + Feature selection
        cmd = f"python dpal_pipeline.py --dataset {dataset} {ext} --lm_ensemble_size 10 --LF_selected_size auto " \
              f"--label_model {lm} --use_valid_labels --use_soft_labels --tag {tag}"
        print(cmd)
        os.system(cmd)

        # Ensemble + Feature selection + bootstrap
        cmd = f"python dpal_pipeline.py --dataset {dataset} {ext} --lm_ensemble_size 10 --LF_selected_size auto " \
              f"--bootstrap --label_model {lm} --use_valid_labels --use_soft_labels --tag {tag}"
        print(cmd)
        os.system(cmd)

    if test_mode:
        break
