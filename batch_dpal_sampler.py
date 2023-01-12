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

bert_embedding_datasets = ["youtube", "sms", "imdb", "yelp", "trec", "agnews", "spouse", "cdr", "semeval", "chemprot"]
tag = "09"
label_model_list = ["metal"]
test_mode = False
for dataset in dataset_list:
    for lm in label_model_list:
        if dataset in bert_embedding_datasets:
            ext = " --extract_fn bert"
        else:
            ext = ""

        # evaluate the effect of sampler
        for sampler in ["passive", "uncertain", "uncertain-rm", "dal", "cluster-margin"]:
            cmd = f"python dpal_pipeline.py --dataset {dataset} {ext} --max_dim 300 --sampler {sampler} " \
                  f"--label_model {lm} --use_valid_labels --use_soft_labels --repeats 10 --tag {tag}"
            print(cmd)
            os.system(cmd)

    if test_mode:
        break
