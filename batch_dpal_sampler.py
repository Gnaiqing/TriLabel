import os

dataset_list = [
    ## text datasets
    "youtube",
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
    "trec",
    "agnews"
]

bert_embedding_datasets = ["youtube", "imdb", "yelp", "trec", "agnews"]
label_model_list = ["metal"]
tag = "07"
test_mode = True
for dataset in dataset_list:
    for lm in label_model_list:
        if dataset in bert_embedding_datasets:
            ext = " --extract_fn bert"
        else:
            ext = ""

        # evaluate the effect of ensemble size, bootstrap and select features
        for sampler in ["passive", "uncertain", "uncertain-rm", "dal", "uncertain-joint"]:
            cmd = f"python dpal_pipeline.py --dataset {dataset} {ext} --sampler {sampler} " \
                  f"--label_model {lm} --use_valid_labels --use_soft_labels --repeats 20 --tag {tag}"
            print(cmd)
            os.system(cmd)

    if test_mode:
        break
