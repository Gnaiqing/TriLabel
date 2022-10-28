import os

dataset_list = [
    # "youtube",
    # "trec",
    # "agnews",
    "imdb",
    "yelp",
]

revision_model_list = [
    # "logistic",
    # "linear-svm",
    # "decision-tree",
    # "random-forest",
    "voting"
]

for dataset in dataset_list:
    for revision_model in revision_model_list:
        cmd = f"python main_rlf.py --dataset {dataset} --revision_model_class {revision_model} --use_valid_data"
        print(cmd)
        os.system(cmd)