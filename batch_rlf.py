import os

dataset_list = [
    "youtube",
    "trec",
    "imdb",
    "yelp",
    "census",
    "semeval"
]

label_model_list = [
    "mv",
    "metal"
]

revision_model_list = [
    "mlp",
    "expert-label"
]

for dataset in dataset_list:
    for lm in label_model_list:
        for rm in revision_model_list:
            cmd = f"python main_rlf.py --dataset {dataset} --label_model {lm} --end_model mlp --em_epochs 100" \
                  f" --revision_model_class {rm} --use_valid_labels --repeats 20 --tag 2"
            print(cmd)
            os.system(cmd)