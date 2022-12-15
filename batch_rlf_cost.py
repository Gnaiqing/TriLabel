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

label_model_list = [
    "mv",
    "metal"
]

end_model_list = [
    "mlp"
]

tag = "00"
device = "cuda:0"
repeats = 10

for dataset in dataset_list:
    for lm in label_model_list:
        rm = "mlp"
        sampler = "passive"  # use passive sampler when comparing revisers
        for cost in [0.1,0.2,0.3,0.4, None]:
            if dataset in text_datasets:
                ext = " --extract_fn bert"
            else:
                ext = ""
            if cost is not None:
                cmd = f"python main_rlf.py --device {device} --dataset {dataset} {ext} --label_model {lm} " \
                      f"--end_model mlp --em_epochs 100 --sampler {sampler} --revision_model {rm} " \
                      f"--rejection_cost {cost} --use_valid_labels --repeats {repeats} --tag {tag}"
            else:
                cmd = f"python main_rlf.py --device {device} --dataset {dataset} {ext} --label_model {lm} " \
                      f"--end_model mlp --em_epochs 100 --sampler {sampler} --revision_model {rm} " \
                      f"--use_valid_labels --repeats {repeats} --tag {tag}"
            print(cmd)
            os.system(cmd)