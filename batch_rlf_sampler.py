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

sampler_list = [
    "passive",
    "uncertain",
    "uncertain-rm",
    "dal",
    "abstain",
    "disagreement"
]

tag = "00"
device = "cuda:0"
repeats = 10

for dataset in dataset_list:
    for lm in label_model_list:
        for sampler in sampler_list:
            rm = "mlp"  # use mlp reviser when comparing AL methods
            if dataset in text_datasets:
                ext = " --extract_fn bert"
            else:
                ext = ""
            cmd = f"python main_rlf.py --device {device} --dataset {dataset} {ext} --label_model {lm} " \
                  f"--end_model mlp --em_epochs 100 --sampler {sampler} --revision_model {rm} " \
                  f"--use_valid_labels --repeats {repeats} --tag {tag}"
            print(cmd)
            os.system(cmd)