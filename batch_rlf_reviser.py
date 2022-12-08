import os

dataset_list = [
    # "youtube",
    # "trec",
    # "yelp",
    # "basketball",
    # "tennis",
    # "spambase",
    "mushroom",
    "bank-marketing",
    "PhishingWebsites",
    "Bioresponse",
    "census",
]

text_datasets = [
    "youtube",
    "trec",
    "yelp"
]

tabular_datasets = [
    "census",
    "spambase",
    "mushroom",
    "bank-marketing",
    "PhishingWebsites",
    "Bioresponse"
]

image_datasets = [
    "basketball",
    "tennis"
]

label_model_list = [
    # "mv",
    "metal"
]

end_model_list = [
    "mlp"
]

sampler_list = [
    "passive",
    "uncertain",
    "uncertain-rm",
    "dal",
    "abstain",
    "disagreement"
]

revision_model_list = [
    "expert-label",
    "mlp",
    "mlp-temp",
    "mc-dropout",
    "cs-hinge",
    "cs-sigmoid",
    "ensemble"
]

no_proba_revision_models = [
    "expert-label",
    "cs-hinge",
    "cs-sigmoid"
]

require_proba_samplers = [
    "uncertain-rm",
    "disagreement"
]

tag = "01"
device = "cuda:1"
repeats = 10
debug_mode = False

for dataset in dataset_list:
    for lm in label_model_list:
        for em in end_model_list:
            for rm in revision_model_list:
                sampler = "passive" # use passive sampler when comparing revisers
                if em == "mlp":
                    n_epochs = 100
                else:
                    n_epochs = 5
                if dataset in text_datasets:
                    ext = " --extract_fn bert"
                else:
                    ext = ""
                cmd = f"python main_rlf.py --device {device} --dataset {dataset} {ext} --label_model {lm} " \
                      f"--end_model {em} --em_epochs {n_epochs} --sampler {sampler} --revision_model {rm} " \
                      f"--use_valid_labels --repeats {repeats} --tag {tag}"
                print(cmd)
                os.system(cmd)