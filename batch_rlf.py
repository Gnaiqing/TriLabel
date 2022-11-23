import os

dataset_list = [
    # "youtube",
    # "trec",
    # "census",
    # "yelp",
    "basketball",
    "tennis",
    "commercial"
]

label_model_list = [
    "mv",
    "metal"
]

end_model_list = [
    "mlp"
]

sampler_list = [
    "passive"
]

revision_model_list = [
    "expert-label",
    "mlp",
    "mlp-temp",
    "dropout"
]

tag = "00"
repeats = 10
debug_mode = False

for dataset in dataset_list:
    for lm in label_model_list:
        for em in end_model_list:
            for sampler in sampler_list:
                for rm in revision_model_list:
                    if em == "mlp":
                        n_epochs = 100
                    else:
                        n_epochs = 5
                    cmd = f"python main_rlf.py --dataset {dataset} --label_model {lm} --end_model {em} --em_epochs {n_epochs}" \
                          f" --sampler {sampler} --revision_model {rm} --use_valid_labels --repeats {repeats} --tag {tag}"
                    print(cmd)
                    os.system(cmd)
                    if debug_mode:
                        exit(0)