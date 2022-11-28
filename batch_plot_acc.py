import os

dataset_list = [
    # "youtube",
    # "trec",
    # "census",
    # "yelp",
    # "basketball",
    # "tennis",
    "spambase",
    "mushroom",
    "bank-marketing",
    "PhishingWebsites",
    "Bioresponse"
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
    "mv",
    "metal"
]

end_model_list = [
    "mlp"
]

tag = "00"

for dataset in dataset_list:
    for lm in label_model_list:
        for em in end_model_list:
            cmd = f"python plot_test_accuracy.py --dataset {dataset} --label_model {lm} " \
                  f"--end_model {em} --tag {tag} --plot_sample_frac"
            print(cmd)
            os.system(cmd)