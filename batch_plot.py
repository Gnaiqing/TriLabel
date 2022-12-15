import os

dataset_list = [
    "spambase",
    "mushroom",
    "bank-marketing",
    "PhishingWebsites",
    "Bioresponse",
    "census",
    "trec",
    "semeval",
    "tennis",
]

tag = "02"

for dataset in dataset_list:
    cmd = f"python plot_test_accuracy.py --dataset {dataset} --exp_type baseline --tag {tag}"
    print(cmd)
    os.system(cmd)