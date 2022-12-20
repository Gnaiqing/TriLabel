import os
from utils import compare_baseline_performance

dataset_list = [
    # "spambase",
    # "mushroom",
    # "youtube",
    "imdb",
    "yelp",
    # "tennis"
    # "PhishingWebsites",
    # "Bioresponse",
    # "bank-marketing",
    # "census",
    # "tennis",
    # "yelp"
]

text_datasets = [
    "youtube",
    "imdb",
    "yelp"
]

tag = "04"
device = "cuda:0"

for dataset in dataset_list:
    if dataset in text_datasets:
        ext = " --extract_fn bert"
    else:
        ext = ""

    dpal_cmd = f"python dpal_pipeline.py --device {device} --dataset {dataset} {ext} " \
                f"--use_valid_labels --use_soft_labels --tag {tag}"
    print(dpal_cmd)
    os.system(dpal_cmd)
    # relief_cmd = f"python main_rlf.py --device {device} --dataset {dataset} {ext} " \
    #              f"--use_valid_labels --use_soft_labels --tag {tag}"
    # print(relief_cmd)
    # os.system(relief_cmd)
    # al_cmd = f"python al_pipeline.py --device {device} --dataset {dataset} {ext} " \
    #          f"--use_valid_labels --use_soft_labels --tag {tag}"
    # print(al_cmd)
    # os.system(al_cmd)
    # nashaat_cmd = f"python main_rlf.py --device {device} --dataset {dataset} {ext} " \
    #               f"--sampler uncertain --revision_model expert-label --revision_type LF " \
    #               f"--use_valid_labels --use_soft_labels --tag {tag}"
    # print(nashaat_cmd)
    # os.system(nashaat_cmd)

# for dataset in dataset_list:
#     filepaths = {
#         "ReLieF": f"output/{dataset}/metal_mlp_dalen_uncertain-joint_{tag}.json",
#         "Active Learning": f"output/{dataset}/None_mlp_al_uncertain-rm_{tag}.json",
#         "Nashaat": f"output/{dataset}/metal_mlp_expert-label_uncertain_{tag}.json"
#     }
#     compare_baseline_performance(filepaths, dataset, tag)
