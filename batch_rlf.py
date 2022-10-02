import os

dataset_list = [
    "agnews",
    "imdb",
    "sms",
    "trec",
    "yelp",
    # "youtube"
]

for dataset in dataset_list:
    cmd = f"python main_rlf.py --dataset {dataset} --tag non_contrast --sample_budget 500 " \
          f"--sample_budget_init 100 --sample_budget_inc 100"
    print(cmd)
    os.system(cmd)
    cmd = f"python main_rlf.py --dataset {dataset} --tag golden --contrastive_mode golden --sample_budget 500 " \
          f"--sample_budget_init 100 --sample_budget_inc 100"
    print(cmd)
    os.system(cmd)