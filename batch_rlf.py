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
    cmd = f"python main_rlf.py --dataset {dataset} --tag linearLF-all --sample_budget 350 " \
          f"--sample_append 50 --sample_revise 50"
    print(cmd)
    os.system(cmd)
    cmd = f"python main_rlf.py --dataset {dataset} --tag linearLF-uncov --sample_budget 350 " \
          f"--sample_append 50 --sample_revise 50 --only_append_uncovered"
    print(cmd)
    os.system(cmd)