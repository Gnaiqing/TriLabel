import os

dataset_list = [
    "agnews",
    "imdb",
    "sms",
    "trec",
    "yelp",
    "youtube"
]

for dataset in dataset_list:
    # cmd = f"python main_rlf.py --dataset {dataset} --tag linearLF-all --sample_budget 500 " \
    #       f"--sample_append 100 --sample_revise 100"
    # cmd = f"python main_rlf.py --dataset {dataset} --tag linearLF-all_predy --sample_budget 500 " \
    #       f"--sample_append 100 --sample_revise 100 --contrastive_mode golden --plot_tsne"
    cmd = f"python exp_golden_revise.py --dataset {dataset}"
    print(cmd)
    os.system(cmd)
    # cmd = f"python main_rlf.py --dataset {dataset} --tag linearLF-uncov_goldcst --sample_budget 500 " \
    #       f"--sample_append 100 --sample_revise 100 --only_append_uncovered --contrastive_mode golden"
    # print(cmd)
    # os.system(cmd)