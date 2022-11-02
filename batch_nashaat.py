import os

dataset_list = [
    "youtube",
    "trec",
    "imdb",
    "semeval"
]

for dataset in dataset_list:
    cmd = f"python nashaat_pipeline.py --dataset {dataset}"
    print(cmd)
    os.system(cmd)