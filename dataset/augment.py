import copy
import argparse
import json
from pathlib import Path
from .eda import eda


def augment_text(data_home, dataset, split="train", num_aug=2, method="eda", **kwargs):
    dataset_path = Path(data_home) / dataset / f"{split}.json"
    f = open(dataset_path)
    data = json.load(f)
    f.close()
    aug_data = []
    for i in range(num_aug):
        aug_data.append(copy.copy(data))

    for idx in data:
        text = data[idx]["data"]["text"]
        if method == "eda":
            aug_text = eda(text, num_aug=num_aug, **kwargs)
            for i in range(num_aug):
                aug_data[i][idx]["data"]["text"] = aug_text[i]

    for i in range(num_aug):
        aug_path = Path(data_home) / dataset / f"{split}_da{i}.json"
        with open(aug_path, 'w') as f:
            json.dump(aug_data[i], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="youtube")
    parser.add_argument("--dataset_path", type=str, default="../wrench-1.1/datasets/")