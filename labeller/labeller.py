def oracle(item):
    return item["label"]


def get_labeller(labeller_type):
    if labeller_type == "oracle":
        labeller = oracle
    else:
        raise ValueError(f"labeller {labeller_type} not implemented.")
    return labeller


def get_item(data, idx):
    item = {
        "id": data.ids[idx],
        "label": data.labels[idx],
        "example": data.examples[idx],
        "weak_labels": data.weak_labels[idx]
    }
    if data.features is not None:
        item["features"] = data.features[idx,:]
    return item