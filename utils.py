import numpy as np
from wrench.labelmodel import Snorkel, DawidSkene, MajorityVoting, MeTaL
from baselines.active_weasul.label_model import LabelModel
from wrench.endmodel import EndClassifierModel, Cosine
from dataset.load_dataset import load_real_dataset, load_synthetic_dataset
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from pytorch_lightning import seed_everything
from pathlib import Path
import json

ABSTAIN = -1


def preprocess_data(args):
    if args.dataset[:3] == "syn":
        train_data, valid_data, test_data = load_synthetic_dataset(args.dataset)
    else:
        train_data, valid_data, test_data = load_real_dataset(args.dataset_path, args.dataset, args.extract_fn)

    if args.max_dim is not None and train_data.features.shape[1] > args.max_dim:
        # use truncated SVD to reduce feature dimensions
        pca = PCA(n_components=args.max_dim)
        pca.fit(train_data.features)
        train_data.features = pca.transform(train_data.features)
        valid_data.features = pca.transform(valid_data.features)
        test_data.features = pca.transform(test_data.features)

    if args.sample_budget < 1:
        args.sample_budget = np.ceil(args.sample_budget * len(train_data)).astype(int)
        args.sample_per_iter = np.ceil(args.sample_per_iter * len(train_data)).astype(int)
    else:
        args.sample_budget = int(args.sample_budget)
        args.sample_per_iter = int(args.sample_per_iter)

    return train_data, valid_data, test_data


def get_label_model(model_type, **kwargs):
    if model_type == "snorkel":
        label_model = Snorkel(lr=0.01, l2=0.0, n_epochs=100)
    elif model_type == "ds":
        label_model = DawidSkene()
    elif model_type == "mv":
        label_model = MajorityVoting()
    elif model_type == "metal":
        label_model = MeTaL(lr=0.01, n_epochs=100)
    elif model_type == "aw":
        if "penalty_strength" in kwargs:
            label_model = LabelModel(n_epochs=100, lr=0.01, active_learning=True,
                                     penalty_strength=kwargs["penalty_strength"])
        else:
            label_model = LabelModel(n_epochs=100, lr=0.01, active_learning=True)
    else:
        raise ValueError(f"label model {model_type} not supported.")
    return label_model


def get_end_model(model_type):
    if model_type == "mlp":
        end_model = EndClassifierModel(
            backbone="MLP",
            batch_size=512,
            test_batch_size=512,
            optimizer="Adam",
            optimizer_lr=1e-2,
            optimizer_weight_decay=1e-5
        )
    elif model_type == "bert":
        end_model = EndClassifierModel(
            batch_size=32,
            real_batch_size=32,
            test_batch_size=32,
            backbone="BERT",
            backbone_model_name="bert-base-cased",
            backbone_max_tokens=128,
            backbone_fine_tune_layers=-1,
            optimizer="AdamW",
            optimizer_lr=5e-5,
            optimizer_weight_decay=0.0
        )
    elif model_type == "roberta":
        end_model = EndClassifierModel(
            batch_size=32,
            real_batch_size=32,
            test_batch_size=32,
            backbone="BERT",
            backbone_model_name="roberta-base",
            backbone_max_tokens=128,
            backbone_fine_tune_layers=-1,
            optimizer="AdamW",
            optimizer_lr=5e-5,
            optimizer_weight_decay=0.0
        )
    elif model_type == "cosine-bert":
        end_model = Cosine(
            batch_size=32,
            real_batch_size=32,  # for accumulative gradient update
            test_batch_size=32,
            lamda=0.1,
            backbone='BERT',
            backbone_model_name='bert-base-cased',
            backbone_max_tokens=128,
            backbone_fine_tune_layers=-1,  # fine  tune all
            optimizer='AdamW',
            optimizer_lr=5e-5,
            optimizer_weight_decay=1e-4,
    )
    elif model_type == "cosine-roberta":
        end_model = Cosine(
            batch_size=32,
            real_batch_size=32,  # for accumulative gradient update
            test_batch_size=32,
            lamda=0.1,
            backbone='BERT',
            backbone_model_name='roberta-base',
            backbone_max_tokens=128,
            backbone_fine_tune_layers=-1,  # fine  tune all
            optimizer='AdamW',
            optimizer_lr=5e-5,
            optimizer_weight_decay=1e-4,
        )
    else:
        raise ValueError(f"end model {model_type} not implemented.")
    return end_model


def get_sampler(sampler_type, train_data, labeller, label_model=None, revision_model=None, encoder=None, seed=None):
    from sampler import PassiveSampler, UncertaintySampler, MaxKLSampler, CoreSetSampler, \
        RmUncertaintySampler, DALSampler, JointUncertaintySampler, ClusterMarginSampler, BadgeSampler
    if sampler_type == "passive":
        return PassiveSampler(train_data, labeller, label_model, revision_model, encoder, seed=seed)
    elif sampler_type == "uncertain":
        return UncertaintySampler(train_data, labeller, label_model, revision_model, encoder, seed=seed)
    elif sampler_type == "uncertain-rm":
        return RmUncertaintySampler(train_data, labeller, label_model, revision_model, encoder, seed=seed)
    elif sampler_type == "uncertain-joint":
        return JointUncertaintySampler(train_data, labeller, label_model, revision_model, encoder, seed=seed)
    elif sampler_type == "dal":
        return DALSampler(train_data, labeller, label_model, revision_model, encoder, seed=seed)
    elif sampler_type == "cluster-margin":
        return ClusterMarginSampler(train_data, labeller, label_model, revision_model, encoder, seed=seed)
    elif sampler_type == "maxkl":
        return MaxKLSampler(train_data, labeller,label_model, revision_model, encoder, seed=seed)
    elif sampler_type == "badge":
        return BadgeSampler(train_data, labeller, label_model, revision_model, encoder, seed=seed)
    elif sampler_type == "coreset":
        return CoreSetSampler(train_data, labeller, label_model, revision_model, encoder, seed=seed)
    else:
        raise ValueError(f"sampler {sampler_type} not implemented.")


def get_reviser(reviser_type, train_data, valid_data, encoder,  device, seed):
    from reviser import EnsembleReviser, ExpertLabelReviser, MCDropoutReviser, \
        MLPReviser, MLPTempReviser
    if reviser_type == "mlp":
        return MLPReviser(train_data, encoder, device, valid_data, seed)
    elif reviser_type == "mlp-temp":
        return MLPTempReviser(train_data, encoder, device, valid_data, seed)
    elif reviser_type == "ensemble":
        return EnsembleReviser(train_data, encoder, device, valid_data, seed)
    elif reviser_type == "expert-label":
        return ExpertLabelReviser(train_data, encoder, device, valid_data, seed)
    elif reviser_type == "mc-dropout":
        return MCDropoutReviser(train_data, encoder, device, valid_data, seed)
    else:
        raise ValueError(f"reviser {reviser_type} not implemented.")


def evaluate_label_quality(labels, probs):
    N = len(labels)
    n_class = probs.shape[1]
    labels = np.array(labels)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(labels, preds)
    # to prevent inf values, use laplace smoothing before evaluting nll
    smoothed_probs = (probs + 1e-6) / (1 + 1e-6 * probs.shape[1])
    nll = - np.log(smoothed_probs[np.arange(N), labels]).mean().item()
    brier_score = (1 - 2 * probs[np.arange(N), labels] + np.sum(probs ** 2, axis=1)) / n_class
    brier_score = brier_score.mean().item()
    return acc, nll, brier_score


def evaluate_end_model(pred_train_data, pred_train_labels, valid_data, test_data, args, seed):
    """
    Evaluate end model's performance with predicted labels
    """
    seed_everything(seed, workers=True)
    end_model = get_end_model(args.end_model)
    n_steps = int(np.ceil(len(pred_train_data) / end_model.hyperparas["batch_size"])) * args.em_epochs
    evaluation_step = int(np.ceil(len(pred_train_data) / end_model.hyperparas["batch_size"]))  # evaluate every epoch
    end_model.fit(
        dataset_train=pred_train_data,
        y_train=pred_train_labels,
        dataset_valid=valid_data,
        y_valid=valid_data.labels,
        metric=args.metric,
        device=args.device,
        verbose=args.verbose,
        n_steps=n_steps,
        evaluation_step=evaluation_step
    )
    test_acc = end_model.test(test_data, "acc", device=args.device)
    test_f1 = end_model.test(test_data, "f1_macro", device=args.device)
    perf = {
        "test_acc": test_acc,
        "test_f1": test_f1
    }
    return perf, end_model


def save_results(results_list, output_path, dataset, filename):
    filepath = Path(output_path) / dataset / filename
    dirname = Path(output_path) / dataset
    Path(dirname).mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as write_file:
        json.dump({"data": results_list}, write_file, indent=4)


def update_results(results_dict, **kwargs):
    for key in kwargs:
        results_dict[key].append(kwargs[key])