from wrench.synthetic.syntheticdataset import SyntheticDataset
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_syn_dataset(X, y, weights, labels, title, figpath):
    """
    PLot synthetic dataset with weak labels
    :return:
    """
    plt.figure()
    x0_min, x0_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    x1_min, x1_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    color = np.where(np.array(y) == 0, "red", "blue")
    plt.xlim(x0_min, x0_max)
    plt.ylim(x1_min, x1_max)
    plt.scatter(X[:,0],X[:,1], c=color)

    xx0, xx1 = np.meshgrid(np.linspace(x0_min, x0_max, 100),
                           np.linspace(x1_min, x1_max, 100))
    x_in = np.c_[xx0.ravel(), xx1.ravel()]
    x_in = np.hstack((x_in, np.ones((len(x_in),1))))
    act = x_in @ weights.T
    labels_mat = np.tile(labels, len(x_in)).reshape((len(x_in), -1))
    L = np.where(act >= 0, labels_mat, -1)

    for i in range(len(weights)):
        # iterate over weak label sources
        if labels[i] == 0:
            plt.contourf(xx0,xx1, L[:,i].reshape(xx0.shape), levels=[-1.5,-0.5,0.5], colors=["white","red"], alpha=0.1)
        else:
            plt.contourf(xx0,xx1, L[:,i].reshape(xx0.shape), levels=[-1.5,0,1.5], colors=["white", "blue"], alpha=0.1)

    plt.title(title)
    dirname = os.path.dirname(figpath)
    Path(dirname).mkdir(parents=True, exist_ok=True)
    plt.savefig(figpath)


def generate_gaussian_mixture(means, covs, sizes, labels, seed=0):
    """
    Generate datasets following gaussian mixture distribution
    :param means: mean value of the clusters
    :param covs: covariances of clusters
    :param sizes: sizes of clusters
    :param labels: ground truth labels of clusters
    :return: X, y: dataset
    """
    np.random.seed(seed)
    n_cluster = len(means)
    assert len(covs) == n_cluster and len(sizes) == n_cluster and len(labels) == n_cluster
    n_features = len(means[0])
    X = np.array([]).reshape((-1,n_features))
    y = np.array([])
    for i in range(n_cluster):
        Xc = np.random.multivariate_normal(means[i], covs[i], sizes[i])
        yc = np.repeat(labels[i], sizes[i])
        X = np.vstack((X,Xc))
        y = np.hstack((y,yc))

    return X, y.astype(int).tolist()


def generate_label_matrix(X, weights, labels):
    """
    generate weak labels by LFs
    LF: if w*X+b>=0 then assign label else -1
    :param X: feature values
    :param weights: weights of LFs
    :param labels: labels of LFs
    :return: L: label matrix
    """
    n_LF = len(weights)
    n_data = len(X)
    assert len(weights) == len(labels)
    Xb = np.hstack((X, np.ones((n_data, 1))))
    act = Xb @ weights.T
    labels = np.tile(labels, n_data).reshape((n_data,n_LF))
    L = np.where(act>=0, labels, -1)
    return L.tolist()


def generate_gm_dataset(X, y, L, train_split=0.8, valid_split=0.1, random_state=42):
    train_X, test_X, train_y, test_y, train_L, test_L = train_test_split(X, y, L, train_size=train_split,
                                                                         random_state=random_state)
    valid_X, test_X, valid_y, test_y, valid_L, test_L = train_test_split(test_X, test_y, test_L,
                                                                         train_size=valid_split/(1-train_split),
                                                                         random_state=random_state)

    train_data = SyntheticDataset(
        split="train",
        ids=[i for i in range(len(train_y))],
        labels=train_y,
        examples=[i for i in range(len(train_y))],
        weak_labels=train_L,
        id2label={0: 0, 1: 1},
        features=train_X
    )
    valid_data = SyntheticDataset(
        split="valid",
        ids=[i for i in range(len(valid_y))],
        labels=valid_y,
        examples=[i for i in range(len(valid_y))],
        weak_labels=valid_L,
        id2label={0: 0, 1: 1},
        features=valid_X
    )
    test_data = SyntheticDataset(
        split="test",
        ids=[i for i in range(len(test_y))],
        labels=test_y,
        examples=[i for i in range(len(test_y))],
        weak_labels=test_L,
        id2label={0: 0, 1: 1},
        features=test_X
    )
    return train_data, valid_data, test_data


def generate_syn_1():
    # two gaussian mixtures that have overlaps
    means = [[-1,-1], [1,1]]
    covs = [
            [[1,0],[0,1]],
            [[1,0],[0,1]]
           ]
    sizes = [500,500]
    labels = [0,1]
    X, y = generate_gaussian_mixture(means, covs, sizes, labels)
    weights = np.array(
        [[-1,0, 1],  # x<=1: 0
         [0,-1, 1],  # y<=1: 0
         [1, 0, 1],  # x>=-1: 1
         [0, 1, 1]]  # y>=-1: 1
    )
    labels = np.array([0,0,1,1])
    L = generate_label_matrix(X, weights, labels)
    train_data, valid_data, test_data = generate_gm_dataset(X, y, L)
    plot_syn_dataset(train_data.features, train_data.labels, weights, labels,
                     title="syn_1", figpath="output/syn_1/syn_1.png")
    return train_data, valid_data, test_data


def generate_syn_2():
    # six gaussian mixtures with two large ones and four small ones
    means = [
        [-9,9],
        [-9,-9],
        [0,6],
        [9,-9],
        [9,9],
        [0,-6]
    ]
    covs = [
        [[0.5,0],[0,0.5]],
        [[0.5,0],[0,0.5]],
        [[1,0],[0,1]],
        [[0.5, 0], [0, 0.5]],
        [[0.5, 0], [0, 0.5]],
        [[1, 0], [0, 1]]
    ]
    sizes = [50,50,400,50,50,400]
    labels = [0,0,0,1,1,1]
    X, y = generate_gaussian_mixture(means, covs, sizes, labels)
    weights = np.array(
        [[-1, 0, -1],   # x<=-1: 0
         [0, 1, 0],     # y>=0: 0
         [1, 0, -1],    # x>=1: 1
         [0, -1, 0]]    # y<=0: 1
    )
    labels = [0,0,1,1]
    L = generate_label_matrix(X, weights, labels)
    train_data, valid_data, test_data = generate_gm_dataset(X, y, L)
    plot_syn_dataset(train_data.features, train_data.labels, weights, labels,
                     title="syn_2", figpath="output/syn_2/syn_2.png")
    return train_data, valid_data, test_data


def generate_syn_3():
    # three gaussian mixtures forming a U shape
    means = [
       [3, 3],
       [-3,-3],
       [3,-3]
    ]
    covs = [
        [[1, -0.5], [-0.5, 1]],
        [[1, -0.5], [-0.5, 1]],
        [[1, 0.5], [0.5, 1]],
    ]
    sizes = [300, 300, 400]
    labels = [0, 0, 1]
    X, y = generate_gaussian_mixture(means, covs, sizes, labels)
    weights = np.array(
        [[-1, 0, 0],  # x<=0: 0
         [0, 1, 0],   # y>=0: 0
        [1, -1, 0]]  # x>=y: 1
    )
    labels = [0, 0, 1]
    L = generate_label_matrix(X, weights, labels)
    train_data, valid_data, test_data = generate_gm_dataset(X, y, L)
    plot_syn_dataset(train_data.features, train_data.labels, weights, labels,
                     title="syn_3", figpath="output/syn_3/syn_3.png")
    return train_data, valid_data, test_data


def generate_syn_4():
    # three gaussian mixtures with class imbalance
    means = [
       [-3, 0],
       [2, 1],
       [2,-1]
    ]
    covs = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 1]],
        [[1, 0], [0, 1]],
    ]
    sizes = [900, 50, 50]
    labels = [0, 0, 1]
    X, y = generate_gaussian_mixture(means, covs, sizes, labels)
    weights = np.array(
        [[-1, 0, 0],  # x<=0: 0
         [0, 1, 0],   # y>=0: 0
         [1, 0, 0]]  # x>=0: 1
    )
    labels = [0, 0, 1]
    L = generate_label_matrix(X, weights, labels)
    train_data, valid_data, test_data = generate_gm_dataset(X, y, L)
    plot_syn_dataset(train_data.features, train_data.labels, weights, labels,
                     title="syn_4", figpath="output/syn_4/syn_4.png")
    return train_data, valid_data, test_data

def generate_syn_5():
    # six gaussian mixtures forming a claw shape
    means = [
        [-6,0],
        [2,6],
        [2,-6],
        [-4,2],
        [-4,-2],
        [6,0]
    ]
    covs = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 1]],
        [[1, 0], [0, 1]],
        [[1, 0], [0, 1]],
        [[1, 0], [0, 1]],
        [[1, 0], [0, 1]],
    ]
    sizes = [200,200,200,200,200,200]
    labels = [0,0,0,1,1,1]
    X,y = generate_gaussian_mixture(means, covs, sizes, labels)
    weights = np.array(
        [
            [-1,0,0],  # x<=0: 0
            [0,1,-4],  # y>=4: 0
            [0,-1,-4], # y<=-4: 0
            [1,0,-4]   # x>=4: 1
        ]
    )
    labels = [0,0,0,1]
    L = generate_label_matrix(X, weights, labels)
    train_data, valid_data, test_data = generate_gm_dataset(X, y, L)
    plot_syn_dataset(train_data.features, train_data.labels, weights, labels,
                     title="syn_5", figpath="output/syn_5/syn_5.png")
    return train_data, valid_data, test_data


def generate_syn_6():
    # six gaussian mixtures forming a claw shape
    means = [
        [-6,0],
        [2,6],
        [2,-6],
        [-4,2],
        [-4,-2],
        [6,0]
    ]
    covs = [
        [[0.5, 0], [0, 0.5]],
        [[0.5, 0], [0, 0.5]],
        [[0.5, 0], [0, 0.5]],
        [[0.5, 0], [0, 0.5]],
        [[0.5, 0], [0, 0.5]],
        [[0.5, 0], [0, 0.5]],
    ]
    sizes = [200,400,400,200,200,400]
    labels = [0,0,0,1,1,1]
    X,y = generate_gaussian_mixture(means, covs, sizes, labels)
    weights = np.array(
        [
            [-1,0,0],  # x<=0: 0
            [0,1,-4],  # y>=4: 0
            [0,-1,-4], # y<=-4: 0
            [1,0,-4]   # x>=4: 1
        ]
    )
    labels = [0,0,0,1]
    L = generate_label_matrix(X, weights, labels)
    train_data, valid_data, test_data = generate_gm_dataset(X, y, L)
    plot_syn_dataset(train_data.features, train_data.labels, weights, labels,
                     title="syn_6", figpath="output/syn_6/syn_6.png")
    return train_data, valid_data, test_data



