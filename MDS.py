import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import MDS as SklearnMDS


def cal_pairwise_dist(X):
    sum_X = np.sum(np.square(X), axis=1)
    dist = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    dist[dist < 0] = 0
    return dist


def MDS(X, n_dim):
    """
    :param X: (n_samples, n_features)
    :param n_dim: target dimensions
    :return: (n_samples, n_dim)
    """
    n_samples = X.shape[0]

    dist = cal_pairwise_dist(X)
    dist[dist < 0] = 0

    T1 = np.ones((n_samples, n_samples)) * np.sum(dist) / (n_samples ** 2)
    T2 = np.sum(dist, axis=1, keepdims=True) / n_samples
    T3 = np.sum(dist, axis=0, keepdims=True) / n_samples

    B = -(T1 - T2 - T3 + dist) / 2

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(-eigvals)[:n_dim]

    picked_eigvals = np.maximum(eigvals[idx], 0)
    picked_eigvecs = eigvecs[:, idx]

    data_ndim = picked_eigvecs * np.sqrt(picked_eigvals.reshape(1, -1))
    return data_ndim


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    Y = iris.target

    data_2d1 = MDS(X, 2)
    data_2d2 = SklearnMDS(n_components=2, random_state=0).fit_transform(X)

    plt.figure(figsize=(8, 4))

    plt.subplot(121)
    plt.title("my_MDS")
    plt.scatter(data_2d1[:, 0], data_2d1[:, 1], c=Y)

    plt.subplot(122)
    plt.title("sklearn_MDS")
    plt.scatter(data_2d2[:, 0], data_2d2[:, 1], c=Y)

    plt.tight_layout()
    plt.savefig("MDS.png")
    plt.show()