import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve
from sklearn.manifold import Isomap as SklearnIsomap


def floyd(dist, n_neighbors=15):
    max_dist = np.max(dist) * 1000
    n_samples = dist.shape[0]

    dist_new = np.ones((n_samples, n_samples)) * max_dist
    dist_arg = np.argsort(dist, axis=1)

    for i in range(n_samples):
        dist_new[i, dist_arg[i, :n_neighbors + 1]] = dist[i, dist_arg[i, :n_neighbors + 1]]

    for k in range(n_samples):
        for i in range(n_samples):
            for j in range(n_samples):
                if dist_new[i, k] + dist_new[k, j] < dist_new[i, j]:
                    dist_new[i, j] = dist_new[i, k] + dist_new[k, j]

    return dist_new


def cal_pairwise_dist(X):
    sum_X = np.sum(np.square(X), axis=1)
    dist = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    dist[dist < 0] = 0
    return dist


def mds(dist, n_dim):
    dist = dist ** 2
    n_samples = dist.shape[0]

    T1 = np.ones((n_samples, n_samples)) * np.sum(dist) / (n_samples ** 2)
    T2 = np.sum(dist, axis=1, keepdims=True) / n_samples
    T3 = np.sum(dist, axis=0, keepdims=True) / n_samples

    B = -(T1 - T2 - T3 + dist) / 2

    eigvals, eigvecs = np.linalg.eigh(B)
    index_ = np.argsort(-eigvals)[:n_dim]

    picked_eigvals = np.maximum(eigvals[index_], 0)
    picked_eigvecs = eigvecs[:, index_]

    data_ndim = picked_eigvecs * np.sqrt(picked_eigvals.reshape(1, -1))
    return data_ndim


def Isomap(X, n_dim, n_neighbors=30):
    """
    :param X: (n_samples, n_features)
    :param n_dim: target dimensions
    :param n_neighbors: number of nearest neighbors
    :return: (n_samples, n_dim)
    """
    dist = cal_pairwise_dist(X)
    dist = np.sqrt(dist)

    dist_floyd = floyd(dist, n_neighbors)
    data_ndim = mds(dist_floyd, n_dim)

    return data_ndim


if __name__ == '__main__':
    X, Y = make_s_curve(n_samples=500, noise=0.1, random_state=42)

    data_2d1 = Isomap(X, 2, n_neighbors=10)
    data_2d2 = SklearnIsomap(n_neighbors=10, n_components=2).fit_transform(X)

    plt.figure(figsize=(8, 4))

    plt.subplot(121)
    plt.title("my_Isomap")
    plt.scatter(data_2d1[:, 0], data_2d1[:, 1], c=Y)

    plt.subplot(122)
    plt.title("sklearn_Isomap")
    plt.scatter(data_2d2[:, 0], data_2d2[:, 1], c=Y)

    plt.tight_layout()
    plt.savefig("ISOMAP.png")
    plt.show()