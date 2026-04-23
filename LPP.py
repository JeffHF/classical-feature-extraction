import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA


def rbf(dist, t=1.0):
    return np.exp(-(dist / t))

def cal_pairwise_dist(X):
    sum_X = np.sum(np.square(X), axis=1)
    dist = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    dist[dist < 0] = 0
    return dist

def cal_rbf_dist(X, n_neighbors=10, t=1.0):
    dist = cal_pairwise_dist(X)
    n = dist.shape[0]
    rbf_dist = rbf(dist, t)

    W = np.zeros((n, n))
    for i in range(n):
        index_ = np.argsort(dist[i])[1:1 + n_neighbors]
        W[i, index_] = rbf_dist[i, index_]
        W[index_, i] = rbf_dist[index_, i]

    return W


def LPP(X, n_dim, n_neighbors=30, t=1.0):
    """
    :param X: (n_samples, n_features)
    :param n_dim: target dimensions
    :param n_neighbors: k nearest neighbors
    :param t: parameter for rbf
    :return: (n_samples, n_dim)
    """
    n_samples = X.shape[0]

    W = cal_rbf_dist(X, n_neighbors=n_neighbors, t=t)
    D = np.diag(np.sum(W, axis=1))
    L = D - W

    XDXT = np.dot(np.dot(X.T, D), X)
    XLXT = np.dot(np.dot(X.T, L), X)

    eigvals, eigvecs = np.linalg.eig(np.dot(np.linalg.pinv(XDXT), XLXT))
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    sort_index = np.argsort(eigvals)
    eigvals = eigvals[sort_index]
    eigvecs = eigvecs[:, sort_index]

    j = 0
    while j < len(eigvals) and eigvals[j] < 1e-6:
        j += 1

    picked_eigvecs = eigvecs[:, j:j + n_dim]
    data_ndim = np.dot(X, picked_eigvecs)

    return data_ndim


if __name__ == '__main__':
    X = load_digits().data
    y = load_digits().target

    dist = cal_pairwise_dist(X)
    max_dist = np.max(dist)

    data_2d1 = LPP(X, 2, n_neighbors=5, t=0.01 * max_dist)
    data_2d2 = PCA(n_components=2).fit_transform(X)

    plt.figure(figsize=(8, 4))

    plt.subplot(121)
    plt.title("my_LPP")
    plt.scatter(data_2d1[:, 0], data_2d1[:, 1], c=y)

    plt.subplot(122)
    plt.title("sklearn_PCA")
    plt.scatter(data_2d2[:, 0], data_2d2[:, 1], c=y)

    plt.tight_layout()
    plt.savefig("LPP.png")
    plt.show()