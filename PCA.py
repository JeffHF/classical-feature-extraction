import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pca(X, n_dim):
    """
    :param X: (n_samples, n_features(D))
    :param n_dim: target dimensions
    :return: (n_samples, n_dim)
    """

    n_samples, n_features = X.shape
    X = X - np.mean(X, axis=0, keepdims=True)

    if n_features > n_samples:
        N = n_samples
        Ncov = np.dot(X, X.T)

        Neigvals, Neigvecs = np.linalg.eig(Ncov)
        idx = np.argsort(-Neigvals)[:n_dim]
        Npicked_eigvals = Neigvals[idx]
        Npicked_eigvecs = Neigvecs[:, idx]

        picked_eigvecs = np.dot(X.T, Npicked_eigvecs)
        picked_eigvecs = picked_eigvecs / (N * Npicked_eigvals.reshape(-1, n_dim)) ** 0.5

        data_ndim = np.dot(X, picked_eigvecs)
    else:
        cov = np.dot(X.T, X)

        eigvals, eigvecs = np.linalg.eig(cov)
        idx = np.argsort(-eigvals)[:n_dim]

        picked_eigvals = eigvals[idx]
        picked_eigvecs = eigvecs[:, idx]

        data_ndim = np.dot(X, picked_eigvecs)

    return data_ndim

def pca_energy(X, energy_ratio=0.98):
    """
    :param X: (n_samples, n_features(D))
    :param energy_ratio: float, default=0.98
    :return: (n_samples, n_dim)
    """

    n_samples, n_features = X.shape
    X = X - np.mean(X, axis=0, keepdims=True)

    if n_features > n_samples:
        N = n_samples
        Ncov = np.dot(X, X.T)

        Neigvals, Neigvecs = np.linalg.eig(Ncov)
        idx = np.argsort(-Neigvals)
        Npicked_eigvals = Neigvals[idx]
        Npicked_eigvecs = Neigvecs[:, idx]

        cumulative_energy = np.cumsum(Neigvals) / np.sum(Neigvals)
        n_dim = np.searchsorted(cumulative_energy, energy_ratio) + 1

        picked_eigvecs = np.dot(X.T, Npicked_eigvecs)
        picked_eigvecs = picked_eigvecs / (N * Npicked_eigvals.reshape(-1, n_dim)) ** 0.5

        data_ndim = np.dot(X, picked_eigvecs)

    else:
        cov = np.dot(X.T, X)

        eigvals, eigvecs = np.linalg.eig(cov)
        idx = np.argsort(-eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        cumulative_energy = np.cumsum(eigvals) / np.sum(eigvals)
        n_dim = np.searchsorted(cumulative_energy, energy_ratio) + 1

        picked_eigvals = eigvals[:n_dim]
        picked_eigvecs = eigvecs[:, n_dim]
        data_ndim = np.dot(X, picked_eigvecs)

    return data_ndim

if __name__ == "__main__":
    data = load_iris()
    X = data.data
    Y = data.target
    data_2d1 = pca(X, 2)

    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.title("my_PCA")
    plt.scatter(data_2d1[:, 0], data_2d1[:, 1], c = Y)

    sklearn_pca = PCA(n_components=2)
    data_2d2 = sklearn_pca.fit_transform(X)
    plt.subplot(122)
    plt.title("sklearn_PCA")
    plt.scatter(data_2d2[:, 0], data_2d2[:, 1], c = Y)

    plt.tight_layout()
    plt.savefig("PCA.png")
    plt.show()