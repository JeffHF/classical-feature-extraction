import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import TruncatedSVD


def SVD(X, n_dim):
    """
    :param X: (n_samples, n_features)
    :param n_dim: target dimensions
    :return:
        data_ndim: (n_samples, n_dim)
        U: (n_samples, n_dim)
        Sigma: (n_dim, n_dim)
        VT: (n_dim, n_features)
    """
    X = X - np.mean(X, axis=0, keepdims=True)

    U, s, VT = np.linalg.svd(X, full_matrices=False)

    U = U[:, :n_dim]
    s = s[:n_dim]
    VT = VT[:n_dim, :]

    Sigma = np.diag(s)
    data_ndim = np.dot(U, Sigma)

    return data_ndim, U, Sigma, VT


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    Y = iris.target

    data_2d1, U, Sigma, VT = SVD(X, 2)
    data_2d2 = TruncatedSVD(n_components=2, random_state=0).fit_transform(X)

    plt.figure(figsize=(8, 4))

    plt.subplot(121)
    plt.title("my_SVD")
    plt.scatter(data_2d1[:, 0], data_2d1[:, 1], c=Y)

    plt.subplot(122)
    plt.title("sklearn_SVD")
    plt.scatter(data_2d2[:, 0], data_2d2[:, 1], c=Y)

    plt.tight_layout()
    plt.savefig("SVD.png")
    plt.show()