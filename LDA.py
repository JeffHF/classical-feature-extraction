import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def LDA(X, y, n_dim):
    """
    :param X: (n_samples, n_features)
    :param y: (n_samples,)
    :param n_dim: target dimensions
    :return: (n_samples, n_dim)
    """
    classes = np.unique(y)
    n_classes = len(classes)
    n_features = X.shape[1]

    if n_dim > n_classes - 1:
        raise ValueError("n_dim must be <= number of classes - 1")


    mean_total = np.mean(X, axis=0)

    Sw = np.zeros((n_features, n_features))
    Sb = np.zeros((n_features, n_features))

    for c in classes:
        Xc = X[y == c]
        mean_c = np.mean(Xc, axis=0)

        Xc_centered = Xc - mean_c
        Sw += np.dot(Xc_centered.T, Xc_centered)

        n_c = Xc.shape[0]
        mean_diff = (mean_c - mean_total).reshape(-1, 1)
        Sb += n_c * np.dot(mean_diff, mean_diff.T)

    S = np.dot(np.linalg.pinv(Sw), Sb)
    eigvals, eigvecs = np.linalg.eig(S)

    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    idx = np.argsort(-eigvals)[:n_dim]
    W = eigvecs[:, idx]

    data_ndim = np.dot(X, W)
    return data_ndim


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    Y = iris.target

    data_2d1 = LDA(X, Y, 2)
    data_2d2 = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, Y)

    plt.figure(figsize=(8, 4))

    plt.subplot(121)
    plt.title("my_LDA")
    plt.scatter(data_2d1[:, 0], data_2d1[:, 1], c=Y)

    plt.subplot(122)
    plt.title("sklearn_LDA")
    plt.scatter(data_2d2[:, 0], data_2d2[:, 1], c=Y)

    plt.tight_layout()
    plt.savefig("LDA.png")
    plt.show()