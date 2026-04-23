import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import FastICA

def whiten(X, eps=1e-10):
    X = np.asarray(X, dtype=np.float64)
    X_centered = X - X.mean(axis=0)

    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    W_white = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + eps)) @ eigvecs.T
    X_white = X_centered @ W_white
    return X_white

def logcosh(x, alpha=1.0):
    gx = np.tanh(alpha * x)
    g_x = alpha * (1.0 - gx ** 2)
    return gx, g_x.mean(axis=0)

def symmetric_decorrelation(W):
    s, u = np.linalg.eigh(W @ W.T)
    return u @ np.diag(1.0 / np.sqrt(s + 1e-10)) @ u.T @ W

def ICA(X, n_dim=2, max_iter=200, tol=1e-4, random_state=0):
    """
    :param X: (n_samples, n_features)
    :param n_dim: target dimension
    :return: (n_samples, n_dim)
    """
    X_white = whiten(X)
    n_samples, n_features = X_white.shape

    rng = np.random.default_rng(random_state)
    W = rng.normal(size=(n_dim, n_features))
    W = symmetric_decorrelation(W)

    for _ in range(max_iter):
        WX = X_white @ W.T
        gwx, g_wx = logcosh(WX)

        W_new = (gwx.T @ X_white) / n_samples - np.diag(g_wx) @ W
        W_new = symmetric_decorrelation(W_new)

        lim = np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1.0))
        W = W_new
        if lim < tol:
            break

    X_ica = X_white @ W.T
    return X_ica


if __name__ == "__main__":
    X = load_iris().data
    Y = load_iris().target

    data_2d1 = ICA(X, n_dim=2)

    sklearn_ica = FastICA(n_components=2, random_state=0, whiten="unit-variance")
    data_2d2 = sklearn_ica.fit_transform(X)

    plt.figure(figsize=(8, 4))

    plt.subplot(121)
    plt.title("my_ICA")
    plt.scatter(data_2d1[:, 0], data_2d1[:, 1], c=Y)

    plt.subplot(122)
    plt.title("sklearn_ICA")
    plt.scatter(data_2d2[:, 0], data_2d2[:, 1], c=Y)

    plt.tight_layout()
    plt.savefig("ICA.png")
    plt.show()