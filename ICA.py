import numpy as np

def whiten(X):
    X = np.asarray(X, dtype=np.float64)
    X_centered = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    D = np.diag(1.0 / np.sqrt(eigvals + 1e-12))
    W_white = np.dot(eigvecs, D).dot(np.transpose(eigvecs))
    X_white = np.dot(X_centered, W_white)
    return X_white

def logcosh(x, alpha=1.0):
    gx = np.tanh(alpha * x)
    g_x = alpha * (1.0 - gx ** 2)
    return gx, g_x.mean(axis=0)

def symmetric_decorrelation(W):
    s, u = np.linalg.eigh(np.dot(W, np.transpose(W)))
    D = np.diag(1.0 / np.sqrt(s + np.finfo(float).eps))
    return np.dot(np.dot(u, D), np.transpose(u)).dot(W)

def ica(X, n_dim=2, max_iter=200, tol=1e-4, random_state=0):
    """
    :param X: (n_samples, n_features)
    :param n_dim: target dimension
    """
    X = np.asarray(X, dtype=np.float64)
    X_white = whiten(X)
    n_samples, n_features = X_white.shape
    rng = np.random.default_rng(random_state)
    W = rng.normal(size=(n_dim, n_features))
    W = symmetric_decorrelation(W)
    for _ in range(max_iter):
        WX = np.dot(X_white, np.transpose(W))
        gwx, g_wx = logcosh(WX)
        W_new = (np.dot(np.transpose(gwx), X_white)) / n_samples - np.dot(np.diag(g_wx), W)
        W_new = symmetric_decorrelation(W_new)

        lim = np.max(np.abs(np.abs(np.diag(np.dot(W_new, W.T))) - 1.0))
        W = W_new
        if lim < tol:
            break
    return X_white, np.transpose(W)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.decomposition import FastICA
    data = load_iris()
    X = data.data
    Y = data.target
    X_white, W = ica(X=X, n_dim=2, max_iter=100, random_state=42)

    data_2d1 = np.dot(X_white, W)

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
