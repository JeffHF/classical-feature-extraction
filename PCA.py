import numpy as np

def pca(X, n_dim):
    """
    :param X: (n_samples, n_features)
    :param n_dim: target dimensions
    """

    n_samples, n_features = X.shape
    Xc = X - np.mean(X, axis=0, keepdims=True)
    if n_features > n_samples:
        Ncov = np.dot(Xc, np.transpose(Xc))
        eigvals, eigvecs = np.linalg.eigh(Ncov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        eigvals = eigvals[:n_dim]
        eigvecs = eigvecs[:, :n_dim]
        W = np.dot(Xc.T, eigvecs) / np.sqrt(eigvals.reshape(1, -1) + np.finfo(float).eps)
    else:
        cov = np.dot(np.transpose(Xc), Xc)
        eigvals, eigvecs = np.linalg.eig(cov)
        idx = np.argsort(-eigvals)[:n_dim]
        eigvecs = eigvecs[:, idx]
        W = eigvecs[:, :n_dim]
    return W

def pca_energy(X, energy_ratio=0.98):
    """
    :param X: (n_samples, n_features(D))
    :param energy_ratio: float, default=0.98
    """
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape
    Xc = X - np.mean(X, axis=0, keepdims=True)
    if n_features > n_samples:
        Ncov = np.dot(Xc, np.transpose(Xc))
        eigvals, eigvecs = np.linalg.eigh(Ncov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        cumulative_energy = np.cumsum(eigvals) / (np.sum(eigvals) + np.finfo(float).eps)
        n_dim = np.searchsorted(cumulative_energy, energy_ratio) + 1
        eigvals_selected = eigvals[:n_dim]
        eigvecs_selected = eigvecs[:, :n_dim]
        W = np.dot(Xc.T, eigvecs_selected) / np.sqrt(eigvals_selected.reshape(1, -1) + np.finfo(float).eps)
    else:
        cov = np.dot(np.transpose(Xc), Xc)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        cumulative_energy = np.cumsum(eigvals) / (np.sum(eigvals) + np.finfo(float).eps)
        n_dim = np.searchsorted(cumulative_energy, energy_ratio) + 1
        W = eigvecs[:, :n_dim]
    return W

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    data = load_digits()
    X = data.data
    Y = data.target
    W = pca(X, 2)
    data_2d1 = np.dot(X, W)

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
