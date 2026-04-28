import numpy as np
from scipy.spatial.distance import pdist, squareform

def sigmoid(X, coef=0.25):
    X = np.dot(X, X.T)
    return np.tanh(coef * X + 1)

def linear(X):
    X = np.dot(X, X.T)
    return X

def rbf(X, gamma=15):
    sq_dists = pdist(X, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    return np.exp((-gamma * mat_sq_dists))

def kpca(X, n_dim=10, kernel=rbf):
    """
    :param data: (n_samples, n_features)
    :param n_dims: target n_dims
    :param kernel: kernel functions)
    """
    X = np.asarray(X, dtype=float)
    K = kernel(X)
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    eigvals, eigvecs = np.linalg.eig(K)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx[:n_dim]]
    eigvecs = eigvecs[:, idx[:n_dim]]
    eigvals = eigvals ** 0.5
    Vi = eigvecs / eigvals.reshape(1, -1)
    return K, Vi


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.decomposition import KernelPCA
    import matplotlib.pyplot as plt
    data = load_iris()
    X = data.data
    Y = data.target
    K, V = kpca(X, kernel=rbf)
    data_2d1 = np.dot(K, V)
    sklearn_kpca = KernelPCA(n_components=2, kernel="rbf", gamma=15)
    data_2d2 = sklearn_kpca.fit_transform(X)

    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.title("my_KPCA")
    plt.scatter(data_2d1[:, 0], data_2d1[:, 1], c=Y)

    plt.subplot(122)
    plt.title("sklearn_KPCA")
    plt.scatter(data_2d2[:, 0], data_2d2[:, 1], c=Y)

    plt.tight_layout()
    plt.savefig("KPCA.png")
    plt.show()
