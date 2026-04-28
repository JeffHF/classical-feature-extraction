import numpy as np

def cal_pairwise_dist(X):
    sum_X = np.sum(np.square(X), axis=1)
    dist = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    dist[dist < 0] = 0
    return dist


def get_n_neighbors(X, n_neighbors=10):
    dist = cal_pairwise_dist(X)
    dist = np.sqrt(dist)
    n = dist.shape[0]
    N = np.zeros((n, n_neighbors), dtype=np.int32)
    for i in range(n):
        index_ = np.argsort(dist[i])[1:n_neighbors + 1]
        N[i] = index_
    return N

def LLE(X, n_dim, n_neighbors=10):
    """
    :param X: (n_samples, n_features)
    :param n_dim: target dimensions
    :param n_neighbors: number of nearest neighbors
    """
    X = np.asarray(X, dtype=float)
    N = get_n_neighbors(X, n_neighbors)
    n_samples, n_features = X.shape

    if n_neighbors > n_features:
        tol = 1e-3
    else:
        tol = 0.0
    W = np.zeros((n_neighbors, n_samples))
    I = np.ones((n_neighbors, 1))
    for i in range(n_samples):
        Xi = np.tile(X[i], (n_neighbors, 1)).T
        Ni = X[N[i]].T
        Si = np.dot((Xi - Ni).T, (Xi - Ni))
        Si = Si + np.eye(n_neighbors) * tol * np.trace(Si)
        Si_inv = np.linalg.pinv(Si)
        wi = np.dot(Si_inv, I) / np.dot(np.dot(I.T, Si_inv), I)[0, 0]
        W[:, i] = wi[:, 0]
    W_y = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        index = N[i]
        for j in range(n_neighbors):
            W_y[index[j], i] = W[j, i]
    I_y = np.eye(n_samples)
    M = np.dot((I_y - W_y), (I_y - W_y).T)
    eigvals, eigvecs = np.linalg.eigh(M)
    idx = np.argsort(eigvals)[1:n_dim + 1]
    data_ndim = eigvecs[:, idx]
    return data_ndim

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.manifold import LocallyLinearEmbedding

    data = load_iris()
    X = data.data
    Y = data.target

    data_2d1 = LLE(X, 2, n_neighbors=30)
    data_2d2 = LocallyLinearEmbedding(n_components=2, n_neighbors=30).fit_transform(X)

    plt.figure(figsize=(8, 4))

    plt.subplot(121)
    plt.title("my_LLE")
    plt.scatter(data_2d1[:, 0], data_2d1[:, 1], c=Y)

    plt.subplot(122)
    plt.title("sklearn_LLE")
    plt.scatter(data_2d2[:, 0], data_2d2[:, 1], c=Y)

    plt.tight_layout()
    plt.savefig("LLE.png")
    plt.show()
