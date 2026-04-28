import numpy as np

def cal_pairwise_dist(X):
    sum_X = np.sum(np.square(X), axis=1)
    dist = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    dist[dist < 0] = 0
    return dist

def cal_perplexity(dist, idx=0, beta=1.0):
    prob = np.exp(-dist * beta)
    prob[idx] = 0
    sum_prob = np.sum(prob)
    if sum_prob < 1e-12:
        prob = np.maximum(prob, 1e-12)
        perp = -12
    else:
        perp = np.log(sum_prob) + beta * np.sum(dist * prob) / sum_prob
        prob = prob / sum_prob
    return perp, prob

def search_prob(X, tol=1e-5, perplexity=30.0):
    """
    Binary search for beta and compute pairwise conditional probabilities.
    """
    print("Computing pairwise distances...")
    n, d = X.shape

    dist = cal_pairwise_dist(X)
    dist[dist < 0] = 0

    pair_prob = np.zeros((n, n))
    beta = np.ones((n, 1))
    base_perp = np.log(perplexity)
    for i in range(n):
        if i % 500 == 0:
            print("Computing pair_prob for point %d of %d ..." % (i, n))
        betamin = -np.inf
        betamax = np.inf
        perp, this_prob = cal_perplexity(dist[i], i, beta[i])
        perp_diff = perp - base_perp
        tries = 0
        while np.abs(perp_diff) > tol and tries < 50:
            if perp_diff > 0:
                betamin = beta[i].copy()
                if np.isinf(betamax):
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if np.isinf(betamin):
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            perp, this_prob = cal_perplexity(dist[i], i, beta[i])
            perp_diff = perp - base_perp
            tries += 1
        pair_prob[i] = this_prob
    print("Mean value of sigma:", np.mean(np.sqrt(1 / beta)))
    return pair_prob

def TSNE(X, n_dim, perplexity=30.0, max_iter=1000):
    """
    :param X: (n_samples, n_features)
    :param n_dim: target dimensions
    :param perplexity: perplexity parameter
    :param max_iter: maximum iterations
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, n_dim)
    dY = np.zeros((n, n_dim))
    iY = np.zeros((n, n_dim))
    gains = np.ones((n, n_dim))

    P = search_prob(X, tol=1e-5, perplexity=perplexity)
    P = P + P.T
    P = P / np.sum(P)
    P = P * 4
    P = np.maximum(P, 1e-12)

    oldC = None
    for it in range(max_iter):
        sum_Y = np.sum(np.square(Y), axis=1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
        num[np.arange(n), np.arange(n)] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        PQ = P - Q
        for i in range(n):
            dY[i] = np.sum(np.tile(PQ[:, i] * num[:, i], (n_dim, 1)).T * (Y[i] - Y), axis=0)
        if it < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + \
                (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.mean(Y, axis=0, keepdims=True)
        if (it + 1) % 100 == 0:
            C = np.sum(P * np.log(P / Q))
            if oldC is not None:
                ratio = C / oldC
                print("ratio:", ratio)
                if ratio >= 0.95:
                    break
            oldC = C
        if it == 100:
            P = P / 4
    return Y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
    from sklearn.manifold import TSNE as SklearnTSNE
    data = load_digits()
    X = data.data
    Y = data.target

    data_2d1 = TSNE(X, 2, perplexity=30.0, max_iter=1000)
    data_2d2 = SklearnTSNE(
        n_components=2,
        perplexity=30.0,
        max_iter=1000,
        random_state=0,
        init="random"
    ).fit_transform(X)

    plt.figure(figsize=(8, 4))

    plt.subplot(121)
    plt.title("my_TSNE")
    plt.scatter(data_2d1[:, 0], data_2d1[:, 1], c=Y)

    plt.subplot(122)
    plt.title("sklearn_TSNE")
    plt.scatter(data_2d2[:, 0], data_2d2[:, 1], c=Y)

    plt.tight_layout()
    plt.savefig("TSNE.png")
    plt.show()
