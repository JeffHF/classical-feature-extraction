# coding:utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

class AEModel(nn.Module):
    def __init__(self, n_inputs, hidden_layers):
        super(AEModel, self).__init__()

        encoder_layers = []
        prev_dim = n_inputs
        for i, h in enumerate(hidden_layers):
            encoder_layers.append(nn.Linear(prev_dim, h))
            if i != len(hidden_layers) - 1:
                encoder_layers.append(nn.ReLU())
            prev_dim = h

        decoder_layers = []
        rev_hidden = hidden_layers[:-1][::-1]
        prev_dim = hidden_layers[-1]
        for h in rev_hidden:
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            prev_dim = h

        decoder_layers.append(nn.Linear(prev_dim, n_inputs))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return z, out

def AutoEncoder(X,
                n_dim,
                hidden_layers=None,
                noise=0.0,
                drop_rate=0.0,
                n_epochs=300,
                learning_rate=0.01,
                optimizer_type='adam',
                verbose=1,
                seed=42):
    """
    :param X: (n_samples, n_features)
    :param n_dim: target dimensions
    :param hidden_layers: hidden layer structure, e.g. [128, 64]
    :param noise: Gaussian noise std
    :param drop_rate: dropout rate
    :param n_epochs: training epochs
    :param learning_rate: learning rate
    :param optimizer_type: 'adam' or 'sgd'
    :param verbose: print training info or not
    :param seed: random seed
    :return: (n_samples, n_dim)
    """
    set_seed(seed)

    n_samples, n_features = X.shape
    X = X.astype(np.float32)

    if hidden_layers is None:
        hidden_layers = []

    hidden_layers = list(hidden_layers) + [n_dim]

    model = AEModel(n_features, hidden_layers)
    criterion = nn.MSELoss()

    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    dropout = nn.Dropout(p=drop_rate)

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        X_input = X_tensor
        if noise > 0:
            X_input = X_input + noise * torch.randn_like(X_input)
        if drop_rate > 0:
            X_input = dropout(X_input)

        z, outputs = model(X_input)
        loss = criterion(outputs, X_tensor)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 and verbose:
            print(f"{epoch} Train MSE: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        data_ndim, _ = model(X_tensor)
        data_ndim = data_ndim.cpu().numpy()

    return data_ndim


if __name__ == '__main__':
    digits = load_digits()
    X = digits.data
    Y = digits.target

    data_2d1 = AutoEncoder(
        X,
        2,
        hidden_layers=[],
        noise=0.0,
        drop_rate=0.0,
        n_epochs=1000,
        learning_rate=0.01,
        optimizer_type='adam',
        verbose=1
    )

    data_2d2 = PCA(n_components=2).fit_transform(X)

    plt.figure(figsize=(8, 4))

    plt.subplot(121)
    plt.title("my_AutoEncoder")
    plt.scatter(data_2d1[:, 0], data_2d1[:, 1], c=Y)

    plt.subplot(122)
    plt.title("sklearn_PCA")
    plt.scatter(data_2d2[:, 0], data_2d2[:, 1], c=Y)

    plt.tight_layout()
    plt.savefig("AutoEncoder.png")
    plt.show()