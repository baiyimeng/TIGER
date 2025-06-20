import torch
import torch.nn as nn


def activation_layer(name="relu"):
    name = name.lower() if isinstance(name, str) else None
    return {
        "relu": nn.ReLU(),
        "leakyrelu": nn.LeakyReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "none": None,
        None: None,
    }.get(name, None)


class MLPLayers(nn.Module):
    def __init__(self, layers, dropout=0.0, activation="relu", bn=False):
        super().__init__()
        self.mlp_layers = self.build_mlp(layers, dropout, activation, bn)
        self.apply(self.init_weights)

    def build_mlp(self, layers, dropout, activation, bn):
        modules = []
        activation_func = activation_layer(activation)

        for i in range(len(layers) - 1):
            in_dim, out_dim = layers[i], layers[i + 1]
            modules.append(nn.Linear(in_dim, out_dim))

            if bn:
                modules.append(nn.BatchNorm1d(out_dim))
            if activation_func is not None and i != len(layers) - 2:
                modules.append(activation_func)
            if dropout > 0.0:
                modules.append(nn.Dropout(dropout))

        return nn.Sequential(*modules)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp_layers(x)


def sinkhorn_algorithm(distances, epsilon=0.05, sinkhorn_iterations=50):
    Q = torch.exp(-distances / epsilon)  # [B, K]
    Q = Q / (Q.sum() + 1e-8)  # normalize globally to avoid numerical issues

    B, K = Q.shape

    for _ in range(sinkhorn_iterations):
        Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-8)  # row normalize
        Q = Q / B
        Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-8)  # col normalize
        Q = Q / K

    return Q * B  # ensure columns sum to 1
