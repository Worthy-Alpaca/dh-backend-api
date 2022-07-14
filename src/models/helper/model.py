import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_layers: int,
        n_units_layers: list,
        p: float = 0.2,
        activation: torch.nn = nn.GELU,
    ) -> nn.Module:
        """Subclass of a PyTorch Module. Creates a neural deep forward Network.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            n_layers (int): Number of layers in the Model.
            n_units_layers (list): Number of output features for each Model layer. Number must correspond with n_layers.
            p (float, optional): Dropout probability. Defaults to 0.2.
            activation (torch.nn, optional): Activation function. Defaults to nn.GELU.

        Returns:
            nn.Module: A PyTorch neural network module.
        """
        super(Network, self).__init__()
        self.input_size = in_features
        self.output_size = out_features
        layers = []
        layers2 = []

        running_features = in_features
        dropOut1 = nn.Dropout(p)
        activation1 = activation()

        for i in range(n_layers):
            layers.append(activation1)
            running_layer = nn.Linear(running_features, n_units_layers[i])
            torch.nn.init.xavier_normal_(
                running_layer.weight,  # gain=nn.init.calculate_gain("tanh", 0.2)
            )
            torch.nn.init.zeros_(running_layer.bias)
            layers.append(running_layer)
            layers.append(dropOut1)
            layers.append(nn.BatchNorm1d(n_units_layers[i]))
            running_features = n_units_layers[i]

        layers2.append(nn.Linear(running_features, out_features))
        layers2.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)
        self.layers2 = nn.Sequential(*layers2)

    def forward(self, x):
        x = self.layers(x)
        return self.layers2(x)
