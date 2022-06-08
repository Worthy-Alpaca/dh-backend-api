from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_


class Network(nn.Module):
    def __init__(self, n_inputs: int, n_output: int) -> None:
        super(Network, self).__init__()
        self.hidden1 = nn.Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity="relu")
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity="relu")
        self.act2 = nn.ReLU()
        # third hidden layer and output
        self.hidden3 = nn.Linear(8, 3)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = nn.Softmax(dim=1)
        """self.layers2 = nn.Sequential(
            nn.Linear(15, 64),
            nn.Sigmoid(),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, 3),
        )
        self.layers3 = nn.Sequential(
            nn.Linear(3, 4), nn.SiLU(), nn.Linear(4, n_output), nn.Sigmoid()
        )

        self.logReg = nn.Sequential(nn.Sigmoid(), nn.Linear(3, 2))
        self.lstm = nn.LSTM(n_input, 15, num_layers=2)

        self.layers = nn.Sequential(
            nn.Linear(n_input, n_output),
            nn.ReLU(),
        )"""

    def forward(self, x):
        x = self.layers(x)
        # x, _ = self.lstm(x)
        # x = self.layers2(x)
        # x = self.layers3(x)
        """x, _ = self.gru(x)
        x = self.relu1(x)
        x = self.linear1(x)
        x, _ = self.rnn(x)
        x = self.relu2(x)
        x = self.linear2(x)
        x, _ = self.lstm(x)
        x = self.relu3(x)
        x = self.linear3(x)"""
        return x


class Network2(nn.Module):
    def __init__(
        self, n_layers: int, n_input_size: int, n_hidden_size: int, n_output_size
    ) -> None:
        super(Network, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.GRU(n_input_size, n_hidden_size))
        for i in range(n_layers):
            if (i % 2) == 0:
                self.layers.append(nn.LSTMCell(n_hidden_size, n_hidden_size))
            else:
                self.layers.append(nn.Linear(n_hidden_size, n_hidden_size))
            # self.layers.append(nn.Sigmoid())
        self.layers.append(nn.Linear(n_hidden_size, n_output_size))

    def forward(self, x):
        l = 0
        for layer in self.layers:
            # x = nn.ReLU(x)
            x, *states = layer(x)
        # x = F.softmax(self.layers[:-1](x))
        return x


if __name__ == "__main__":
    Network(5, 3, 10, 2)
