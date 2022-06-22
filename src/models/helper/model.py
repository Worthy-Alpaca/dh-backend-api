import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_


class Network4(nn.Module):
    def __init__(
        self, embedding_size, num_numerical_cols, output_size, layers, p=0.4
    ) -> None:
        super().__init__()
        self.all_embedings = nn.ModuleList(
            [nn.Embedding(ni, nf) for ni, nf in embedding_size]
        )
        self.embedding_dropout = nn.Dropout(p)
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols + num_numerical_cols

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))
        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical, x_numerical):
        embeddings = []
        for i, e in enumerate(self.all_embedings):
            embeddings.append(e(x_categorical[:, i]))
        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)

        x_numerical = self.batch_norm_num(x_numerical)
        x = torch.cat([x, x_numerical], 1)
        x = self.layers(x)
        return x


class Network9(nn.Module):
    def __init__(self, n_inputs: int, layers, n_output: int) -> None:
        super(Network, self).__init__()
        self.hidden1 = nn.Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity="relu")
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity="relu")
        self.act2 = nn.ReLU()
        # third hidden layer and output
        self.hidden3 = nn.Linear(8, n_output)
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
        # x = self.layers(x)
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
        x = self.hidden1(x)
        x = self.act1(x)
        x = self.hidden2(x)
        x = self.act2(x)
        x = self.hidden3(x)
        x = self.act3(x)
        return x


class Network0(nn.Module):
    """THIS WORKS RELATIVLY GOOD"""

    def __init__(self, input_size: int, layers, n_output_size, p=0.4) -> None:
        super().__init__()
        self.embedding_dropout = nn.Dropout(p)
        self.batch_norm_num = nn.BatchNorm1d(input_size)
        self.lstm1 = nn.LSTM(input_size, input_size, num_layers=3)

        all_layers = []

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], n_output_size))
        self.layers = nn.Sequential(*all_layers)
        self.outputRelu = nn.ReLU()
        self.lstm2 = nn.LSTM(n_output_size, n_output_size, num_layers=2)
        self.seq = nn.Sequential(nn.Sigmoid(), nn.Linear(input_size, n_output_size))

    def forward(self, x):
        # print(x.shape, 1)
        # x = self.embedding_dropout(x)
        # print(x.shape, 2)
        # x = self.batch_norm_num(x)
        # print(x.shape, 3)
        # x, _ = self.lstm1(x)
        # print(x.shape, 4)
        # x = self.layers(x)
        # x = self.outputRelu(x)
        # x, _ = self.lstm2(x)
        # x = self.outputRelu(x)
        return self.seq(x)


class Network7(nn.Module):
    def __init__(self, input_size: int, layers, n_output_size, p=0.2) -> None:
        super().__init__()

        # self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_size, input_size * 2)
        self.batchnorm1 = nn.BatchNorm1d(input_size * 2)
        self.dropout = nn.Dropout(p)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(input_size * 2, n_output_size**2)
        self.lstm = nn.LSTM(n_output_size**2, n_output_size, num_layers=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x, _ = self.lstm(x)
        x = self.sigmoid(x)
        return x


class Network3(torch.nn.Module):
    def __init__(self, input_size: int, layers, output_size, p=0.2):
        super(Network, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.act1 = torch.nn.GELU()
        self.act2 = torch.nn.GELU()
        self.act3 = torch.nn.GELU()
        self.act4 = torch.nn.GELU()
        self.hid1 = torch.nn.Linear(input_size, 7)  # 8-(10-10)-1
        self.lstm1 = torch.nn.RNN(7, 14, dropout=p, num_layers=3)
        self.hid2 = torch.nn.Linear(7, 14)
        self.lstm2 = torch.nn.GRU(10, 5, dropout=p, num_layers=3)
        self.hid3 = torch.nn.Linear(14, 20)
        self.hid4 = torch.nn.Linear(20, 14)
        self.oupt = torch.nn.Linear(14, output_size)
        self.dropout1 = nn.Dropout(p)
        self.dropout2 = nn.Dropout(p)
        self.dropout3 = nn.Dropout(p)
        # torch.nn.init.xavier_uniform_(self.hid1.weight)
        # torch.nn.init.zeros_(self.hid1.bias)
        # torch.nn.init.xavier_uniform_(self.hid2.weight)
        # torch.nn.init.zeros_(self.hid2.bias)
        # torch.nn.init.xavier_uniform_(self.oupt.weight)
        # torch.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = self.act1(x)
        z = self.hid1(z)
        z = self.dropout1(z)
        # z = self.act1(z)
        # z, _ = self.lstm1(z)
        # z = self.dropout2(z)
        z = self.act1(z)
        z = self.hid2(z)
        z = self.dropout1(z)
        z = self.act2(z)
        z = self.hid3(z)
        z = self.dropout2(z)
        z = self.act3(z)
        z = self.hid4(z)
        # z, _ = self.lstm2(z)
        z = self.dropout2(z)
        z = self.act2(z)
        z = self.oupt(z)  # no activation
        return torch.sigmoid(z)  # torch.sigmoid(z)


class Network(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        n_layers,
        n_units_layers,
        p=0.2,
        activation=nn.GELU,
    ) -> None:
        super(Network, self).__init__()
        self.input_size = in_features
        self.output_size = out_features
        layers = []
        layers2 = []

        # self.recurrent = nn.GRU(in_features, 7, num_layers=3, dropout=p)
        running_features = in_features

        dropOut1 = nn.Dropout(p)
        activation1 = activation()

        for i in range(n_layers):
            layers.append(nn.AdaptiveMaxPool1d(running_features))
            layers.append(activation1)
            running_layer = nn.Linear(running_features, n_units_layers[i])
            torch.nn.init.orthogonal_(
                running_layer.weight, gain=nn.init.calculate_gain("relu", 0.2)
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
        # x, _ = self.recurrent(x)
        x = self.layers(x)
        return self.layers2(x)


if __name__ == "__main__":
    net = Network(6, [200, 100, 50], 2)
    print(net)
