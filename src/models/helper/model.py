from turtle import forward
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self) -> None:
        super(Network, self).__init__()
        self.layers2 = nn.Sequential(
            nn.Linear(3, 64),
            nn.Sigmoid(),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, 2),
        )
        self.layers2 = nn.Sequential(
            nn.Linear(3, 4), nn.SiLU(), nn.Linear(4, 2), nn.Sigmoid()
        )
        
        self.logReg = nn.Sequential(nn.Sigmoid(), nn.Linear(3, 2))

    
        self.gru = nn.GRU(3, 20, num_layers=2)
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(20, 30)
        self.rnn = nn.RNN(30, 20, num_layers=2)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(20, 12)
        self.lstm = nn.LSTM(12, 4, num_layers=2)
        self.relu3 = nn.ReLU()
        self.linear3 = nn.Linear(4, 2)


    def forward(self, x):
        x, _ = self.gru(x)
        x = self.relu1(x)
        x = self.linear1(x)
        x, _ = self.rnn(x)
        x = self.relu2(x)
        x = self.linear2(x)
        x, _ = self.lstm(x)
        x = self.relu3(x)
        x = self.linear3(x)
        return x


class Network2(nn.Module):
    def __init__(self, n_layers: int, n_input_size: int, n_hidden_size: int, n_output_size) -> None:
        super(Network, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.GRU(n_input_size, n_hidden_size))
        for i in range(n_layers):
            if (i % 2) == 0:
                self.layers.append(nn.LSTMCell(n_hidden_size, n_hidden_size))
            else:   
                self.layers.append(nn.Linear(n_hidden_size, n_hidden_size))
            #self.layers.append(nn.Sigmoid())
        self.layers.append(nn.Linear(n_hidden_size, n_output_size))

    def forward(self, x):
        l = 0
        for layer in self.layers:
            #x = nn.ReLU(x)
            x, *states = layer(x)
        #x = F.softmax(self.layers[:-1](x))
        return x



if __name__ == '__main__':
    Network(5, 3, 10, 2)
