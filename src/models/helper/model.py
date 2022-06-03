import torch.nn as nn


class Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers2 = nn.Sequential(
            nn.Linear(3, 64),
            nn.Sigmoid(),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, 2),
        )
        self.layers2 = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(3, 8),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

    def forward(self, x):
        return self.layers(x)
