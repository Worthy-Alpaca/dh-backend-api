import torch
import numpy as np
from torch.utils.data import Dataset


class MachineDataSet(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Initiate a Dataset for the Neural Network.

        Args:
            X (np.ndarray): The Datapoints for X.
            y (np.ndarray): The Datapoints for Y.
        """
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)
        else:
            self.X = X
            self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index: any):
        return self.X[index], self.y[index]
