import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class MachineDataSet(Dataset):
    def __init__(self, X, y, scale_data=True) -> None:
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            # Apply scaling if necessary
            if scale_data:
                X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index: any):
        return self.X[index], self.y[index]
