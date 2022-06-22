import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
from pathlib import Path
import os


class MachineDataSet(Dataset):
    def __init__(self, X, y, scale_data=True) -> None:
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            # Apply scaling if necessary
            # plt.plot(X)
            # plt.show()
            if scale_data:
                scaler = MinMaxScaler()
                scaler.fit(X)
                # X = scaler.transform(X)
                # plt.plot(normalized_X)
                # plt.show()
                # standardized_X = StandardScaler().fit_transform(X)
                # plt.plot(standardized_X)
                # plt.show()

                # pyplot.subplot(211)
                # pyplot.hist(standardized_X[:, 0])
                # pyplot.subplot(212)
                # pyplot.hist(standardized_X[:, 1])
                # pyplot.show()
                # histogram of target variable
                # pyplot.hist(y)
                # pyplot.show()

                # X = (X - np.min(X)) / (np.max(X) - np.min(X))
                # y = (y - np.min(y)) / (np.max(y) - np.min(y))
                X = MinMaxScaler().fit_transform(X)
                y = MinMaxScaler().fit_transform(y)
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)
        else:
            self.X = X
            self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index: any):
        return self.X[index], self.y[index]


if __name__ == "__main__":
    dataPath = Path(
        os.getcwd()
        + os.path.normpath("/data/logs/combined/machine_timings_combined.csv")
    )
    df = pd.read_csv(
        dataPath,
        low_memory=False,
        encoding="unicode_escape",
    )
    df = df[["placementsNeeded", "heads", "cph", "machine", "timeNeeded"]]

    df[["placementsNeeded", "heads", "cph", "timeNeeded"]] = df[
        ["placementsNeeded", "heads", "cph", "timeNeeded"]
    ].astype(int)

    def encode(x):
        if x == "m10":
            return 0
        else:
            return 1

    df["machine"] = df["machine"].apply(lambda x: encode(x))

    x = df.drop(["timeNeeded", "cph"], axis=1)

    y = df[["timeNeeded", "cph"]]

    data = x.to_numpy(dtype=np.float32)
    labels = y.to_numpy(dtype=np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    MachineDataSet(x_train, y_train)
