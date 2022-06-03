from pathlib import Path
from types import FunctionType
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from helper.MachineDataSet import MachineDataSet
from helper.model import Network


class MachinePredictions:
    def __init__(self, dataPath: str | Path) -> None:
        torch.manual_seed(42)
        self.dataPath = dataPath
        self.model = Network()

    def fit(
        self,
        epochs: int,
        trainloader: DataLoader,
        testloader: DataLoader,
        learning_rate: float = 1e-4,
        optimizer: FunctionType = None,
        loss_function: FunctionType = None,
    ):
        """
        - `epochs` : Number of epochs to train
        - `trainloader` : `torch.utils.data.DataLoader` Class with training data
        - `testloader` : `torch.utils.data.DataLoader` Class with validation data
        - `learning_rate` : learning rate to be applied
        - `optimizer` : optimizer function -> default is ADAM
        - `loss_function` : loss function -> default is MSE Loss
        """
        self.epochs = epochs
        self._train_losses = []
        self._train_accuracies = []
        self._test_losses = []
        self._test_accuracies = []
        optimizer = torch.optim.Adam if optimizer == None else optimizer
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self._loss_function = (
            torch.nn.MSELoss() if loss_function == None else loss_function
        )

        best_accu = 0
        for epoch in range(epochs):
            self.__train(epoch, trainloader)
            test_accu = self.__test(epoch, testloader)
            if test_accu > best_accu:
                self.saveModel(epoch)
                best_accu = test_accu

        return (self._train_losses, self._train_accuracies), (
            self._test_losses,
            self._test_accuracies,
        )

    def saveModel(self, epoch: int):
        torch.save(self.model.state_dict(), f"data/model/machineModel_{epoch}.model")
        print("Checkpoint saved")

    def __test(self, epoch: int, testloader: DataLoader):
        self.model.eval()
        current_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in tqdm(testloader, ascii=True):
                inputs, targets = data[0], data[1]
                inputs, targets = inputs.float(), targets.float()
                targets = targets.reshape((targets.shape[0], 2))

                outputs = self.model(inputs)

                test_loss = self._loss_function(outputs, targets)

                current_loss += test_loss.item()

                _, predicted = torch.max(outputs, 0)

                total += targets.size(0)

                correct += torch.sum(predicted == targets.data).float()

        test_loss = current_loss / len(testloader)
        accu = 100.0 * correct / total

        self._test_losses.append(test_loss)
        self._test_accuracies.append(accu)
        print("Test Loss: %.3f | Accuracy: %.3f" % (test_loss, accu))
        return accu.item()

    def __train(self, epoch: int, trainloader: DataLoader):
        print(f"Starting epoch {epoch+1} / {self.epochs} ")
        self.model.train()
        current_loss = 0.0
        correct = 0
        total = 0
        for data in tqdm(trainloader, ascii=True):
            inputs, targets = data[0], data[1]
            # inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 2))

            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self._loss_function(outputs, targets)

            loss.backward()

            self.optimizer.step()

            current_loss += loss.item()

            total += targets.size(0)
            _, predicted = outputs.max(0)
            correct += torch.sum(predicted == targets.data).float()

        self.__adjust_learning_rate(epoch)
        accu = 100.0 * correct / total
        train_loss = current_loss / len(trainloader)
        self._train_losses.append(train_loss)
        self._train_accuracies.append(accu.item())
        print("Train Loss: %.3f | Accuracy: %.3f" % (train_loss, accu))
        return accu.item()

    def prepareData(self, scale_data: bool = True):
        """
        returns trainloader and testloader
        """
        df = pd.read_csv(
            self.dataPath,
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
        trainDataset = MachineDataSet(x_train, y_train, scale_data)
        trainLoader = DataLoader(
            trainDataset, batch_size=20, shuffle=True, num_workers=1
        )

        testDataset = MachineDataSet(x_test, y_test, scale_data)
        testLoader = DataLoader(testDataset, batch_size=15, shuffle=True, num_workers=1)

        return trainLoader, testLoader

    def __adjust_learning_rate(self, epoch):
        lr = 0.0001

        if epoch > 10:
            lr = lr / 100000
        elif epoch > 8:
            lr = lr / 10000
        elif epoch > 6:
            lr = lr / 1000
        elif epoch > 4:
            lr = lr / 100
        elif epoch > 2:
            lr = lr / 10

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


if __name__ == "__main__":

    predictions = MachinePredictions(
        Path(
            r"C:\Users\stephan.schumacher\Documents\repos\dh-backend-api\data\logs\combined\machine_timings_combined.csv"
        ),
    )
    trainloader, testloader = predictions.prepareData()
    predictions.fit(20, trainloader, testloader)
