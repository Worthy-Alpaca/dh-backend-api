from pathlib import Path
from types import FunctionType
from typing import Any, Dict, Literal
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from datetime import datetime
import optuna
from torchmetrics.functional import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import os

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torchinfo import summary

from helper.MachineDataSet import MachineDataSet
from helper.model import Network


class TrainModel:
    def __init__(self, dataPath: Path, model: torch.nn.Module) -> None:
        torch.manual_seed(42)
        self.dataPath = dataPath
        self.model = model
        self.device = "cpu"
        self.scaleX = MinMaxScaler()
        self.scaleY = MinMaxScaler()
        if torch.cuda.is_available():
            self.device = "cuda:0"
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def fit(
        self,
        epochs: int,
        trainLoader: DataLoader,
        testLoader: DataLoader,
        learning_rate: float = 1e-5,
        optimizer: FunctionType = torch.optim.Adam,
        loss_function: FunctionType = nn.MSELoss,
        optim_args: Dict = {},
        trial: optuna.Trial = None,
        show_summary: bool = False,
        validate: Literal["maximize", "minimize"] = "maximize",
    ):
        """
        - `epochs` : Number of epochs to train
        - `trainloader` : `torch.utils.data.DataLoader` Class with training data
        - `testloader` : `torch.utils.data.DataLoader` Class with validation data
        - `learning_rate` : learning rate to be applied
        - `optimizer` : optimizer function -> default is ADAM
        - `loss_function` : loss function -> default is MSE Loss
        - `optim_args` : Arguments for optimizer
        - `trial` : Optuna Trial -> default is `None`
        - `validate` : Which aspect to tune -> maximise = Accuracy | minimize = Loss

        > Returns: (train loss, train accuracy), (test loss, test accuracy)
        """
        # Assigning empty lists for parameters
        self.epochs = epochs
        self._train_losses = []
        self._train_accuracies = []
        self._val_losses = []
        self._val_accuracies = []
        # creating optimizer and loss function for model
        self.optimizer = optimizer(
            self.model.parameters(), lr=learning_rate, **optim_args
        )
        self.loss_function = loss_function()
        # creating Learning Rate adjuster
        self.scheduler1 = ExponentialLR(self.optimizer, gamma=1e-3)
        self.scheduler2 = MultiStepLR(
            self.optimizer, milestones=[int(epochs / 3), int(epochs * 2 / 3)]
        )
        # creating timestamp and run name for tensorboard
        timestamp = datetime.now().strftime("%m-%d-%Y_%H_%M_%S")
        self.run_name = f"{self.optimizer.__class__.__name__}_{self.loss_function.__class__.__name__}-{self.epochs}@{timestamp}"
        # creating tensorboard integration
        self.writer = SummaryWriter(os.getcwd() + os.path.normpath("/data/tensorboard"))
        # adding Model overview graph to tensorboard
        data, labels = next(iter(trainLoader))
        self.writer.add_graph(self.model, data.to(self.device))
        if show_summary:
            summary(self.model, input_size=data.shape)

        # running training and testing loop
        for epoch in range(epochs):
            train_loss, train_acc = self.__train(trainLoader, epoch)
            val_loss, val_acc = self.__validate(testLoader, epoch)
            # logic for trial pruning
            if trial == None:
                continue
            if validate == "maximize":
                trial.report(val_acc, epoch)
            else:
                trial.report(val_loss, epoch)
            if trial.should_prune():
                self.writer.close()
                raise optuna.exceptions.TrialPruned()
        # closing tensoboard writer
        self.writer.close()
        # returning calculated values
        return (train_loss, train_acc), (val_loss, val_acc)

    def __train(self, trainLoader: DataLoader, epoch: int):
        """
        > returns: loss, accuracy
        """
        # setting model into training mode
        self.model.train()

        # assigning collection variables
        total_acc_train = 0
        total_loss_train = 0
        # looping over batches in Dataloader
        for train_input, train_target in trainLoader:
            # assigning input and target to device, casting to float
            train_input = train_input.to(self.device).float()
            train_target = train_target.to(self.device).float()

            # clear the gradients
            self.optimizer.zero_grad()

            # running input through model
            output = self.model(train_input)

            # calculating the batch loss
            batch_loss = self.loss_function(output, train_target)
            # adding batch loss to collection
            total_loss_train += batch_loss.item()
            # calculating the batch accuracy
            # acc = (output.argmax(dim=1) == train_target).sum().float().item()
            # acc = acc / train_target.size(0)
            acc = mean_absolute_error(output, train_target)
            # adding batch accuracy to collection
            total_acc_train += acc

            # preparing model and loss for next iteration
            batch_loss.backward()
            self.optimizer.step()

        # calculating mean loss and accuracy to class collectors
        mean_loss_train = total_loss_train / float(len(trainLoader))
        mean_acc_train = total_acc_train / float(len(trainLoader))
        # adding mean loss and accuracy to class collectors
        self._train_accuracies.append(mean_acc_train)
        self._train_losses.append(mean_loss_train)
        # adjusting learning rate
        self.scheduler1.step()
        self.scheduler2.step()
        # creating TensorBoard entries for training
        self.__createTensorboardLogs("training", epoch, mean_loss_train, mean_acc_train)
        # returning loss and accuracy
        print(
            "Train Loss @ Epoch %i/%i : %.5f | MAE %.5f"
            % (epoch + 1, self.epochs, mean_loss_train, mean_acc_train)
        )
        return mean_loss_train, mean_acc_train

    def __validate(self, testLoader: DataLoader, epoch: int):
        """
        > returns: loss, accuracy
        """
        # setting model into evaluation mode
        self.model.eval()
        # assigning collection variables
        total_acc_val = 0
        total_loss_val = 0
        # turning off gradient calculations
        with torch.no_grad():
            # looping over batches in testLoader
            for val_input, val_target in testLoader:
                # assigning input and target to device, casting to float
                val_target = val_target.to(self.device).float()
                val_input = val_input.to(self.device).float()

                # running input through model
                output = self.model(val_input)

                # calculating batch loss
                batch_loss = self.loss_function(output, val_target)
                # adding batch loss to collection
                total_loss_val += batch_loss.item()

                # calculating batch accuracy
                acc = mean_absolute_error(output, val_target)
                # acc = (output.argmax(dim=1) == val_target).sum().item()
                # acc = acc / val_target.size(0)
                # adding batch accuracy to collection
                total_acc_val += acc

        # calculating mean loss and accuracy to class collectors
        mean_loss_val = total_loss_val / float(len(testLoader))
        mean_acc_val = total_acc_val / float(len(testLoader))
        # adding mean loss and accuracy to class collectors
        self._val_accuracies.append(mean_acc_val)
        self._val_losses.append(mean_loss_val)
        # adding TensorBoard entries for validation
        self.__createTensorboardLogs("validation", epoch, mean_loss_val, mean_acc_val)
        # returning loss and accuracy
        print(
            "Test Loss @ Epoch %i/%i : %.5f | MAE %.5f"
            % (epoch + 1, self.epochs, mean_loss_val, mean_acc_val)
        )
        return mean_loss_val, mean_acc_val

    def __createTensorboardLogs(self, mode: str, epoch: int, loss: Any, accu: Any):
        """
        Create TensorBoard entries for Loss and Accuracy as well as weight and bias for all possible children
        """
        self.writer.add_scalar(f"Loss@{mode}", loss, epoch)
        self.writer.add_scalar(f"Accuracy@{mode}", accu, epoch)
        for name, module in self.model.named_children():
            try:
                self.writer.add_histogram(f"{name}.bias@{mode}", module.bias, epoch)
                self.writer.add_histogram(f"{name}.weight@{mode}", module.weight, epoch)
                self.writer.add_histogram(
                    f"{name}.weight.grad@{mode}", module.weight.grad, epoch
                )
            except:
                continue

    def predict(self, data: np.ndarray):
        data = data.reshape(1, -1)
        if self.scale_data:
            data = self.scaleX.transform(data)
        data = torch.from_numpy(data)
        prediction = self.model(data.float())
        prediction = prediction.detach().numpy()
        if self.scale_data:
            prediction = self.scaleData(prediction, inverse=True)

        return prediction

    def scaleData(self, data, labels=None, inverse=False):
        """
        Scales input data. Also allows for inverse transformation
        """
        if inverse:
            return self.scaleY.inverse_transform(data)
        else:
            if type(labels) == np.ndarray:
                self.scaleY.fit(labels)
                self.scaleX.fit(data)
                return self.scaleY.transform(labels), self.scaleX.transform(data)
            else:
                self.scaleX.fit(data)
                return self.scaleX.transform(data)

    def prepareData(
        self, scale_data: bool = True, batch_size: int = 15, shuffle: bool = True
    ):
        """
        Prepare the Data from dataPath
        - `scale_data` : Wether to scale the data
        - `batch_size` : Size of batches
        - `shuffle` : Wether to shuffle the data

        > returns: trainLoader, testLoader
        """
        # adding batch size to class variables
        self.batch_size = batch_size
        self.scale_data = scale_data
        # loading dataset
        df = pd.read_csv(self.dataPath, low_memory=False, encoding="unicode_escape")
        # pulling only the needed features
        df = df[
            [
                "placementsNeeded",
                "machine",
                "width",
                "height",
                "timeNeeded",
            ]
        ]
        # casting values to float
        df[["placementsNeeded", "width", "height", "timeNeeded"]] = df[
            ["placementsNeeded", "width", "height", "timeNeeded"]
        ].astype(float)
        # encoding machine
        df[["machine"]] = df[["machine"]].astype("category")
        df["machine"] = df["machine"].cat.codes.values
        # dropping NaN values
        df = df.dropna()
        # splitting data in features and labels
        x = df.drop(["timeNeeded"], axis=1)
        y = df[["timeNeeded"]]
        # converting to numpy arrays
        data = x.to_numpy(dtype=np.float32)
        labels = y.to_numpy(dtype=np.float32)
        # scaling Data
        if scale_data:
            labels, data = self.scaleData(data, labels)
        # splitting the data into training and validation sets
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.25, random_state=42
        )
        # converting to torch Datasets and adding to Dataloader
        trainDataset = MachineDataSet(x_train, y_train)
        trainLoader = DataLoader(
            trainDataset, batch_size=batch_size, shuffle=shuffle, num_workers=1
        )
        testDataset = MachineDataSet(x_test, y_test)
        testLoader = DataLoader(
            testDataset, batch_size=batch_size, shuffle=False, num_workers=1
        )

        # returning trainLoader and testLoader
        return trainLoader, testLoader


if __name__ == "__main__":
    DATA_PATH = Path(os.getcwd() + os.path.normpath("/data/all/trainDataTogether.csv"))
    model = Network(4, 1, 3, [16, 32, 16])

    trainModel = TrainModel(DATA_PATH, model)
    trainLoader, testLoader = trainModel.prepareData()
    trainModel.fit(2, trainLoader, testLoader, show_summary=True)
    data = np.array([308, 1, 306, 500])
    pred = trainModel.predict(data)
    print(pred)
