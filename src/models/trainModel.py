import pickle
from pathlib import Path
from types import FunctionType
from typing import Any, Dict, Literal, Union
from sqlalchemy import true
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
from tqdm import tqdm
import joblib
import os

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torchinfo import summary

try:
    from helper.MachineDataSet import MachineDataSet
    from helper.model import Network
except:
    from src.models.helper.MachineDataSet import MachineDataSet
    from src.models.helper.model import Network

PATH = Path(os.getcwd() + os.path.normpath("/data/models"))


class TrainModel:
    def __init__(
        self, dataPath: Path = None, model: torch.nn.Module = None, deploy: bool = False
    ) -> None:
        """Class that initializes a Model training instance.

        Args:
            dataPath (Path): Path to a datasource or Modelsource if deploy is True
            model (torch.nn.Module): The model that is used for training. If deploy is true, pass an uninitialized model
            deplot (bool): If the model is in deploy mode. Defaults to False.

        Exception:
            KeyError: If dataPath or model = None while deploy = False
        """
        torch.manual_seed(42)

        if deploy == False and dataPath == None or deploy == False and model == None:
            raise KeyError("The dataPath or model cannot be None if deploy is False")
        if deploy:
            self.loadInternalStates(dataPath)
        else:
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
        validate: Literal["maximize", "minimize"] = "minimize",
    ) -> tuple[
        tuple[Union[Any, float], Union[torch.Tensor, float]],
        tuple[Union[Any, float], Union[torch.Tensor, float]],
    ]:
        """Train the Model for the specified number of epochs.

        Args:
            epochs (int): Number of epochs to train.
            trainLoader (DataLoader): Training Data.
            testLoader (DataLoader): Validation Data.
            learning_rate (float, optional): Learning Rate for optimizer. Defaults to 1e-5.
            optimizer (FunctionType, optional): Pytorch optimizer instance. Defaults to torch.optim.Adam.
            loss_function (FunctionType, optional): Pytorch Loss Function instance. Defaults to nn.MSELoss.
            optim_args (Dict, optional): Optional Arguments for Pytorch optimizer. See documentation for examples. Defaults to {}.
            trial (optuna.Trial, optional): Current Optuna Trial. Defaults to None.
            show_summary (bool, optional): Controlls if a Model summary is shown before training starts. Defaults to False.
            validate (Literal[&quot;maximize&quot;, &quot;minimize&quot;], optional): Controlls which Metric to return. Defaults to "minimize".

        Raises:
            optuna.exceptions.TrialPruned: Stops training when Optuna prunes a trial according to optuna Pruner instance.

        Returns:
            tuple[tuple[Any | float, torch.Tensor | float], tuple[Any | float, torch.Tensor | float]]: Can either be the Loss or the Mean Absolute Error
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
        # creating tensorboard integrationa
        self.writer = SummaryWriter(
            os.getcwd() + os.path.normpath(f"/data/tensorboard/{self.run_name}")
        )
        # adding Model overview graph to tensorboard
        data, labels = next(iter(trainLoader))
        self.writer.add_graph(self.model, data.to(self.device))
        if show_summary:
            summary(self.model, input_size=data.shape)

        # running training and testing loop
        mean_train_loss = 0.0
        mean_val_loss = 0.0
        mean_train_acc = 0.0
        mean_val_acc = 0.0
        for epoch in range(epochs):
            train_loss, train_acc = self.__train(trainLoader, epoch)
            val_loss, val_acc = self.__validate(testLoader, epoch)
            mean_train_loss += train_loss
            mean_val_loss += val_loss
            mean_train_acc += train_acc
            mean_val_acc += val_acc
            if val_loss == np.nan:
                self.writer.close()
                raise optuna.exceptions.TrialPruned()
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
        # saving the model
        if trial == None:
            self.saveInternalStates(PATH)
        # returning calculated values
        return (mean_train_loss / epochs, mean_train_acc / epochs), (
            mean_val_loss / epochs,
            mean_val_acc / epochs,
        )

    def __train(
        self, trainLoader: DataLoader, epoch: int
    ) -> tuple[Union[Any, float], Union[torch.Tensor, float]]:
        """Runs a training loop. Model is in training mode.

        Args:
            trainLoader (DataLoader): DataLoader that contains the training data.
            epoch (int): The current epoch.

        Returns:
            tuple[Any  float, torch.Tensor  float]: Returns the mean training Loss and MAE for all batches.
        """
        # setting model into training mode
        self.model.train()

        # assigning collection variables
        total_acc_train = 0
        total_loss_train = 0
        # looping over batches in Dataloader
        for train_input, train_target in tqdm(trainLoader):
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
            "Train Loss @ Epoch %i/%i : %.5f  MAE %.5f"
            % (epoch + 1, self.epochs, mean_loss_train, mean_acc_train)
        )
        return mean_loss_train, mean_acc_train

    def __validate(
        self, testLoader: DataLoader, epoch: int
    ) -> tuple[Union[Any, float], Union[torch.Tensor, float]]:
        """Runs a validation loop. Model is in validation mode.

        Args:
            testLoader (DataLoader): DataLoader that contains the validation data.
            epoch (int): The current epoch.

        Returns:
            tuple[Any  float, torch.Tensor  float]: Returns the mean validation Loss and MAE for all batches.
        """
        # setting model into evaluation mode
        self.model.eval()
        # assigning collection variables
        total_acc_val = 0
        total_loss_val = 0
        # turning off gradient calculations
        with torch.no_grad():
            # looping over batches in testLoader
            for val_input, val_target in tqdm(testLoader):
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
            "Test Loss @ Epoch %i/%i : %.5f  MAE %.5f"
            % (epoch + 1, self.epochs, mean_loss_val, mean_acc_val)
        )
        return mean_loss_val, mean_acc_val

    def __createTensorboardLogs(
        self, mode: Literal["training", "validation"], epoch: int, loss: Any, accu: Any
    ):
        """Create TensorBoard entries for Loss and Accuracy as well as weight and bias for all possible children.

        Args:
            mode (Literal[&#39;training&#39;, &#39;validation&#39;]): The current mode. Can either be training or validation.
            epoch (int): The current epoch.
            loss (Any): The current Loss.
            accu (Any): The current Accuracy.
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

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict an outcome using the trained Model.

        Args:
            data (np.ndarray): The Features for prediction.

        Returns:
            np.ndarray: The generated prediction.
        """
        data = data.reshape(1, -1)
        if self.scale_data:
            data = self.scaleX.transform(data)
        data = torch.from_numpy(data).to(self.device)
        prediction = self.model(data.float())
        prediction = prediction.cpu().detach().numpy()
        if self.scale_data:
            prediction = self.scaleData(prediction, inverse=True)

        return prediction

    def saveInternalStates(self, path: Path) -> None:
        """Save the model and scaler state to the specified path.

        Args:
            path (Path): Folder in which the model should be saved.
        """
        if not os.path.exists(path / self.run_name):
            os.makedirs(path / self.run_name)
        model_scripted = torch.jit.script(self.model)
        model_scripted.save(path / self.run_name / "modelState.pt")
        # with open(path / self.run_name / "modelState.p", "wb") as fp:
        # pickle.dump(self.model, fp, protocol=pickle.HIGHEST_PROTOCOL)
        joblib.dump(self.scaleX, path / self.run_name / "scaleStateX.gz")
        joblib.dump(self.scaleY, path / self.run_name / "scaleStateY.gz")

    def scaleData(
        self, data: np.ndarray, labels: np.ndarray = None, inverse: bool = False
    ) -> (Union[np.ndarray, tuple[np.ndarray, np.ndarray]]):
        """Scales input data. Also allows for inverse transformation.

        Args:
            data (np.ndarray): The input data.
            labels (np.ndarray): The label data.
            inverse (bool): If an inverse transformation should be done to the data input. Defaults to False.

        Returns:
            np.ndarray: The Scaled Data.
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
    ) -> tuple[DataLoader, DataLoader]:
        """Prepare the Data found in dataPath.

        Args:
            scale_data (bool, optional): If the found data should be scaled. Defaults to True.
            batch_size (int, optional): The batch size to be used. Defaults to 15.
            shuffle (bool, optional): If the found data should be shuffled. Defaults to True.

        Raises:
            FileNotFoundError: Raises this error if the given Datapath leads to an unreadable file.

        Returns:
            tuple[DataLoader, DataLoader]: Returns the trainLoader and the testLoader.
        """
        # adding batch size to class variables
        self.batch_size = batch_size
        self.scale_data = scale_data
        # loading dataset
        try:
            df = pd.read_csv(self.dataPath, low_memory=False, encoding="unicode_escape")
        except:
            raise FileNotFoundError(
                "The given dataPath does not lead to a readable file."
            )
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
    trainLoader, testLoader = trainModel.prepareData(scale_data=True)
    trainModel.fit(2, trainLoader, testLoader, show_summary=True)
    # data = np.array([308, 1, 306, 500])
    data = np.array([4, 0, 255, 224.678])
    pred = trainModel.predict(data)
    print(pred)
    data = np.array([600, 1, 225.6, 128.0])
    pred = trainModel.predict(data)
    print(pred)
