from pathlib import Path
from types import FunctionType
from typing import Any
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import time as tm
import pandas as pd
from datetime import datetime

from itertools import product


import os

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torchinfo import summary

from helper.MachineDataSet import MachineDataSet
from helper.model import Network
import helper.losses as losses


class MachinePredictions:
    def __init__(self, dataPath: Path, model: torch.nn.Module) -> None:
        torch.manual_seed(42)
        self.dataPath = dataPath
        self.model = model
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def fit(
        self,
        epochs: int,
        trainloader: DataLoader,
        testloader: DataLoader,
        learning_rate: float = 1e-5,
        optimizer: FunctionType = torch.optim.Adam,
        loss_function: FunctionType = nn.MSELoss,
        optim_args: dict = {},
        trial=Any,
    ):
        """
        - `epochs` : Number of epochs to train
        - `trainloader` : `torch.utils.data.DataLoader` Class with training data
        - `testloader` : `torch.utils.data.DataLoader` Class with validation data
        - `learning_rate` : learning rate to be applied
        - `optimizer` : optimizer function -> default is ADAM
        - `loss_function` : loss function -> default is MSE Loss

        > Returns: (train losses, train accuracies), (test losses, test accuracies)
        """
        self.epochs = epochs
        self._train_losses = []
        self._train_accuracies = []
        self._test_losses = []
        self._test_accuracies = []
        self._learning_rate = learning_rate
        self.optimizer = optimizer(
            self.model.parameters(),
            lr=learning_rate,
            **optim_args,
        )
        self._loss_function = loss_function()
        self.scheduler1 = ExponentialLR(self.optimizer, gamma=0.9)
        self.scheduler2 = MultiStepLR(
            self.optimizer, milestones=[int(epochs / 3), int(epochs * 2 / 3)]
        )
        timestamp = datetime.now().strftime("%m-%d-%Y_%H_%M_%S")
        self.run_name = f"{self.optimizer.__class__.__name__}_{self._loss_function.__class__.__name__}_{self.epochs}@{timestamp}"
        self.writer = SummaryWriter(f"./data/tensorboard/runs/{self.run_name}")
        data, labels = next(iter(trainloader))

        self.writer.add_graph(self.model, data.to(self.device))
        summary(self.model, input_size=data.shape)
        best_accu = 0
        for epoch in range(epochs):
            self.__train(epoch, trainloader)
            test_accu = self.__test(epoch, testloader)
            trial.report(test_accu, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        self.writer.close()

        return (self._train_losses, self._train_accuracies), (
            self._test_losses,
            self._test_accuracies,
        )

    def saveModel(self, epoch: int):

        torch.save(
            self.model.state_dict(),
            f"data/model/machineModel_{self.run_name}.model",
        )
        print("Checkpoint saved")

    def loadModel(self):
        torch.load()

    def predict(self):
        pass

    def __test(self, epoch: int, testloader: DataLoader):
        self.model.eval()
        current_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                # for data in tqdm(testloader, ascii=True):
                inputs, targets = data[0].to(self.device), data[1].to(self.device)
                inputs, targets = inputs.float(), targets.float()
                targets = targets.reshape((targets.shape[0], self.model.output_size))

                outputs = self.model(inputs)

                test_loss = self._loss_function(outputs, targets)
                current_loss += test_loss.item()

                _, predicted = torch.max(outputs, 0)

                total += targets.size(0)

                correct += torch.sum(predicted == targets.data).float()

        test_loss = current_loss / len(testloader)
        accu = correct / float(total)

        self._test_losses.append(test_loss)
        self._test_accuracies.append(accu.item())
        print("Test Loss: %.3f | Accuracy: %.3f" % (test_loss, accu))

        # adding scalars to tensorboard
        self.createTensorLogs("test", epoch, test_loss, correct, accu)
        return accu.item()

    def __train(self, epoch: int, trainloader: DataLoader):
        # print(f"\nStarting epoch {epoch+1} / {self.epochs} ")
        self.model.train()
        current_loss = 0.0
        correct = 0
        total = 0

        # for data in tqdm(trainloader, ascii=True):
        for data in trainloader:
            inputs, targets = data[0].to(self.device), data[1].to(self.device)

            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], self.model.output_size))

            outputs = self.model(inputs)

            loss = self._loss_function(outputs, targets)
            # loss = torch.autograd.Variable(loss, requires_grad=True)

            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())

            loss = loss + l2_lambda * l2_norm

            # self.writer.flush()
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            print(loss.item(), file=open("output.txt", "a"))
            current_loss += loss.item()

            total += targets.size(0)
            _, predicted = outputs.max(0)
            correct += torch.sum(predicted == targets.data).float()

        # self.__adjust_learning_rate(epoch)
        self.scheduler1.step()
        self.scheduler2.step()
        accu = correct / float(total)
        train_loss = current_loss / len(trainloader)
        self._train_losses.append(train_loss)
        self._train_accuracies.append(accu.item())
        print("Train Loss: %.3f | Accuracy: %.3f" % (train_loss, accu))

        # adding scalars to tensorboard
        self.createTensorLogs("train", epoch, loss, correct, accu)

        return accu.item()

    def createTensorLogs(
        self, mode: str, epoch: int, loss: any, correct: any, accu: any
    ):
        self.writer.add_scalar(f"Loss@{mode}", loss, epoch)
        self.writer.add_scalar(f"Number_Correct@{mode}", correct, epoch)
        self.writer.add_scalar(f"Accuracy@{mode}", accu, epoch)
        # for name, param in self.model.named_parameters():
        # self.writer.add_histogram(f"{name}@{mode}", param, epoch)
        for name, module in self.model.named_children():
            try:
                self.writer.add_histogram(f"{name}.bias@{mode}", module.bias, epoch)
                self.writer.add_histogram(f"{name}.weight@{mode}", module.weight, epoch)
                self.writer.add_histogram(
                    f"{name}.weight.grad@{mode}", module.weight.grad, epoch
                )
            except:
                continue

    def prepareData(
        self, scale_data: bool = True, batch_size: int = 15, shuffle: bool = True
    ):
        """
        returns trainloader and testloader
        """
        self.batch_size = batch_size
        df = pd.read_csv(
            self.dataPath,
            low_memory=False,
            encoding="unicode_escape",
        )
        df = df[
            [
                "placementsNeeded",
                "machine",
                "width",
                "height",
                "timeNeeded",
            ]
        ]

        df[["placementsNeeded", "width", "height", "timeNeeded"]] = df[
            ["placementsNeeded", "width", "height", "timeNeeded"]
        ].astype(float)

        df[["machine"]] = df[["machine"]].astype("category")

        df["machine"] = df["machine"].cat.codes.values

        df = df.dropna()
        x = df.drop(["timeNeeded"], axis=1)
        y = df[["timeNeeded"]]
        data = x.to_numpy(dtype=np.float32)
        labels = y.to_numpy(dtype=np.float32)

        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.25, random_state=42
        )
        trainDataset = MachineDataSet(x_train, y_train, scale_data)
        trainLoader = DataLoader(
            trainDataset, batch_size=batch_size, shuffle=shuffle, num_workers=1
        )

        testDataset = MachineDataSet(x_test, y_test, scale_data)
        testLoader = DataLoader(
            testDataset, batch_size=batch_size, shuffle=False, num_workers=1
        )

        return trainLoader, testLoader

    def plotLoss(self):
        plt.plot(self._train_losses, "-o")
        plt.plot(self._test_losses, "-o")
        plt.xlabel("epoch")
        plt.ylabel("losses")
        plt.legend(["Train", "Valid"])
        plt.title("Train vs Valid Losses")

        plt.show()

    def plotAccuracies(self):
        plt.plot(self._train_accuracies, "-o")
        plt.plot(self._test_accuracies, "-o")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend(["Train", "Valid"])
        plt.title("Train vs Valid Accuracies")

        plt.show()

    def plotTrainIndicators(self):
        plt.plot(self._train_accuracies, "-o")
        plt.plot(self._train_losses, "-o")
        plt.xlabel("epoch")
        plt.ylabel("index")
        plt.legend(["Accuracy", "Loss"])
        plt.title("Accuracy vs Loss in Training")
        plt.show()

    def __adjust_learning_rate(self, epoch):
        lr = self._learning_rate

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
    print(os.getcwd())
    exit()
    DATA_PATH = Path(os.getcwd() + os.path.normpath("/data/all/trainDataTogether.csv"))

    paramsRun = {
        "n_layers": 6,
        "learning_rate": 0.08245042333799399,
        "optimizer": "SGD",
        "loss_function": "FocalTverskyLoss",
        "batch_size": 68,
        "weight_decay": 0.001277860711937618,
        "dropout": 0.4419852282783314,
        "n_units_l0": 64,
        "n_units_l1": 10,
        "n_units_l2": 62,
        "n_units_l3": 51,
        "n_units_l4": 33,
        "n_units_l5": 21,
        "activation": "GELU",
    }

    import optuna

    EPOCHS = 5
    RUNS = {}

    def objective(trial):
        params = {
            "n_layers": trial.suggest_int("n_layers", 1, 5),
            "n_units_layers": [],
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 9e-2),
            "optimizer": trial.suggest_categorical("optimizer", ["SGD", "ASGD"]),
            "scale_data": trial.suggest_categorical("scale_data", [True, False]),
            "loss_function": trial.suggest_categorical(
                "loss_function",
                [
                    # "FocalTverskyLoss",
                    # "KLDivLoss",
                    # "ComboLoss",
                    "L1Loss",
                    # "HuberLoss",
                ],
            ),
            "activation": trial.suggest_categorical(
                "activation", ["ReLU", "Sigmoid", "ELU"]
            ),
            "batch_size": trial.suggest_int("batch_size", 45, 70),
            "weight_decay": trial.suggest_loguniform("weight_decay", 9e-5, 9e-2),
            "dampening": trial.suggest_loguniform("dampening", 1e-1, 7e-1),
            "momentum": trial.suggest_loguniform("momentum", 1e-1, 7e-1),
            "dropout": trial.suggest_loguniform("dropout", 0.2, 0.5),
        }

        for i in range(params["n_layers"]):
            params["n_units_layers"].append(
                trial.suggest_int("n_units_l{}".format(i), 4, 70)
            )

        loss, accuracy = test_model(params, trial, plotModel=False)

        return loss

    def test_model(params: dict, trial, plotModel: bool = False):
        print(params)
        try:
            model = Network(
                4, 1, params["n_layers"], params["n_units_layers"], p=params["dropout"]
            )
        except:
            params["n_units_layers"] = []
            for i in range(params["n_layers"]):
                params["n_units_layers"].append(params[f"n_units_l{i}"])

            model = Network(
                4,
                1,
                params["n_layers"],
                params["n_units_layers"],
                p=params["dropout"],
                activation=getattr(torch.nn, params["activation"]),
            )
        predictions = MachinePredictions(DATA_PATH, model=model)
        trainLoader, testLoader = predictions.prepareData(
            scale_data=params["scale_data"],
            batch_size=params["batch_size"],
            shuffle=True,
        )

        try:
            params["loss_function"] = getattr(nn, params["loss_function"])
        except:
            params["loss_function"] = getattr(losses, params["loss_function"])

        # optim_args = {"weight_decay": weight_decay}
        optim_args = {
            "weight_decay": params["weight_decay"],
            # "dampening": params["dampening"],
            # "momentum": params["momentum"],
        }
        train, test = predictions.fit(
            EPOCHS,
            trainLoader,
            testLoader,
            loss_function=params["loss_function"],
            optimizer=getattr(torch.optim, params["optimizer"]),
            optim_args=optim_args,
            learning_rate=params["learning_rate"],
            trial=trial,
        )

        if plotModel == False:
            return sum(test[0]) / EPOCHS, sum(test[1]) / EPOCHS
        else:
            RUNS[predictions.run_name] = {
                "train": train,
                "test": test,
            }

            fig, axs = plt.subplots(1, 4, figsize=(18, 6))
            fig.suptitle(
                "Training Accuracy | Testing Accuracy | Training Loss | Testing Loss"
            )
            for key in RUNS:
                axs[0].plot(RUNS[key]["train"][1], "-o", label=key)
                axs[1].plot(RUNS[key]["test"][1], "-o", label=key)
                axs[2].plot(RUNS[key]["train"][0], "-o", label=key)
                axs[3].plot(RUNS[key]["test"][0], "-o", label=key)
            axs[0].legend(loc="upper left")
            axs[1].legend(loc="upper left")
            axs[2].legend(loc="upper left")
            axs[3].legend(loc="upper left")
            plt.show()

    # test_model(paramsRun, plotModel=True)
    # exit()
    study = optuna.create_study(
        # directions=["minimize", "maximize"],
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
    )
    study.optimize(
        objective,
        n_trials=30,
        show_progress_bar=True,
        catch=(RuntimeError, RuntimeWarning, TypeError),
    )

    best_trial = study.best_trial
    optuna.visualization.matplotlib.plot_param_importances(
        study, target=lambda t: t.values[0]
    )
    plt.show()
    # for best_trial in best_trials:
    print("==== NEW TRIAL ====")
    print("==== NEW TRIAL ====", file=open("best_trials.txt", "a"))
    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value), file=open("best_trials.txt", "a"))
        print("{}: {}".format(key, value))

    test_model(best_trial.params, None, plotModel=True)

    """parameters2 = dict(
        lr=[0.0001, 0.00001],
        batch_size=[20, 15],
        shuffle=[True],
        scale=[True, False],
    )
    parameters = dict(
        lr=[15e-06],
        batch_size=[36],
        shuffle=[True],
        scale=[True],
        weight_decay=[0.4],
        dampening=[0.32],
        momentum=[0.3],
    )
    param_values = [v for v in parameters.values()]

    runs = {}

    for lr, batch_size, shuffle, scale, weight_decay, dampening, momentum in product(
        *param_values
    ):
        print(lr, batch_size, shuffle, scale, weight_decay, dampening, momentum)
        # print(lr, batch_size, shuffle, scale, file=open("output.txt", "a"))

        model = Network(4, [20, 40, 80, 160, 80, 40, 20], 1, p=0.5)
        predictions = MachinePredictions(global_string, model=model)
        trainLoader, testLoader = predictions.prepareData(
            scale_data=scale, batch_size=batch_size, shuffle=shuffle
        )

        # optim_args = {"weight_decay": weight_decay}
        optim_args = {
            "weight_decay": weight_decay,
            "dampening": dampening,
            "momentum": momentum,
        }
        data1, data2 = predictions.fit(
            5,
            trainLoader,
            testLoader,
            # loss_function=nn.CrossEntropyLoss,
            # loss_function=nn.L1Loss,
            # loss_function=nn.KLDivLoss,
            loss_function=losses.FocalTverskyLoss,
            # loss_function=nn.BCELoss,
            optimizer=torch.optim.SGD,
            optim_args=optim_args,
            learning_rate=lr,
        )
        # print(data1, data2, "\n", file=open("output.txt", "a"))
        runs[f"{lr, batch_size, shuffle, scale, weight_decay}"] = {
            "train": data1,
            "test": data2,
        }
        # predictions.plotTrainIndicators()
    # print(runs, file=open("output.txt", "a"))
    fig, axs = plt.subplots(1, 4, figsize=(18, 6))
    fig.suptitle("Training Accuracy | Testing Accuracy | Training Loss | Testing Loss")
    for key in runs:
        axs[0].plot(runs[key]["train"][1], "-o", label=key)
        axs[1].plot(runs[key]["test"][1], "-o", label=key)
        axs[2].plot(runs[key]["train"][0], "-o", label=key)
        axs[3].plot(runs[key]["test"][0], "-o", label=key)
    axs[0].legend(loc="upper left")
    axs[1].legend(loc="upper left")
    axs[2].legend(loc="upper left")
    axs[3].legend(loc="upper left")

    plt.show()"""
