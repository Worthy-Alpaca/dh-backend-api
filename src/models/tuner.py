from pathlib import Path
from typing import Literal

import torch
import optuna
from trainModel import TrainModel
from helper.model import Network
import helper.losses as losses
import torch.nn as nn


class Tuner:
    def __init__(
        self,
        dataPath: Path,
        epochs: int,
        direction: Literal["minimize", "maximize"] = "maximize",
        sampler: optuna.samplers = optuna.samplers.TPESampler,
        pruner: optuna.pruners = optuna.pruners.HyperbandPruner,
    ) -> None:
        # assigning variables
        self.dataPath = dataPath
        self.epochs = epochs
        self.direction = direction
        # create study
        self.study = optuna.create_study(
            direction=direction, sampler=sampler(), pruner=pruner()
        )

    def optimize(
        self,
        n_trials: int,
    ):

        self.study.optimize(
            self.__objective,
            n_trials=n_trials,
            catch=(RuntimeError, RuntimeWarning, TypeError),
        )

        return self.study.best_trial

    def __objective(self, trial):
        params = {
            "n_layers": trial.suggest_int("n_layers", 1, 5),
            "n_units_layers": [],
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 9e-2),
            "optimizer": trial.suggest_categorical(
                "optimizer", ["SGD", "ASGD", "Adam"]
            ),
            "scale_data": trial.suggest_categorical("scale_data", [True]),
            "loss_function": trial.suggest_categorical(
                "loss_function",
                [
                    # "FocalTverskyLoss",
                    # "KLDivLoss",
                    # "ComboLoss",
                    "L1Loss",
                    "MSELoss",
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

        loss, accuracy = self.tuneModel(params, trial)

        if self.direction == "maximize":
            return accuracy
        elif self.direction == "minimize":
            return loss
        else:
            raise TypeError("Unknown direction specified")

    def tuneModel(self, params: dict, trial):
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
        trainModel = TrainModel(self.dataPath, model)
        trainLoader, testLoader = trainModel.prepareData(
            scale_data=params["scale_data"],
            batch_size=params["batch_size"],
            shuffle=True,
        )

        try:
            params["loss_function"] = getattr(nn, params["loss_function"])
        except:
            params["loss_function"] = getattr(losses, params["loss_function"])

        optim_args = {"weight_decay": params["weight_decay"]}

        train, test = trainModel.fit(
            self.epochs,
            trainLoader,
            testLoader,
            loss_function=params["loss_function"],
            optimizer=getattr(torch.optim, params["optimizer"]),
            optim_args=optim_args,
            learning_rate=params["learning_rate"],
            trial=trial,
            validate=self.direction,
        )

        return test[0], test[1]


if __name__ == "__main__":
    import os

    DATA_PATH = Path(os.getcwd() + os.path.normpath("/data/all/trainDataTogether.csv"))

    tuner = Tuner(DATA_PATH, epochs=20, direction="minimize")
    best_trial = tuner.optimize(n_trials=30)
    tuner.tuneModel(best_trial.params, None)
