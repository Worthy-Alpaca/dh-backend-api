from pathlib import Path
from os.path import exists
from typing import Any, Literal, Union
import numpy as np

import torch
import optuna

try:
    from trainModel import TrainModel
    from helper.model import Network
    import helper.losses as losses
except:
    from src.models.trainModel import TrainModel
    from src.models.helper.model import Network
    import src.models.helper.losses as losses
import torch.nn as nn
import os

import pickle

PATH = Path(os.getcwd() + os.path.normpath("/data/models"))


class Tuner:
    def __init__(
        self,
        dataPath: Path,
        epochs: int,
        direction: Literal["minimize", "maximize"] = "maximize",
        sampler: optuna.samplers = optuna.samplers.TPESampler,
        pruner: optuna.pruners = optuna.pruners.HyperbandPruner,
    ) -> optuna.study:
        """Class to initiate a Model tuning session.

        Args:
            dataPath (Path): Path to a data source. May change to DB connection
            epochs (int): Number of epochs in each Trial run
            direction (Literal[&quot;minimize&quot;, &quot;maximize&quot;], optional): Direction to optimize. &quot;minimize&quot; optimizes Loss,  &quot;maximize&quot; optimizes Accuracy. Defaults to "maximize".
            sampler (optuna.samplers, optional): Optuna Sampler Algorythm to use. Defaults to optuna.samplers.TPESampler.
            pruner (optuna.pruners, optional): Optuna Pruner Algorythm to use. Defaults to optuna.pruners.HyperbandPruner.

        Returns:
            optuna.study: Optuna Study like session.
        """
        # assigning variables
        self.dataPath = dataPath
        self.epochs = epochs
        self.direction = direction
        # create study
        self.study = optuna.create_study(
            direction=direction,
            sampler=sampler(),
            pruner=pruner(),
            storage="sqlite:///50Studies_p4.db",
        )

    def optimize(
        self,
        n_trials: int,
    ) -> optuna.trial.FrozenTrial:
        """Starts the optimization process.

        Args:
            n_trials (int): Number of trials to run.

        Returns:
            optuna.trial.FrozenTrial: The best trial according to optimizer.
        """
        print(
            f"Starting optimizer session with {n_trials} Trials and {self.epochs} Epochs"
        )
        self.study.optimize(
            self.__objective,
            n_trials=n_trials,
            catch=(RuntimeError, RuntimeWarning, TypeError, ValueError),
            callbacks=[self.__logging_callback],
        )

        return self.study.best_trial

    def __objective(self, trial: optuna.trial) -> float:
        """Generate the study objectives.

        Args:
            trial (optuna.trial): The current trial.

        Raises:
            TypeError: Unknown Direction. Shouldn't occur. Sanity check.

        Returns:
            float: The current objective. Either accuracy or loss
        """
        params = {
            "n_layers": trial.suggest_int("n_layers", 1, 5),
            "n_units_layers": [],
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 9e-3),
            "optimizer": trial.suggest_categorical(
                "optimizer",
                ["Adamax"],  # "ASGD", "Adam", "Adamax"]
            ),
            "scale_data": trial.suggest_categorical("scale_data", [True]),
            "loss_function": trial.suggest_categorical(
                "loss_function",
                [
                    # "FocalTverskyLoss",
                    # "TverskyLoss",
                    # "L1Loss",
                    # "MSELoss",
                    "HuberLoss",
                ],
            ),
            "activation": trial.suggest_categorical(
                "activation", ["Sigmoid"]  # ["ReLU", "Sigmoid", "ELU"]
            ),
            "batch_size": trial.suggest_int("batch_size", 50, 70),
            "weight_decay": trial.suggest_loguniform("weight_decay", 9e-5, 9e-2),
            "dampening": trial.suggest_loguniform("dampening", 1e-1, 7e-1),
            "momentum": trial.suggest_loguniform("momentum", 2e-1, 4e-1),
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

    def tuneModel(
        self, params: dict, trial: optuna.trial = None, saveState: bool = False
    ) -> tuple[Union[torch.tensor, float], Union[torch.Tensor, float]]:
        """Tunes the model accoring to objective parameters.

        Args:
            params (dict): Paramters generated by Objective.
            trial (optuna.trial): Current optuna trial. Defaults to None.
            saveState (bool): If the parameters should be saved to file. Defaults to False.

        Returns:
            tuple[torch.tensor  float, torch.Tensor  float]: Mean Loss and Accuracy for validation.
        """
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
        self.trainModel = TrainModel(self.dataPath, model)
        trainLoader, testLoader = self.trainModel.prepareData(
            scale_data=params["scale_data"],
            batch_size=params["batch_size"],
            shuffle=True,
        )

        try:
            params["loss_function"] = getattr(nn, params["loss_function"])
        except:
            params["loss_function"] = getattr(losses, params["loss_function"])

        optim_args = {
            "weight_decay": params["weight_decay"],
            "eps": params["momentum"],
            # "dampening": params["momentum"],
        }

        try:
            train, test = self.trainModel.fit(
                10,
                trainLoader,
                testLoader,
                loss_function=params["loss_function"],
                optimizer=getattr(torch.optim, params["optimizer"]),
                optim_args=optim_args,
                learning_rate=params["learning_rate"],
                trial=trial,
                validate=self.direction,
            )
        except:
            train, test = self.trainModel.fit(
                10,
                trainLoader,
                testLoader,
                loss_function=params["loss_function"],
                optimizer=getattr(torch.optim, params["optimizer"]),
                learning_rate=params["learning_rate"],
                trial=trial,
                validate=self.direction,
            )

        if saveState:
            self.saveBestTrial(params)

        return test[0], test[1]

    def saveBestTrial(self, params: dict, path: Path = PATH):
        """Save the state and Parameters of the best trial.

        Args:
            params (optuna.trial.FrozenTrial): The best trial as determined by optuna
            path (Path, optional): Path to saving location. Defaults to PATH.
        """
        pathway = Path(os.getcwd() + "/data/models/bestTrial5")
        if not exists(pathway):
            os.mkdir(pathway)
        self.trainModel.saveInternalStates(pathway)

        if not exists(path / "updatedModelUnscaled"):
            os.mkdir(path / "updatedModelUnscaled")
        with open(path / "updatedModelUnscaled" / "modelParameters.p", "wb") as fp:
            pickle.dump(params, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def saveStudy(self, path: Path):
        """Saves the study to the provided path.

        Args:
            path (Path): Path to saving location.
        """
        if not exists(path):
            os.mkdir(path)
        with open(path / f"{self.study.study_name}.p", "wb") as fp:
            print(f"Saving Study with Name: {self.study.study_name}")
            pickle.dump(self.study, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def loadStudy(self, path: Path) -> optuna.study.Study:
        """Load a study from the provided location.

        Args:
            path (Path): The location of the study.

        Returns:
            optuna.study.Study: The loaded study.
        """
        with open(path, "rb") as file:
            data = pickle.load(file)
        return data

    def __logging_callback(
        self, study: optuna.study.Study, frozen_trial: optuna.trial.FrozenTrial
    ):
        """Internal method used to save the current best trial.

        Args:
            study (optuna.study.Study): The current study.
            frozen_trial (optuna.trial.FrozenTrial): The current best trial.
        """
        previous_best_value = study.user_attrs.get("previous_best_value", None)
        if previous_best_value != study.best_value:
            study.set_user_attr("previous_best_value", study.best_value)
            print(
                "Trial {} finished with best value: {} and parameters: {}. ".format(
                    frozen_trial.number,
                    frozen_trial.value,
                    frozen_trial.params,
                )
            )
            self.saveBestTrial(frozen_trial.params)


if __name__ == "__main__":
    import os

    DATA_PATH = Path(os.getcwd() + os.path.normpath("/data/all/trainDataTogether.csv"))
    STUDY_PATH = Path(
        os.getcwd() + os.path.normpath("/data/model/studies/5TrialsScaled")
    )
    STUDY_LOAD = Path(
        os.getcwd() + os.path.normpath(r"\data\model\studies\50trialsScaled.p")
    )

    tuner = Tuner(
        DATA_PATH, epochs=1, direction="minimize", sampler=optuna.samplers.CmaEsSampler
    )
    best_trial = tuner.optimize(n_trials=50)
    tuner.saveStudy(STUDY_PATH)
    # study = tuner.loadStudy(STUDY_LOAD)

    params = {
        "n_layers": 3,
        "epochs": 2,
        "learning_rate": 0.2505950956626377,
        "optimizer": "Adamax",
        "scale_data": True,
        "loss_function": "MSELoss",
        "activation": "GELU",
        "batch_size": 65,
        "weight_decay": 0.00035351893312748976,
        "dampening": 0.1721094552252759,
        "momentum": 0.21608929927906143,
        "dropout": 0.2535510202298927,
        "n_units_l0": 27,
        "n_units_l1": 20,
        "n_units_l2": 14,
    }
    # tuner.tuneModel(best_trial.params, None, True)
