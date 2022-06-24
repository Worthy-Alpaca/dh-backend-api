from pathlib import Path
import numpy as np
import torch
import joblib
from sklearn.preprocessing import MinMaxScaler
import os


class DeplotmentException(Exception):
    pass


class DeployModel:
    def __init__(self, path: Path) -> None:
        """Initiate the deployment of the Neural Network.

        Args:
            path (Path): Path to the saved NN Parameters.

        Raises:
            DeplotmentException: If the Save Folder doesn't exist.
            DeplotmentException: If something in the Model Loading process went wrong. Will notify of the error.
        """
        if not os.path.exists(path):
            raise DeplotmentException(
                "The Model cannot be deployed from the given Path. Make sure that the Path is correct"
            )
        try:
            self.scaleX: MinMaxScaler = joblib.load(path / "scaleStateX.gz")
            self.scaleY: MinMaxScaler = joblib.load(path / "scaleStateY.gz")
            self.model: torch.nn.Module = torch.jit.load(path / "modelState.pt")
        except Exception as e:
            raise DeplotmentException(f"The Loading of the Model failed. {e}")

    def predict(self, data: np.ndarray) -> np.ndarray:
        data = data.reshape(1, -1)

        data = self.scaleX.transform(data)
        data = torch.from_numpy(data)
        prediction = self.model(data.float())
        prediction = prediction.cpu().detach().numpy()

        prediction = self.scaleY.inverse_transform(prediction)

        return prediction


if __name__ == "__main__":
    import sys

    if sys.stdin and sys.stdin.isatty():
        # running interactively
        print("running interactively")
    else:
        with open("output", "w") as f:
            f.write("running in the background!\n")
    path = Path(
        r"C:\Users\stephan.schumacher\Documents\repos\dh-backend-api\data\models\Adam_MSELoss-1@06-24-2022_08_41_24"
    )
    model = DeployModel(path)

    data = np.array([168, 0, 225.6, 128.0])
    pred = model.predict(data)
    print(pred)
