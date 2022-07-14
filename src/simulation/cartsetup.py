import random
import numpy as np
from pathlib import Path

try:
    from data.dataloader import DataLoader
except:
    from src.data.dataloader import DataLoader


class CartSetup:
    def __init__(
        self, data: tuple, randomInterruptMin: int = 0, randomInterruptMax: int = 30
    ):
        """Class which initializes the cartsetuptime calculations

        Args:
            data (tuple): The current data set.
            randomInterruptMin (int, optional): Minimum value for random interruption calculations. Defaults to 0.
            randomInterruptMax (int, optional): Maximum value for random interruption calculations. Defaults to 30.
        """
        components = data[1]
        feedcart = {}
        for i in components["FeedStyle"].unique():
            feedcart[i] = components.loc[components["FeedStyle"] == i]

        self.randomInterruptMin = randomInterruptMin
        self.randomInterruptMax = randomInterruptMax
        self.feedcart = {k: v for k, v in feedcart.items() if k == k}

    def __call__(self) -> dict[str, int]:
        """Starts the setuptime calculation process

        Returns:
            dict[str, int]: Key, Value pair for time and number of needed Carts.
        """
        cart = 0
        time = 0
        # print(f"Setup for this product in progess: {len(self.feedcart.keys())} Carts needed")
        for key in self.feedcart:
            cart = cart + 1

            # print(f"Setting up Cart {cart} with {len(self.feedcart[key])} components")
            complexity = len(self.feedcart[key]) / 36
            for i in range(len(self.feedcart[key])):
                time = (
                    60
                    + random.randint(self.randomInterruptMin, self.randomInterruptMax)
                    * complexity
                    + 9.8
                ) + time
            # print(f"Needed time: {time / 60} min")

        return {"time": time, "numCarts": len(self.feedcart.keys())}

    def desetup(self) -> float:
        """Calculations to simulate the derigging of feedercarts

        Returns:
            float: Time for derigging.
        """
        time = 0
        for i in range(self.NumComp):
            time = (48 + np.random.randint(0, 30) + 9.9) + time

        return time


if __name__ == "__main__":
    path = Path("/content/3160194")
    dataloader = DataLoader(path, separator=",")
    data = dataloader()
    CartSetup(data)()
