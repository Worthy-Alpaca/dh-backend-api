import sys
import os
import math

PACKAGE_PARENT = "../.."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from schemas import DummyMachine

from pathlib import Path


class Machine:
    def __init__(self, dummyMachine: DummyMachine) -> None:
        """Class that represents a machine"""
        self.machineName = dummyMachine.machine
        self.cph = dummyMachine.cph
        self.nozHeads = dummyMachine.nozHeads
        self.SMD = dummyMachine.SMD
        cps = 3600 / dummyMachine.cph
        self.velocity = math.sqrt(180**2 + 180**2) / cps
        self.offsets = dummyMachine.offsets

    def getData(self) -> dict:
        """Returns the machine properties in a dict"""
        return {
            "machine": self.machineName,
            "cph": self.cph,
            "nozHeads": self.nozHeads,
            "SMD": self.SMD,
            "offsets": self.offsets,
        }


if __name__ == "__main__":
    from data.dataloader import DataLoader

    path = Path(os.getcwd() + os.path.normpath("/data/3011330"))
    dataloader = DataLoader(path)
    data = dataloader()
    for i in data[1]["Nozzle_No"].unique():
        print(i)
