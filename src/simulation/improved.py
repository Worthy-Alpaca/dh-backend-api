import math
import sys
import os
import itertools
import time as tm
from typing import Union
import pandas as pd
from collections import deque
from pathlib import Path
import concurrent.futures

PACKAGE_PARENT = "../"
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from simulation.machine import Machine

global XMOD
XMOD = 1
RANDOM_PROBLEMS = (0, 0)
DROPOFF = 0.1
PICKUP = 0.1


class ManufacturingImproved:
    def __init__(self, data: tuple, machine: Machine) -> object:
        self.machine = machine
        self.components: pd.DataFrame = data[1]
        self.data: pd.DataFrame = data[0]
        self.offsets: pd.DataFrame = data[2]

        if machine.offsets is not None:
            self.OFFSET_X = machine.offsets["pcb"][0]
            self.OFFSET_Y = machine.offsets["pcb"][1]
            self.CHECKPOINT = (
                machine.offsets["checkpoint"][0],
                machine.offsets["checkpoint"][1],
            )
            self.feedercarts = {}
            for machine in machine.offsets["feedercarts"]:
                for key in machine:
                    self.feedercarts[key] = machine[key]

    def __call__(
        self, mulitPickOption: bool = True, plotPCB: bool = False
    ) -> Union[float, dict]:
        self.multiPickOption = mulitPickOption
        time = int
        plotX = list
        plotY = list
        start_time = tm.time()
        results = list
        with concurrent.futures.ThreadPoolExecutor() as executer:
            for i in self.offsets:
                results.append(executer.submit(self.calcTime, i))

        for f in results:
            iter_data = f.result()
            time = time + iter_data["time"]
            plotX.append(iter_data["plot_x"])
            plotY.append(iter_data["plot_y"])

        print("---- Runtime: %s seconds ----" % (tm.time() - start_time))
        if plotPCB == True:
            return {
                "time": time,
                "plot_x": list(itertools.chain.from_iterable(plotX)),
                "plot_y": list(itertools.chain.from_iterable(plotY)),
            }
        return time

    def coating(self):
        """simulates the time for coating a PCB"""
        velocity = 20  # mm/s
        offset = max(self.offsets)
        highestPoint = self.data["Y"].max() + offset[1]
        return math.sqrt(0**2 + highestPoint**2) / velocity

    def __calcVector(self, vectorA: tuple, vectorB: tuple, velocity: float):
        """
        Calculates the time between two given vectors with a given velocity
        - vectorA: Startpoint Location vector
        - vectorB: Endpoint Location vector
        - velocity: Travel velocity given in mm/s
        """
        vector_AB = (
            (float(vectorB[0]) - float(vectorA[0])),
            (float(vectorB[1]) - float(vectorA[1])),
        )
        path_length = math.sqrt(vector_AB[0] ** 2 + vector_AB[1] ** 2)
        return path_length / velocity

    def calcTime(self, offset_row: tuple, multiPickOption: bool = True):
        """
        Calculates the assembly time for a given offset
        """

        TIME = int
        plotting_x = list
        plotting_y = list
        mulitPick = deque()

        for index, row in self.data.iterrows():

            lookupTable = self.components[self.components["index"].str.match(row.Code)]

            cart_coordinates = self.feedercarts[lookupTable.FeedStyle.max()]

            location_vectorA = (
                int(cart_coordinates[0]) + (lookupTable.ST_No.max() * 20),
                int(cart_coordinates[1]),
            )

            location_vectorB = (
                (row.X + self.OFFSET_X + offset_row[0]),
                (row.Y + self.OFFSET_Y + offset_row[1]),
            )

            plot_coordinates = ((row.X + offset_row[0]), (row.Y + offset_row[1]))

            velocity = lookupTable.mean_acceleration.max()

            DROPOFF = (lookupTable.Dropff.max() / 1000) * 0.1

            if multiPickOption == False:
                path_length = self.__calcVector(
                    location_vectorA, self.CHECKPOINT, velocity
                )
                checkpoint = self.__calcVector(
                    self.CHECKPOINT, location_vectorB, velocity
                )
                TIME = path_length + checkpoint + DROPOFF + TIME

                next_index = index + 1
