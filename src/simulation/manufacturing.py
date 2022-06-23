import math
import sys
import os
import itertools
from collections import deque
from pathlib import Path
import concurrent.futures
import time as tm
from typing import Union
import pandas as pd
from tqdm import tqdm

PACKAGE_PARENT = "../"
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from simulation.machine import Machine

global XMOD
XMOD = 1
RANDOM_PROBLEMS = (0, 0)
# DROPOFF = 0.1
PICKUP = 0.1


class Manufacturing:
    def __init__(self, data: tuple, machine: Machine):
        """Simulates the manufacturing process in SMD machines M10 and M20.

        Args:
            data (tuple): The current Dataset.
            machine (Machine): The current Machine.
        """
        # assign machine property
        self.machine = machine
        self.components = data[1]
        self.data: pd.DataFrame = data[0]
        self.offsets = data[2]
        self.heads = {}
        # add offsets in case of SMD machine
        if machine.offsets is not None:
            # offset for the PCB arrival point
            self.OFFSET_X = machine.offsets["pcb"][0]
            self.OFFSET_Y = machine.offsets["pcb"][1]
            # coordinates for the visual checkpoint
            self.CHECKPOINT = (
                machine.offsets["checkpoint"][0],
                machine.offsets["checkpoint"][1],
            )
            # coordinates for the tool staging area
            self.toolPickup = (machine.offsets["tools"][0], machine.offsets["tools"][1])
            self.feedercarts = {}
            # offsets for feedercart positions
            for machine in machine.offsets["feedercarts"]:
                for key in machine:
                    self.feedercarts[key] = machine[key]

    def __calcVector(self, vectorA: tuple, vectorB: tuple, velocity: float) -> float:
        """Calculates the pathlength between two given vectors

        Args:
            vectorA (tuple): Start locationvector.
            vectorB (tuple): End locationvector.
            velocity (float): The current velocity to use.

        Returns:
            float: The calculated travel time.
        """
        # calculate the connection vector
        vector_AB = (
            (float(vectorB[0]) - float(vectorA[0])),
            (float(vectorB[1]) - float(vectorA[1])),
        )
        # calculate the length of the connection vector
        path_length = math.sqrt(vector_AB[0] ** 2 + vector_AB[1] ** 2)
        # calculate the travel time and return it
        return path_length / velocity

    def calcTime(self, offset_row: tuple, useIdealState: bool):
        """Calculates the time for a given offset.

        Args:
            offset_row (tuple): The current offset.
            useIdealState (bool): If the ideal state of the machine should be used. Impacts the veloticy used.
        """

        def isNan(string):
            return string != string

        # assigning properties for later use
        TIME = 0
        self.plotting_x = []
        self.plotting_y = []
        multiPick = deque()
        blocks = []
        # dividing the placement data into blocks
        blockId = (self.data["Task"] == "Start Block").cumsum()
        for n, g in self.data.groupby(blockId):
            g = g.dropna()
            blocks.append(g)
        # Loop over all blocks in the placementdata
        for block in blocks:
            # calculate the
            for index, row in tqdm(block.iterrows()):
                # check for NaN values and continue if found
                if isNan(row.Code):
                    continue
                if row.Task == "Start Block":
                    continue
                # calculating the path, adding offset for coordinate transformation
                lookUp = self.components["index"].str.match(row.Code)
                lookupTable = self.components[lookUp]
                cart_coordinates = self.feedercarts[lookupTable.FeedStyle.max()]
                # creating location vectors
                location_vector_A = (
                    int(cart_coordinates[0]) + (lookupTable.ST_No.max() * 10),
                    int(cart_coordinates[1]),
                )
                location_vector_B = (
                    (row.X + self.OFFSET_X + offset_row[0]),
                    (row.Y + self.OFFSET_Y + offset_row[1]),
                )
                plot_coordinates = ((row.X + offset_row[0]), (row.Y + offset_row[1]))

                currentHead = lookupTable.Nozzle_No.max()
                if currentHead not in self.heads.keys():
                    if len(self.heads) < self.machine.nozHeads:
                        self.heads[currentHead] = tm.time()
                    else:
                        removeKey = min(self.heads, key=self.heads.get)
                        self.heads.pop(removeKey, None)
                        removeKey = min(self.heads, key=self.heads.get)
                        self.heads.pop(removeKey, None)
                        # velocity used is Vmean from both machines
                        Vmean = 1483.635
                        TIME = (
                            self.__calcVector(location_vector_A, self.toolPickup, Vmean)
                            + self.__calcVector(
                                self.toolPickup, location_vector_A, Vmean
                            )
                            + 2
                            + TIME
                        )
                        self.heads[currentHead] = tm.time()
                else:
                    self.heads[currentHead] = tm.time()

                # calculating velocity based on component or idealState
                velocity = 0
                if useIdealState == False:
                    velocity = lookupTable.mean_acceleration.max()
                else:
                    if self.machine.machineName.lower() == "m20":
                        velocity = 1345.87
                    elif self.machine.machineName.lower() == "m10":
                        velocity = 1621.4

                DROPOFF = (lookupTable.Dropoff.max() / 1000) * 0.1

                if self.multiPickOption == True:
                    # picking components with multiple heads at once
                    # path changes from "Pickup -> Component" to "Pickup1 -> Pickup2 -> Pickup3 -> Component3 -> Component2 -> Component1"
                    if row.Task == "Multiple Pickup":
                        # append component vector to queue
                        multiPick.append(location_vector_B)

                        # calculate path/time to next pickup
                        next_index = index + 1
                        while next_index not in block.index:
                            if next_index >= block.index.max():
                                next_index = next_index - 1
                                break
                            next_index = next_index + 1
                        nextLookUpTable = self.components[
                            self.components["index"].str.match(
                                block.loc[next_index, "Code"]
                            )
                        ]
                        nextcart_coordinates = self.feedercarts[
                            nextLookUpTable.FeedStyle.max()
                        ]
                        next_pickup = (
                            int(nextcart_coordinates[0])
                            + (nextLookUpTable.ST_No.max() * 10),
                            int(nextcart_coordinates[1]),
                        )
                        TIME = (
                            self.__calcVector(location_vector_A, next_pickup, velocity)
                            + TIME
                            + PICKUP
                        )

                    elif row.Task == "End Multiple Pickup":
                        # calculate the path to the current component
                        loc_vector_A = (
                            lookupTable.Pickup_X.max(),
                            lookupTable.Pickup_Y.max(),
                        )
                        loc_vector_B = (
                            (row.X + self.OFFSET_X + offset_row[0]),
                            (row.Y + self.OFFSET_Y + offset_row[1]),
                        )
                        checkpoint = self.__calcVector(
                            loc_vector_A, self.CHECKPOINT, velocity
                        )
                        path = self.__calcVector(
                            self.CHECKPOINT, loc_vector_B, velocity
                        )
                        TIME = path + TIME + DROPOFF + checkpoint

                        # set the current component vector as the current postition
                        current_pos = location_vector_B

                        # rotate the queue
                        multiPick.rotate()

                        # loop over queue
                        for i in multiPick:
                            # calculate path and time between components
                            multiPath = self.__calcVector(current_pos, i, velocity)
                            TIME = (multiPath) + TIME + DROPOFF
                            current_pos = i
                        multiPick.clear()

                        # calculate the path/time to return to the next pickup point
                        next_index = index + 1
                        while next_index not in block.index:
                            if next_index >= block.index.max():
                                next_index = next_index - 1
                                break
                            next_index = next_index + 1
                        innerMost = block.loc[next_index, "Code"]
                        outerMost = self.components["index"].str.match(innerMost)
                        nextLookUpTable = self.components[outerMost]
                        nextcart_coordinates = self.feedercarts[
                            nextLookUpTable.FeedStyle.max()
                        ]
                        next_pickup_vector = (
                            int(nextcart_coordinates[0])
                            + (nextLookUpTable.ST_No.max() * 10),
                            int(nextcart_coordinates[1]),
                        )
                        TIME = (
                            (
                                self.__calcVector(
                                    next_pickup_vector, current_pos, velocity
                                )
                            )
                            + TIME
                            + DROPOFF
                        )

                    elif row.Task == "Fiducial":
                        path_length = self.__calcVector(
                            (0, 0), location_vector_B, velocity
                        )
                        TIME = (path_length) + TIME

                    else:
                        # calculate the path/time for a single pickup
                        path_length = self.__calcVector(
                            location_vector_A, self.CHECKPOINT, velocity
                        )
                        checkpoint = self.__calcVector(
                            self.CHECKPOINT, location_vector_B, velocity
                        )
                        TIME = (path_length) + TIME + DROPOFF + checkpoint

                        # calculate the path/time to return to the next pickup point
                        next_index = index + 1
                        while next_index not in block.index:
                            if next_index > block.index.max():
                                next_index = next_index - 1
                                break
                            next_index = next_index + 1
                        nextLookUpTable = self.components[
                            self.components["index"].str.match(
                                block.loc[next_index, "Code"]
                            )
                        ]
                        nextcart_coordinates = self.feedercarts[
                            nextLookUpTable.FeedStyle.max()
                        ]
                        next_pickup_vector = (
                            int(nextcart_coordinates[0])
                            + (nextLookUpTable.ST_No.max() * 10),
                            int(nextcart_coordinates[1]),
                        )

                        TIME = (
                            (
                                self.__calcVector(
                                    next_pickup_vector, location_vector_B, velocity
                                )
                            )
                            + TIME
                            + DROPOFF
                            + PICKUP
                        )

                elif row.Task == "Fiducial":
                    path_length = self.__calcVector((0, 0), location_vector_B, velocity)
                    TIME = (path_length) + TIME

                else:
                    # all components get treated with single pick
                    # regardless if they can be multipicked
                    path_length = self.__calcVector(
                        location_vector_A, self.CHECKPOINT, velocity
                    )
                    checkpoint = self.__calcVector(
                        self.CHECKPOINT, location_vector_B, velocity
                    )
                    TIME = (path_length) + TIME + DROPOFF + checkpoint
                    next_index = index + 1
                    while next_index not in block.index:
                        if next_index > block.index.max():
                            next_index = next_index - 1
                            break
                        next_index = next_index + 1
                    nextLookUpTable = self.components[
                        self.components["index"].str.match(
                            block.loc[next_index, "Code"]
                        )
                    ]
                    nextcart_coordinates = self.feedercarts[
                        nextLookUpTable.FeedStyle.max()
                    ]
                    next_pickup_vector = (
                        int(nextcart_coordinates[0])
                        + (nextLookUpTable.ST_No.max() * 10),
                        int(nextcart_coordinates[1]),
                    )
                    TIME = (
                        (
                            self.__calcVector(
                                location_vector_B, next_pickup_vector, velocity
                            )
                        )
                        + TIME
                        + DROPOFF
                        + PICKUP
                    )

                # saving coordinates for visual plotting
                self.plotting_x.append(plot_coordinates[0])
                self.plotting_y.append(plot_coordinates[1])

        return {"time": TIME, "plot_x": self.plotting_x, "plot_y": self.plotting_y}

    def __call__(
        self,
        multiPickOption: bool = True,
        plotPCB: bool = False,
        useIdealState: bool = False,
    ) -> Union[float, dict]:
        """Start the assembly simulation.

        Args:
            multiPickOption (bool): If the simulation uses multipick. Multipich decreases assembly time. Defaults to True.
            plotPCB (bool): If coordinates should be returned. If true, returns a dict instead of a float. Defaults to False
            useIdealState (bool): If the ideal state of the machine should be used. Impacts the veloticy used.

        Returns:
            float: If plotPCB is set to False.
            dict: If plotPCB is set to True. Contains time, plot_x, plot_y.
        """
        self.multiPickOption = multiPickOption
        time = 0
        plotX = []
        plotY = []
        import time as tm

        start_time = tm.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in self.offsets:

                future = executor.submit(self.calcTime, i, useIdealState)
                iter_data = future.result()

                time = time + iter_data["time"]
                plotX.append(iter_data["plot_x"])
                plotY.append(iter_data["plot_y"])
        # print("--- %s seconds ---" % (tm.time() - start_time))
        if plotPCB == True:
            return {
                "time": time,
                "plot_x": list(itertools.chain.from_iterable(plotX)),
                "plot_y": list(itertools.chain.from_iterable(plotY)),
            }
        return time

    def coating(self) -> float:
        """simulates the time for coating a PCB

        Returns:
            float: The calculated time.
        """
        velocity = 20  # mm/s

        # highest coordinate on PCB
        offset = max(self.offsets)
        high = self.data["Y"].max() + offset[1]
        return math.sqrt(0**2 + high**2) / velocity


if __name__ == "__main__":
    from simulation.cartsetup import CartSetup
    from data.dataloader import DataLoader

    path = Path(os.getcwd() + os.path.normpath("/data/3160194"))
    dataloader = DataLoader(path)
    machine = Machine("M20", 19000, 4)
    manufacturing = Manufacturing(dataloader(), machine)
    manuData = manufacturing(multiPickOption=True, multithread=False, plotPCB=True)
    print(manuData["plot_x"])
