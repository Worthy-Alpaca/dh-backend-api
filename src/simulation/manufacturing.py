import math
import sys
import os
import itertools
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


class Manufacturing:
    """
    Simulates the manufacturing process in SMD machines M10 and M20
    - `machine`: either M10 (default) or M20

    """

    def __init__(self, data: tuple, machine: Machine) -> object:
        # assign machine property
        self.machine = machine
        self.components = data[1]
        self.data = data[0]
        self.offsets = data[2]
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

    def __calcVector(self, vectorA: tuple, vectorB: tuple, velocity: float):
        """
        Calculates the pathlength between two given vectors
        - vectorA: where you are
        - vectorB: where you want to go
        """
        vector_AB = (
            (float(vectorB[0]) - float(vectorA[0])),
            (float(vectorB[1]) - float(vectorA[1])),
        )
        path_length = math.sqrt(vector_AB[0] ** 2 + vector_AB[1] ** 2)

        return path_length / velocity

    def calcTime(self, offset_row: tuple):
        """
        Calculates the time for a given offset
        """

        def isNan(string):
            return string != string

        TIME = 0
        self.plotting_x = []
        self.plotting_y = []
        multiPick = deque()
        for index, row in self.data.iterrows():
            # check for NaN values and continue if found
            if isNan(row.Code):
                print(row.Code)
                continue
            # calculating the path, adding offset for coordinate transformation
            lookupTable = self.components[self.components["index"].str.match(row.Code)]

            cart_coordinates = self.feedercarts[lookupTable.FeedStyle.max()]
            location_vector_A = (
                int(cart_coordinates[0]) + (lookupTable.ST_No.max() * 10),
                int(cart_coordinates[1]),
            )
            location_vector_B = (
                (row.X + self.OFFSET_X + offset_row[0]),
                (row.Y + self.OFFSET_Y + offset_row[1]),
            )
            plot_coordinates = ((row.X + offset_row[0]), (row.Y + offset_row[1]))
            velocity = self.machine.velocity * (lookupTable.mean_acceleration.max() / 1000)

            if self.multiPickOption == True:
                # picking components with multiple heads at once
                # path changes from "Pickup -> Component" to "Pickup1 -> Pickup2 -> Pickup3 -> Component3 -> Component2 -> Component1"
                if row.Task == "Multiple Pickup":
                    # append component vector to queue
                    multiPick.append(location_vector_B)

                    # calculate path/time to next pickup
                    next_index = index + 1
                    while next_index not in self.data.index:
                        if next_index >= self.data.index.max():
                            next_index = next_index - 1
                            break
                        next_index = next_index + 1
                    nextLookUpTable = self.components[
                        self.components["index"].str.match(self.data.loc[next_index, "Code"])
                    ]
                    nextcart_coordinates = self.feedercarts[nextLookUpTable.FeedStyle.max()]
                    next_pickup = (
                        int(nextcart_coordinates[0]) + (nextLookUpTable.ST_No.max() * 10),
                        int(nextcart_coordinates[1]),
                    )
                    TIME = (
                        self.__calcVector(location_vector_A, next_pickup, velocity) + TIME + PICKUP
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
                    checkpoint = self.__calcVector(loc_vector_A, self.CHECKPOINT, velocity)
                    path = self.__calcVector(self.CHECKPOINT, loc_vector_B, velocity)
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
                    while next_index not in self.data.index:
                        if next_index >= self.data.index.max():
                            next_index = next_index - 1
                            break
                        next_index = next_index + 1
                    nextLookUpTable = self.components[
                        self.components["index"].str.match(self.data.loc[next_index, "Code"])
                    ]
                    nextcart_coordinates = self.feedercarts[nextLookUpTable.FeedStyle.max()]
                    next_pickup_vector = (
                        int(nextcart_coordinates[0]) + (nextLookUpTable.ST_No.max() * 10),
                        int(nextcart_coordinates[1]),
                    )
                    TIME = (
                        (self.__calcVector(next_pickup_vector, current_pos, velocity))
                        + TIME
                        + DROPOFF
                    )

                elif row.Task == "Fiducial":
                    path_length = self.__calcVector((0, 0), location_vector_B, velocity)
                    TIME = (path_length) + TIME

                else:
                    # calculate the path/time for a single pickup
                    path_length = self.__calcVector(location_vector_A, self.CHECKPOINT, velocity)
                    checkpoint = self.__calcVector(self.CHECKPOINT, location_vector_B, velocity)
                    TIME = (path_length) + TIME + DROPOFF + checkpoint

                    # calculate the path/time to return to the next pickup point
                    next_index = index + 1
                    while next_index not in self.data.index:
                        if next_index > self.data.index.max():
                            next_index = next_index - 1
                            break
                        next_index = next_index + 1
                    nextLookUpTable = self.components[
                        self.components["index"].str.match(self.data.loc[next_index, "Code"])
                    ]
                    nextcart_coordinates = self.feedercarts[nextLookUpTable.FeedStyle.max()]
                    next_pickup_vector = (
                        int(nextcart_coordinates[0]) + (nextLookUpTable.ST_No.max() * 10),
                        int(nextcart_coordinates[1]),
                    )

                    TIME = (
                        (self.__calcVector(next_pickup_vector, location_vector_B, velocity))
                        + TIME
                        + DROPOFF
                        + PICKUP
                    )

            elif row.Task == "Fiducial":
                path_length = self.__calcVector((0, 0), location_vector_B, velocity)
                TIME = (path_length) + TIME

            else:
                # all components get treated with single pick
                path_length = self.__calcVector(location_vector_A, self.CHECKPOINT, velocity)
                checkpoint = self.__calcVector(self.CHECKPOINT, location_vector_B, velocity)
                TIME = (path_length) + TIME + DROPOFF + checkpoint
                next_index = index + 1
                while next_index not in self.data.index:
                    if next_index > self.data.index.max():
                        next_index = next_index - 1
                        break
                    next_index = next_index + 1
                nextLookUpTable = self.components[
                    self.components["index"].str.match(self.data.loc[next_index, "Code"])
                ]
                nextcart_coordinates = self.feedercarts[nextLookUpTable.FeedStyle.max()]
                next_pickup_vector = (
                    int(nextcart_coordinates[0]) + (nextLookUpTable.ST_No.max() * 10),
                    int(nextcart_coordinates[1]),
                )
                TIME = (
                    (self.__calcVector(location_vector_B, next_pickup_vector, velocity))
                    + TIME
                    + DROPOFF
                    + PICKUP
                )

            # saving coordinates for visual plotting
            self.plotting_x.append(plot_coordinates[0])
            self.plotting_y.append(plot_coordinates[1])
        return {"time": TIME, "plot_x": self.plotting_x, "plot_y": self.plotting_y}

    def __call__(
        self, multiPickOption: bool = True, plotPCB: bool = False, *args: any, **kwds: any
    ) -> (float | dict):
        """Start the assembly simulation"""
        self.multiPickOption = multiPickOption
        time = 0
        plotX = []
        plotY = []
        import time as tm

        start_time = tm.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in self.offsets:

                future = executor.submit(self.calcTime, i)
                iter_data = future.result()

                time = time + iter_data["time"]
                plotX.append(iter_data["plot_x"])
                plotY.append(iter_data["plot_y"])
        print("--- %s seconds ---" % (tm.time() - start_time))
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
