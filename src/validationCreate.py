import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from tqdm import tqdm

PACKAGE_PARENT = "../"
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from src.simulation.manufacturing import Manufacturing
from src.data.dataloader import DataLoader
from src.simulation.machine import Machine
from src.models.deploy import DeployModel


class ValidationCreate:
    def __init__(self) -> None:
        """Used to create validation data."""
        pass

    def __call__(self, mList: List, usable_programms: List) -> None:
        """Run this to start the validation Process.

        Args:
            mList (List): The list of machines to generate data for.
            usable_programms (List): List of programms to be used for data generation.
        """

        for m in mList:
            df = pd.read_csv(
                os.getcwd()
                + os.path.normpath(
                    f"/data/all/validation/{m['machine']}DataFrame_raw.csv"
                ),
                low_memory=False,
            )
            df = df.drop_duplicates()
            runsList = []
            for i in usable_programms:
                runs = df.loc[df["Class"] == i]
                if len(runs) > 10 and len(runs) < 600:
                    runsList.append(runs.Class.iat[0])

            class DummyMachine:
                def __init__(self, machine) -> None:
                    for key in machine:
                        setattr(self, key, machine[key])

            dummyMachine = DummyMachine(m)
            model = DeployModel(
                Path(os.getcwd() + os.path.normpath("\data\models\FINAL MODEL")),
            )
            machineData = []
            print(f'Creating validation Data for {m["machine"]} ')
            for productId in tqdm(runsList):
                try:
                    path = Path(
                        os.getcwd()
                        + os.path.normpath(
                            "/data/programms/" + productId + "/" + dummyMachine.machine
                        )
                    )
                    data = DataLoader(path)
                    machine = Machine(dummyMachine)
                    manufacturing = Manufacturing(data(), machine)
                    data = data()
                    offsets = max(data[2])
                    simulationData = manufacturing(plotPCB=False, useIdealState=True)
                    predArray = np.array(
                        [
                            len(data[0]) * len(data[2]),
                            0 if machine.machineName == "m10" else 1,
                            data[0]["X"].max() + offsets[0],
                            data[0]["Y"].max() + offsets[1],
                        ]
                    )

                    predictedData = model.predict(predArray)
                    for i in range(150):

                        machineData.append(
                            (
                                productId,
                                "Classic",
                                simulationData + np.random.randint(-60, 180),  # [0][0],
                                len(data[0]),
                            )
                        )

                        if predictedData == np.nan:
                            predictedData = [[0]]
                        machineData.append(
                            (
                                productId,
                                "AI Model",
                                predictedData[0][0]
                                + np.random.randint(-60, 180),  # [0][0],
                                len(data[0]),
                            )
                        )

                except Exception as e:
                    print("====== ERROR ======")
                    continue

            machineData = pd.DataFrame(
                machineData, columns=["Class", "Type", "Time", "Placements"]
            )

            machineData.to_csv(
                os.getcwd()
                + os.path.normpath(
                    f"/data/all/validation/{m['machine']}DataFrame_generated_2.csv"
                ),
            )
            df = pd.concat([df, machineData], ignore_index=True)
            df = df.sort_values(by="Class")
            df.to_csv(
                os.getcwd()
                + os.path.normpath(
                    f"/data/all/validation/{m['machine']}DataFrame_2.csv"
                ),
            )


if __name__ == "__main__":
    old_programms = [
        "2495806",
        "2495805",
        "3024122",
        "25ACDAB",
        "2495801",
        "2495803",
        "3030092",
        "3170275",
        "24ACAAB",
        "24ABCAB",
        "2696066",
        "2696082",
        "3042031",
        "3042032",
        "3170218",
        "3024167",
        "26AAUAB",
        "2195890",
        "3170228",
        "3010894",
        "2489800",
        "2196800",
        "2696003",
        "2799702",
        "3160165",
        "3010736",
        "3055470",
        "2799764",
        "2799762",
        "2799760",
        "2799766",
        "3160194",
        "25AAAAB",
        "25ACYAB",
        "3011902",
        "3170261",
        "2696067",
        "2696050",
        "2696001",
        "2696040",
        "2799750",
        "3043814",
        "3055515",
        "3011318",
        "3010844",
        "3010741",
        "3043808",
        "25ACZAB",
        "3010854",
        "2306886",
        "3170205",
        "3055252",
        "27AMWAB",
        "2799700",
        "24AAGAB",
        "2696084",
        "25ACVAB",
        "3052453",
        "3170225",
        "3170295",
        "3010225",
        "3170235",
        "3010834",
        "3021113",
        "2696017",
        "3055062",
        "24BDDAB",
        "2196820",
        "3055320",
        "26AAVAB",
        "3024166",
        "3170241",
        "2799705",
        "24AARAB",
        "24AANAB",
        "3055364",
        "3024135",
        "2799720",
        "2696085",
        "2799770",
        "2799771",
        "3011325",
        "3011330",
        "3110124",
        "3011901",
        "25ADEAB",
        "3010733",
        "2495804",
        "3170584",
        "3170298",
        "2195891",
        "3024137",
        "3024136",
        "2196821",
        "25ACNAB",
        "26AAWAB",
        "26APHAB",
        "2799730",
        "2799728",
        "27ALVAB",
        "2799752",
        "2696051",
        "25ACSAB",
        "3170285",
        "3055490",
        "2696018",
        "3170238",
        "3082771",
        "3024155",
        "3024156",
        "3024157",
        "3170215",
        "3055401",
        "2696086",
        "3062764",
        "3170544",
        "3170251",
        "3160224",
        "3062814",
        "25ACKAB",
        "2799780",
        "3062783",
        "2495844",
        "2095556",
        "25ADAAB",
        "2799724",
        "3170288",
        "27ALXAB",
        "27ALWAB",
        "2696036",
        "3055424",
        "2095558",
        "3020944",
        "3024165",
        "26ABAAB",
        "2095559",
        "2095557",
        "3070042",
        "3055414",
        "3030114",
        "2696101",
        "2696111",
        "2696108",
        "2696118",
        "2696151",
        "2696155",
        "2799703",
        "24ABHAB",
        "3055452",
        "3062774",
        "2695012",
        "2695013",
        "3170564",
        "25ACAAB",
        "3055525",
        "3062561",
        "3170574",
        "2095525",
        "26AAIAB",
        "3082661",
        "3010222",
        "3011884",
        "3011894",
        "2696004",
        "2799781",
        "3170266",
        "3170296",
        "3170554",
        "3170246",
        "2799740",
        "3170740",
        "3170742",
        "2799701",
        "2695011",
        "2196822",
        "PP17801",
        "PP17501",
        "PP17701",
        "PP17601",
        "2799732",
        "4300490",
        "4300491",
        "26APJAB",
        "3055545",
        "3009901",
        "3160196",
        "3160183",
        "27AMZAB",
        "3011962",
        "2296965",
        "2495850",
    ]

    path = Path(os.getcwd() + os.path.normpath("/data/programms"))

    new_programms = set(os.listdir(path))

    usable_programms = list(new_programms.intersection(old_programms))

    mList = [
        {
            "machine": "m10",
            "cph": 23000,
            "nozHeads": 6,
            "SMD": True,
            "offsets": {
                "checkpoint": [30, -20],
                "pcb": [0, 110],
                "tools": [30, 6320],
                "feedercarts": [
                    {"ST-FL": [160, 20]},
                    {"ST-RL": [160, 632]},
                    {"ST-FR": ["0", "0"]},
                    {"ST-RR": ["0", "0"]},
                ],
            },
        },
        {
            "machine": "m20",
            "cph": 19000,
            "nozHeads": 4,
            "SMD": True,
            "offsets": {
                "checkpoint": [110, 40],
                "pcb": [0, 70],
                "tools": [600, 900],
                "feedercarts": [
                    {"ST-FL": [200, 0]},
                    {"ST-RL": [200, 632]},
                    {"ST-FR": ["-390", "0"]},
                    {"ST-RR": ["-390", "632"]},
                ],
            },
        },
    ]

    validationcreate = ValidationCreate()

    validationcreate(mList, usable_programms)
