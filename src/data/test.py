from pathlib import Path
import pandas as pd
import os
from os.path import exists
import re


class MachineDataLoader:
    def __init__(self) -> None:
        path = Path(os.getcwd() + os.path.normpath("/data/logs/machine"))

        os.remove("output.txt")

        timings = []
        output = pd.DataFrame()

        for i in path.iterdir():
            df = pd.read_csv(i, encoding="unicode_escape", skiprows=1)
            df = df[["No.", "Date", "Fiducial#", "Offset"]]
            df = df.dropna()
            df = df.iloc[:-1, :]
            df["Fiducial#"] = df["Fiducial#"].astype(int)
            df["Offset"] = df["Offset"].astype(int)
            df["No."] = df["No."].astype(int)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            list_df = []

            for n, g in df.groupby(["Fiducial#"]):
                print(g, file=open("output.txt", "a"))
                list_df.append(g)

            print(len(list_df))
            placementTimings = []
            for i, v in enumerate(list_df):

                if i + 1 > len(list_df):
                    break

                if v["Offset"].max() == v["Offset"].min():
                    """Logic to combine multiple combs into one PCB assembly"""
                    if v["Offset"].min() == 0 or v["Offset"].max() == 0:
                        start = v["Date"].min()
                        placement = v["No."].min()
                        print(v, file=open("output.txt", "a"))
                        placementTimings.append({"start": start, "placement": placement})
                    else:

                        if i + 1 >= len(list_df):
                            end = v["Date"].max()
                            placement = v["No."].max()
                            print(v, file=open("output.txt", "a"))
                            placementTimings.append({"end": end, "placement": placement})
                            break
                        nextV = list_df[i + 1]
                        if (
                            nextV["Offset"].min() == 0
                            or nextV["Offset"].max() == 0
                            or i + 1 > len(list_df)
                        ):
                            end = v["Date"].max()
                            placement = v["No."].max()
                            print(v, file=open("output.txt", "a"))
                            placementTimings.append({"end": end, "placement": placement})
                            # continue

                else:
                    start = v["Date"].min()
                    end = v["Date"].max()
                    placementStart = v["No."].min()
                    placementEnd = v["No."].max()
                    placementTimings.append({"start": start, "placement": placementStart})
                    placementTimings.append({"end": end, "placement": placementEnd})
                    # timings = timings + placementTimings

                if "end" in list(placementTimings[0].keys()):
                    placementTimings.pop(0)
                # print(placementTimings)

            def pairwise(iterable):
                a = iter(iterable)
                return zip(a, a)

            if "end" in list(placementTimings[0].keys()):
                placementTimings.pop(0)

            for x, y in pairwise(placementTimings):
                timeNeeded = y["end"] - x["start"]
                placementsNeeded = y["placement"] - x["placement"]
                dataDict = {
                    "StartTime": x["start"],
                    "EndTime": y["end"],
                    "placementsNeeded": placementsNeeded,
                    "timeNeeded": timeNeeded.total_seconds(),
                }

                dataDict = pd.DataFrame(dataDict, [0])

                output = pd.concat([output, dataDict], ignore_index=True)

        print(output)


if __name__ == "__main__":
    MachineDataLoader()
