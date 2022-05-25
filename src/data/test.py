from pathlib import Path
import time as tm
import numpy as np
import pandas as pd
import os
from os.path import exists
import concurrent.futures
import threading
import re


class MyFancyException(Exception):
    pass


class MachineDataLoader:
    def __init__(self) -> None:
        path = Path(os.getcwd() + os.path.normpath("/data/logs/machine"))

        """try:
            os.remove("output.txt")
        except:
            pass"""

        timings = []

        for i in path.iterdir():
            start_time = tm.time()
            print(f"Reading new Dataset: {i}", file=open("output.txt", "a"))
            print(f"Reading new Dataset: {i}")
            df = pd.read_csv(i, encoding="unicode_escape", skiprows=1, low_memory=False)
            print(
                "Dataset reading complete",
                tm.time() - start_time,
                file=open("output.txt", "a"),
            )
            print(
                "Dataset reading complete",
                tm.time() - start_time,
            )
            df = df[["No.", "Date", "Fiducial#", "Offset", "Head#", "Feed Style"]]
            df = df.dropna()
            df = df.iloc[:-1, :]
            df["Fiducial#"] = df["Fiducial#"].astype(int)
            df["Offset"] = df["Offset"].astype(int)
            df["No."] = df["No."].astype(int)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Head#"] = df["Head#"].astype(int)
            # print(df["Fiducial#"].value_counts().drop_duplicates())

            DFList = []

            for n, g in df.groupby(by=df["Date"].dt.date):
                DFList.append(g)

            print(f"Reading logs for {len(DFList)} Days ", file=open("output.txt", "a"))
            print(f"Reading logs for {len(DFList)} Days ")

            output = pd.DataFrame()
            dayCounter = 0
            results = []
            start = tm.perf_counter()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for day in DFList:
                    dayCounter = dayCounter + 1
                    print(f"Starting Day {dayCounter} ", file=open("output.txt", "a"))
                    print(f"Starting Day {dayCounter} ")
                    results.append(executor.submit(self.calcTimings, day, dayCounter))

            for f in results:
                dayoutput = f.result()
                output = pd.concat([output, dayoutput], ignore_index=True)
            print(
                f"===== Duration Multithreading: {tm.perf_counter() - start} =====",
                file=open("output.txt", "a"),
            )
            print(f"===== Duration Multithreading: {tm.perf_counter() - start} =====")
            output = output.drop_duplicates()
            # endtime = output["EndTime"].max()
            # starttime = output["StartTime"].min()
            # endtime = re.search("(.+?) (.+?)", str(endtime)).group(1)
            # starttime = re.search("(.+?) (.+?)", str(starttime)).group(1)
            outputPath = Path(
                os.getcwd() + os.path.normpath(f"/data/logs/combined/m20_timings.csv")
            )
            output.to_csv(outputPath, mode="a", header=not os.path.exists(outputPath))
            """start = tm.perf_counter()
            dayCounter = 0
            for day in DFList:
                dayCounter = dayCounter + 1
                print(f"Starting sequential Day {dayCounter} ")
                self.calcTimings(day, dayCounter)

            print(f"===== Duration Sequential: {tm.perf_counter() - start} =====")"""

    def calcTimings(self, day: pd.DataFrame, dayCount: int) -> pd.DataFrame:
        list_df = []
        for n, g in day.groupby(["Fiducial#"]):
            # print(g, file=open("output.txt", "a"))
            list_df.append(g)

        print(
            f"Processing {len(list_df)} Batches for Day {dayCount}",
            file=open("output.txt", "a"),
        )
        print(f"Processing {len(list_df)} Batches for Day {dayCount}")
        placementTimings = []
        for i, v in enumerate(list_df):

            if i + 1 > len(list_df):
                break

            if v["Offset"].max() == v["Offset"].min():
                """Logic to combine multiple combs into one PCB assembly"""
                if v["Offset"].min() == 0 or v["Offset"].max() == 0:
                    start = v["Date"].min()
                    placement = v["No."].min()
                    # print(v, file=open("output.txt", "a"))
                    placementTimings.append(
                        {
                            "start": start,
                            "placement": placement,
                            "head": v["Head#"],
                            "FeedStyle": v["Feed Style"],
                        }
                    )
                else:

                    if i + 1 >= len(list_df):
                        end = v["Date"].max()
                        placement = v["No."].max()
                        # print(v, file=open("output.txt", "a"))
                        placementTimings.append(
                            {
                                "end": end,
                                "placement": placement,
                                "head": v["Head#"],
                                "FeedStyle": v["Feed Style"],
                            }
                        )
                        break
                    nextV = list_df[i + 1]
                    if (
                        nextV["Offset"].min() == 0
                        or nextV["Offset"].max() == 0
                        or i + 1 > len(list_df)
                    ):
                        end = v["Date"].max()
                        placement = v["No."].max()
                        # print(v, file=open("output.txt", "a"))
                        placementTimings.append(
                            {
                                "end": end,
                                "placement": placement,
                                "head": v["Head#"],
                                "FeedStyle": v["Feed Style"],
                            }
                        )
                        # continue

            else:
                start = v["Date"].min()
                end = v["Date"].max()
                placementStart = v["No."].min()
                placementEnd = v["No."].max()
                placementTimings.append(
                    {
                        "start": start,
                        "placement": placementStart,
                        "head": v["Head#"],
                        "FeedStyle": v["Feed Style"],
                    }
                )
                placementTimings.append(
                    {
                        "end": end,
                        "placement": placementEnd,
                        "head": v["Head#"],
                        "FeedStyle": v["Feed Style"],
                    }
                )
                # timings = timings + placementTimings

            if "end" in list(placementTimings[0].keys()):
                placementTimings.pop(0)
            # print(placementTimings)

        def pairwise(iterable):
            a = iter(iterable)
            return zip(a, a)

        if "end" in list(placementTimings[0].keys()):
            placementTimings.pop(0)
        del list_df
        print(f"Calculating Timings for Day {dayCount}", file=open("output.txt", "a"))
        print(f"Calculating Timings for Day {dayCount}")
        while True:
            try:
                # print("Org. List", len(placementTimings))
                # print("Output List", len(output))
                outputDay = pd.DataFrame(None)
                for x, y in pairwise(placementTimings):
                    try:
                        timeNeeded = y["end"] - x["start"]
                        placementsNeeded = y["placement"] - x["placement"]
                        timeNeeded = (
                            1
                            if timeNeeded.total_seconds() == 0
                            else timeNeeded.total_seconds()
                        )
                        dataDict = {
                            "StartTime": x["start"],
                            "EndTime": y["end"],
                            "placementsNeeded": placementsNeeded,
                            "timeNeeded": timeNeeded,
                            "machine": "m20",
                            "heads": day["Head#"].max(),
                            "cph": (placementsNeeded / timeNeeded) * 3600,
                        }

                        dataDict = pd.DataFrame(dataDict, [0])

                        outputDay = pd.concat([outputDay, dataDict], ignore_index=True)
                    except:
                        # print("An error occured. Removing false Data")
                        placementTimings.remove(x)
                        raise MyFancyException
                break
            except:
                continue
        print(f"Calculations for Day {dayCount} complete", file=open("output.txt", "a"))
        print(f"Calculations for Day {dayCount} complete")
        del placementTimings
        return outputDay


class TimeFinder:
    def __init__(self) -> None:
        path = Path(os.getcwd() + os.path.normpath("/data/logs/machine"))

        for i in path.iterdir():
            start_time = tm.time()
            print(f"Reading new Dataset: {i}", file=open("output.txt", "a"))
            print(f"Reading new Dataset: {i}")
            df = pd.read_csv(i, encoding="unicode_escape", skiprows=1, low_memory=False)
            print(
                "Dataset reading complete",
                tm.time() - start_time,
                file=open("output.txt", "a"),
            )
            print(
                "Dataset reading complete",
                tm.time() - start_time,
            )
            df = df[["Date"]]
            df = df.dropna()
            df = df.iloc[:-1, :]
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            print(len(df))
            df = df.drop_duplicates()
            print(len(df))
            df = df.to_numpy()


if __name__ == "__main__":
    MachineDataLoader()
    # TimeFinder()
