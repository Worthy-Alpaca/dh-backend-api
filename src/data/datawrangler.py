from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import os
import concurrent.futures
import time as tm
import re


class DataWrangler:
    """Class that handles Data processing for ML"""

    def __init__(self, machine: str) -> None:
        # replace with DB lookup
        self.machine = machine
        pathM20 = Path(os.getcwd() + os.path.normpath(f"/data/logs/{machine}/"))
        pd.options.mode.chained_assignment = None

        product_DB = pd.read_csv(
            os.getcwd() + os.path.normpath(f"/data/global/product_db_dummy.csv"),
            encoding="unicode_escape",
            sep=";",
        )
        product_DB_cols = ["Kopfmaterial", "Anzahl Komponenten in Kopfmaterial"]
        product_DB = product_DB[product_DB_cols]
        product_DB["Kopfmaterial"] = product_DB["Kopfmaterial"].apply(
            lambda x: x.replace(".", "")
        )
        product_DB = product_DB.rename(
            columns={"Anzahl Komponenten in Kopfmaterial": "NumComp"}
        )
        product_DB = product_DB.drop_duplicates()
        print(product_DB.info())

        def check(row):
            if row == "LP begibt sich zu Punkt LP Einlaufband.":
                return None
            found = re.search("(.+?) Updated PCB Count", row).group(1)
            return found[0:7]

        def compare(x):
            if x == None:
                return 0
            compareData = product_DB[product_DB["Kopfmaterial"].str.match(x)]
            return compareData.NumComp.max()

        temp = pd.DataFrame()

        for i in pathM20.iterdir():
            df = pd.read_csv(i, low_memory=False, encoding="unicode_escape")
            df = df.dropna()
            neededColumns = ["Date", "Contents"]
            df = df[neededColumns]
            matchers = ["Updated PCB Count", "LP begibt sich zu Punkt LP Einlaufband."]
            loc = "|".join(matchers)
            pcbcount = df.loc[df["Contents"].str.contains(loc)]
            print(pcbcount)
            pcbcount["Date"] = pd.to_datetime(pcbcount["Date"])
            pcbcount["Kopfmaterial"] = pcbcount["Contents"].apply(lambda x: check(x))
            pcbcount["NumComponents"] = pcbcount["Kopfmaterial"].apply(
                lambda x: compare(x)
            )
            """pcbcount = pd.merge(
                left=pcbcount,
                left_on="Kopfmaterial",
                right=product_DB,
                right_on="Kopfmaterial",
                how="left",
            )"""

            for index, row in pcbcount.iterrows():
                nextIndex = index - 1
                while nextIndex not in pcbcount.index:
                    if nextIndex <= pcbcount.index.min():
                        nextIndex = nextIndex + 1
                        break
                    nextIndex = nextIndex - 1

                nextLookUpTable = pcbcount.loc[nextIndex]
                timeNeeded = row.Date - nextLookUpTable.Date
                timeNeeded = timeNeeded.total_seconds()
                pcbcount.loc[index, "Time_Needed"] = timeNeeded
            # pd.concat([temp, pcbcount])
            pcbcount = pcbcount[pcbcount["Kopfmaterial"].notna()]
            self.pcbcount = pcbcount

    def returnData(self):

        return self.pcbcount[["Date", "Contents", "Time_Needed"]].drop_duplicates()

    def saveData(self):
        outputpath = os.getcwd() + os.path.normpath(
            f"/data/logs/combined/data{self.machine}.csv"
        )
        self.pcbcount.to_csv(
            outputpath, mode="a", header=not os.path.exists(outputpath)
        )


class MyFancyException(Exception):
    pass


class MachineDataLoader:
    def __init__(self, machine) -> None:
        path = Path(os.getcwd() + os.path.normpath(f"/data/logs/machine {machine}"))

        """try:
            os.remove("output.txt")
        except:
            pass"""

        timings = []

        for i in path.iterdir():
            start_time = tm.time()
            print(f"Reading new Dataset: {i}", file=open("output.txt", "a"))
            print(f"Reading new Dataset: {i}")
            df = self.loadData(i)
            print(
                "Dataset reading complete",
                tm.time() - start_time,
                file=open("output.txt", "a"),
            )
            print(
                "Dataset reading complete",
                tm.time() - start_time,
            )

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
            # output = output.drop_duplicates()
            output.sort_values(by="StartTime", inplace=True)
            # endtime = output["EndTime"].max()
            # starttime = output["StartTime"].min()
            # endtime = re.search("(.+?) (.+?)", str(endtime)).group(1)
            # starttime = re.search("(.+?) (.+?)", str(starttime)).group(1)

            outputPath = Path(
                os.getcwd()
                + os.path.normpath(f"/data/logs/combined/machine_timings_combined.csv")
            )
            output.to_csv(outputPath, mode="a", header=not os.path.exists(outputPath))
            """start = tm.perf_counter()
            dayCounter = 0
            for day in DFList:
                dayCounter = dayCounter + 1
                print(f"Starting sequential Day {dayCounter} ")
                self.calcTimings(day, dayCounter)

            print(f"===== Duration Sequential: {tm.perf_counter() - start} =====")"""

    def loadData(self, path):
        skip = -1
        while True:
            skip = skip + 1

            df = pd.read_csv(
                str(path), skiprows=skip, encoding="unicode_escape", low_memory=False
            )
            if "Date" in df.columns:
                break

        df = df[
            [
                "No.",
                "Date",
                "Fiducial#",
                "Offset",
                "Head#",
                "Feed Style",
                "PCB ID",
                "Place(prgm) X",
                "Place(prgm) Y",
            ]
        ]
        df = df.dropna()
        df = df.iloc[:-1, :]
        df["No."] = df["No."].astype(int)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Head#"] = df["Head#"].astype(int)
        df[["Place(prgm) X", "Place(prgm) Y"]] = (
            df[["Place(prgm) X", "Place(prgm) Y"]].astype(float) / 1000
        )
        return df

    def calcTimings(self, day: pd.DataFrame, dayCount: int) -> pd.DataFrame:
        list_df = []
        for n, g in day.groupby(["PCB ID"]):
            # print(g, file=open("output.txt", "a"))
            list_df.append(g)
        print(
            f"Processing {len(list_df)} Batches for Day {dayCount}",
            file=open("output.txt", "a"),
        )
        print(f"Processing {len(list_df)} Batches for Day {dayCount}")
        placementTimings = []
        for i, v in enumerate(list_df):

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
                    "min_Y": v["Place(prgm) Y"].min(),
                    "min_X": v["Place(prgm) X"].min(),
                    "class": v["PCB ID"],
                }
            )
            placementTimings.append(
                {
                    "end": end,
                    "placement": placementEnd,
                    "head": v["Head#"],
                    "FeedStyle": v["Feed Style"],
                    "max_Y": v["Place(prgm) Y"].max(),
                    "max_X": v["Place(prgm) X"].max(),
                    "class": v["PCB ID"],
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
                        width = y["max_X"] - x["min_X"]
                        height = y["max_Y"] - x["min_Y"]
                        programm_class = x["class"].iat[0]
                        # print(programm_class)
                        timeNeeded = (
                            1
                            if timeNeeded.total_seconds() == 0
                            else timeNeeded.total_seconds()
                        )
                        cph = (placementsNeeded / timeNeeded) * 3600
                        if cph > 100000:
                            continue
                        heads = day["Head#"].max()
                        if heads == 4:
                            machine = "m20"
                        else:
                            machine = "m10"
                        dataDict = {
                            "StartTime": x["start"],
                            "EndTime": y["end"],
                            "placementsNeeded": placementsNeeded,
                            "timeNeeded": timeNeeded,
                            "machine": machine,
                            "heads": heads,
                            "cph": cph,
                            "width": width,
                            "height": height,
                            "class": programm_class,
                        }

                        dataDict = pd.DataFrame(dataDict, [0])

                        outputDay = pd.concat([outputDay, dataDict], ignore_index=True)
                        outputDay.sort_values(by="StartTime", inplace=True)
                    except Exception as e:
                        print("An error occured. Removing false Data")
                        placementTimings.remove(x)
                        raise MyFancyException
                break
            except:
                continue
        print(f"Calculations for Day {dayCount} complete", file=open("output.txt", "a"))
        print(f"Calculations for Day {dayCount} complete")
        del placementTimings
        return outputDay


if __name__ == "__main__":
    for m in ["m20", "m10"]:
        MachineDataLoader(m)

    outputPath = Path(
        os.getcwd()
        + os.path.normpath(f"/data/logs/combined/machine_timings_combined.csv")
    )
    df = pd.read_csv(outputPath)
    print(df.describe())
    plt.scatter(df["placementsNeeded"], df["cph"])
    plt.show()
