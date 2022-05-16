from pathlib import Path
import pandas as pd
import os
from os.path import exists
import re


class DataWrangler:
    """Class that handles Data processing for ML"""

    def __init__(self, machine: str) -> None:
        # replace with DB lookup
        pathM20 = Path(os.getcwd() + os.path.normpath(f"/data/logs/{machine}/"))
        pd.options.mode.chained_assignment = None

        product_DB = pd.read_csv(
            os.getcwd() + os.path.normpath(f"/data/global/product_db_dummy.csv"),
            encoding="unicode_escape",
            sep=";",
        )
        product_DB_cols = ["Kopfmaterial", "Anzahl Komponenten in Kopfmaterial"]
        product_DB = product_DB[product_DB_cols]
        product_DB["Kopfmaterial"] = product_DB["Kopfmaterial"].apply(lambda x: x.replace(".", ""))
        product_DB = product_DB.rename(columns={"Anzahl Komponenten in Kopfmaterial": "NumComp"})
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
            pcbcount["NumComponents"] = pcbcount["Kopfmaterial"].apply(lambda x: compare(x))
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

            outputpath = os.getcwd() + os.path.normpath(f"/data/logs/combined/data{machine}.csv")
            """if exists(outputpath):
                oldDF = pd.read_csv(outputpath, encoding='unicode_escape')
                pd.concat([oldDF, pcbcount], ignore_index=True)
                pcbcount = oldDF#.drop_duplicates()"""
            pcbcount = pcbcount[pcbcount["Kopfmaterial"].notna()]
            pcbcount.to_csv(outputpath, mode="a", header=not os.path.exists(outputpath))

    def returnData(self):
        df = pd.read_csv(
            os.getcwd() + os.path.normpath("/data/logs/combined/data.csv"),
            encoding="unicode_escape",
        )
        return df[["Date", "Contents", "Time_Needed"]].drop_duplicates()


class MachineDataLoader:
    def __init__(self) -> None:
        path = Path(os.getcwd() + os.path.normpath(f"/data/logs/machine/"))

        concatDataframe = pd.DataFrame()

        for i in path.iterdir():

            df = pd.read_csv(i, encoding="unicode_escape")
            df = df[["No.", "Date", "Component Code", "Fiducial#", "Offset"]]
            df = df.dropna()
            df = df.iloc[:-1, :]
            concatDataframe = pd.concat([concatDataframe, df], ignore_index=True)

        concatDataframe = concatDataframe[["No.", "Date", "Component Code", "Fiducial#", "Offset"]]
        list_df = []

        concatDataframe["Date"] = pd.to_datetime(concatDataframe["Date"], errors="coerce")
        concatDataframe["Offset"] = concatDataframe["Offset"].astype(int)

        for n, g in concatDataframe.groupby("Fiducial#"):
            list_df.append(g)

        prev = []
        nxt = []

        for i, v in enumerate(list_df):
            if i + 1 >= len(list_df):
                break
            nextV = list_df[i + 1]
            if nextV["Offset"].max() == 0:
                prev.append({"time": v["Date"].min(), "index": v["Date"].index[0]})
                prev.append({"time": nextV["Date"].max(), "index": nextV["Date"].index[0]})

        prev.pop()
        prev.pop(0)

        def pairwise(iterable):
            "s -> (s0, s1), (s2, s3), (s4, s5), ..."
            a = iter(iterable)
            return zip(a, a)

        output = pd.DataFrame()

        for x, y in pairwise(prev):
            timeNeeded = y["time"] - x["time"]
            placementsNeeded = y["index"] - x["index"]
            dataDict = {
                "StartTime": x["time"],
                "EndTime": y["time"],
                "placementsNeeded": placementsNeeded,
                "timeNeeded": timeNeeded.total_seconds(),
            }

            dataDict = pd.DataFrame(dataDict, [0])

            output = pd.concat([output, dataDict], ignore_index=True)

        self.data = output

    def returnData(self) -> pd.DataFrame:
        return self.data.to_dict()


if __name__ == "__main__":
    MachineDataLoader()
