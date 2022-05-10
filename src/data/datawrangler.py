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

        def check(row):
            found = re.search("(.+?) Updated PCB Count", row).group(1)
            return found

        for i in pathM20.iterdir():
            df = pd.read_csv(i, low_memory=False, encoding="unicode_escape")
            df = df.dropna()
            neededColumns = ["Date", "Contents"]
            df = df[neededColumns]
            pcbcount = df.loc[df["Contents"].str.contains("Updated PCB Count", case=False)]
            pcbcount["Date"] = pd.to_datetime(pcbcount["Date"])
            pcbcount["Product_Code"] = pcbcount["Contents"].apply(lambda x: check(x))

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

        outputpath = os.getcwd() + os.path.normpath("/data/logs/combined/data.csv")
        """if exists(outputpath):
            oldDF = pd.read_csv(outputpath, encoding='unicode_escape')
            pd.concat([oldDF, pcbcount], ignore_index=True)
            pcbcount = oldDF#.drop_duplicates()"""
        print(len(pcbcount.index))
        pcbcount.to_csv(outputpath, mode="a", header=not os.path.exists(outputpath))

    def returnData(self):
        df = pd.read_csv(
            os.getcwd() + os.path.normpath("/data/logs/combined/data.csv"),
            encoding="unicode_escape",
        )
        return df[["Date", "Contents", "Time_Needed"]].drop_duplicates()


if __name__ == "__main__":
    DataWrangler()
    # df = pd.read_csv(os.getcwd() + os.path.normpath('/data/logs/combined/data.csv'), encoding='unicode_escape')
    # print(df.info())
    # print(df[['Date', 'Contents', 'Time_Needed']].drop_duplicates())
