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


if __name__ == "__main__":
    DataWrangler("m20")
    # df = pd.read_csv(os.getcwd() + os.path.normpath('/data/logs/combined/data.csv'), encoding='unicode_escape')
    # print(df.info())
    # print(df[['Date', 'Contents', 'Time_Needed']].drop_duplicates())
