import glob

import pandas as pd
import numpy as np
from gluonts.dataset.common import TrainDatasets
from gluonts.dataset.repository.datasets import get_dataset

from datetime import datetime
from dateutil import relativedelta


def add_meteo(
    path: str = "data/all_weather/",
    start: str = "15-01-2022",
    end: str = "25-02-2022",
    features: list = ["t", "rr3", "pmer"],
    station: int = 7149,
    target: str = "t",
) -> pd.DataFrame:

    start = datetime.strptime(start, "%d-%m-%Y")
    end = datetime.strptime(end, "%d-%m-%Y")

    columns = ["numer_sta", "date"] + features

    diff = relativedelta.relativedelta(end, start)
    month_difference = diff.years * 12 + diff.months

    df = pd.DataFrame(columns=columns)

    for i in range(month_difference + 1):
        date = start + relativedelta.relativedelta(months=i)

        YM_str = date.strftime("%Y%m")
        path_ = path + "synop." + YM_str + ".csv.gz"

        df2 = pd.read_csv(path_, sep=";", compression="gzip")

        df2["date"] = df2["date"].apply(lambda x: datetime.strptime(str(x), "%Y%m%d%H%M%S"))
        df2 = df2[columns]
        df2 = df2[df2["numer_sta"] == station]

        if i == 0:
            df2 = df2[df2["date"] >= start]

        if i == month_difference:
            df2 = df2[df2["date"] <= end]

        df = pd.concat([df, df2], ignore_index=True)

    df.drop(["numer_sta"], axis=1, inplace=True)

    df[features] = df[features].fillna(np.median)
    df.set_index("date", inplace=True)
    df = df.resample("15T").ffill()

    # df.index = df.index.map(lambda x: x.strftime("%d-%m-%Y-%H-%M"))
    # df.reset_index(inplace=True)

    return df

    # return df[[target]]


def climate(path: str = "data/climate_delhi/", target: str = "mean_temp") -> pd.DataFrame:
    list_csv = glob.glob(path + "*.csv")
    df_climate = pd.DataFrame()
    for file in list_csv:
        df_tmp = pd.read_csv(file)
        df_climate = pd.concat([df_climate, df_tmp], axis=0)
    df_climate = df_climate.set_index("date")
    df_climate.index = pd.to_datetime(df_climate.index)
    df_climate = df_climate[[target]]

    return df_climate


def energy(path: str = "data/energy/", target: str = "consommation"):
    list_csv = glob.glob(path + "*.csv")
    df_energy = pd.DataFrame()
    for file in list_csv:
        df_tmp = pd.read_csv(file, sep=";")
        df_energy = pd.concat([df_energy, df_tmp], axis=0)
    df_energy.rename(
        columns={"heure": "hour", "libelle_region": "variable", target: "value"}, inplace=True
    )
    df_energy = df_energy[["date", "hour", "variable", "value"]]
    df_energy = df_energy[df_energy.variable != "Grand Est"]
    df_energy["date"] = pd.to_datetime(
        df_energy.date + " " + df_energy.hour, format="%Y-%m-%d %H:%M"
    )
    df_energy = df_energy.sort_values(by="date", ascending=True).reset_index(drop=True)
    df_energy = pd.pivot_table(df_energy, values="value", index=["date"], columns=["variable"])
    df_energy = df_energy.resample("15T").interpolate(method="linear")

    return df_energy


def gluonts_dataset(dataset_name: str) -> TrainDatasets:
    return get_dataset(dataset_name)
