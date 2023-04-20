import glob

import pandas as pd
from gluonts.dataset.common import TrainDatasets
from gluonts.dataset.repository.datasets import get_dataset


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
