import glob
import pandas as pd
from gluonts.dataset.common import TrainDatasets
from gluonts.dataset.repository.datasets import get_dataset
from load.load_exo_data import add_weather
from typing import Dict


def climate(
    path: str = "data/climate_delhi/",
    target: str = "mean_temp",
    weather: Dict[str, any] = {
        "path_weather": "data/all_weather/",
        "dynamic_features": ["t", "rr3", "pmer"],
        "cat_features": ["cod_tend"],
        "station_name": "ORLY",
    },
) -> pd.DataFrame:
    list_csv = glob.glob(path + "*.csv")
    df_climate = pd.DataFrame()
    for file in list_csv:
        df_tmp = pd.read_csv(file)
        df_climate = pd.concat([df_climate, df_tmp], axis=0)
    df_climate = df_climate.set_index("date")
    df_climate.index = pd.to_datetime(df_climate.index)
    df_climate = df_climate[[target]]

    if weather:
        df_climate = add_weather(df_climate, weather)

    return df_climate


def energy(
    path: str = "data/energy/",
    target: str = "consommation",
    weather: Dict[str, any] = {
        "path_weather": "data/all_weather/",
        "dynamic_features": ["t", "rr3", "pmer"],
        "cat_features": ["cod_tend"],
        "station_name": "ORLY",
    },
) -> pd.DataFrame:
    list_csv = glob.glob(path + "*.csv")
    df_energy = pd.DataFrame()
    for file in list_csv:
        df_tmp = pd.read_csv(file, sep=";")
        df_energy = pd.concat([df_energy, df_tmp], axis=0)
    df_energy.rename(columns={"heure": "hour", "libelle_region": "region"}, inplace=True)
    df_energy = df_energy[["date", "hour", "region", target]]
    df_energy = df_energy[df_energy.region != "Grand Est"]
    df_energy["date_hour"] = pd.to_datetime(
        df_energy.date + " " + df_energy.hour, format="%Y-%m-%d %H:%M"
    )
    df_energy = df_energy.sort_values(by=["region", "date_hour"], ascending=True).reset_index(
        drop=True
    )
    df_energy.index = df_energy["date_hour"]
    df_energy = df_energy[["region", "consommation"]]

    if weather:
        df_energy = add_weather(df_energy, weather)

    return df_energy


def enedis(
    path: str = "data/enedis/",
    target: str = "total_energy",
    weather: Dict[str, any] = {
        "path_weather": "data/all_weather/",
        "dynamic_features": ["t", "rr3", "pmer"],
        "cat_features": ["cod_tend"],
        "station_name": "ORLY",
    },
) -> pd.DataFrame:
    list_csv = glob.glob(path + "*.csv")
    df_enedis = pd.DataFrame()
    for file in list_csv:
        df_tmp = pd.read_csv(file)
        df_enedis = pd.concat([df_enedis, df_tmp], axis=0)
    df_enedis.rename(
        columns={
            "horodate": "date",
            "nb_points_soutirage": "soutirage",
            "total_energie_soutiree_wh": target,
            "plage_de_puissance_souscrite": "power",
        },
        inplace=True,
    )

    df_enedis = df_enedis.sort_values(by=["region", "profil", "power", "date"])
    df_enedis.index = pd.to_datetime(df_enedis.date)
    df_enedis = df_enedis[["region", "profil", "power", target, "soutirage"]]

    df_na = df_enedis[df_enedis.total_energy.isna()]
    groups_with_nan = (
        df_na.groupby(["region", "profil", "power"]).apply(lambda x: x.any()).index.tolist()
    )
    df_enedis = df_enedis[
        ~df_enedis.set_index(["region", "profil", "power"]).index.isin(groups_with_nan)
    ]

    df_enedis["power_min"] = df_enedis["power"].str.extract(r"](\d+)-").fillna(0).astype(int)
    df_enedis["power_max"] = (
        df_enedis["power"]
        .str.extract(r"\-(\d+)]")
        .fillna(df_enedis["power"].str.extract(r"<= (\d+)"))
        .astype(int)
    )
    if weather:
        df_enedis = add_weather(df_enedis, weather)

    return df_enedis


def gluonts_dataset(dataset_name: str) -> TrainDatasets:
    return get_dataset(dataset_name)
