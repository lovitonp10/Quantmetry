import glob
import pandas as pd
from gluonts.dataset.common import TrainDatasets
from load.load_data_aifluence import Aifluence
from gluonts.dataset.repository.datasets import get_dataset as get_gluonts_dataset
from datasets import load_dataset as get_huggingface_dataset
from functools import partial
from utils.custom_objects_pydantic import HuggingFaceDataset
from domain.transformations_pd import transform_start_field
from load.load_exo_data import add_weather
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def climate(
    path: str = "data/climate_delhi/",
    target: str = "mean_temp",
    weather: Dict[str, any] = {
        "path_weather": "data/all_weather/",
        "dynamic_features": ["temperature", "rainfall", "pressure"],
        "cat_features": ["barometric_trend"],
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
        "dynamic_features": ["temperature", "rainfall", "pressure"],
        "cat_features": ["barometric_trend"],
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
        "dynamic_features": ["temperature", "rainfall", "pressure"],
        "cat_features": ["barometric_trend"],
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


def aifluence_public_histo_vrf(
    path: str = "data/idf_mobilites/",
    target: str = "VALD_TOTAL",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    p_data_station: float = 0.9,
    weather: Dict[str, any] = {
        "path_weather": "data/all_weather/",
        "dynamic_features": ["temperature", "rainfall", "pressure"],
        "cat_features": ["barometric_trend"],
        "station_name": "ORLY",
    },
) -> pd.DataFrame:
    """Read a folder for load data from file and save it to a fataframe

    Parameters
    ----------
    path : str, optional
        loaded files, by default "data/idf_mobilites/"
    target : str, optional
        target features, by default "VALD_TOTAL"
    p_data_station : float
        proportion of data for each station, by default 90%
    start_date : Optional[str], optional
        starting date of time series, by default None
    end_date : Optional[str], optional
        ebding date of time series, by default None
    weather : Dict[str, any], optional
        weather feature for the dataset, by default {
            "path_weather": "data/all_weather/",
            "dynamic_features": ["temperature", "rainfall", "pressure"],
            "cat_features": ["barometric_trend"], "station_name": "ORLY", }

    Returns
    -------
    pd.DataFrame
        public data frame from IDF-mobilités
    """
    aifluence = Aifluence(path)
    logger.info("Loading Data")
    df_load = aifluence.load_validations()
    df_load = aifluence.change_column_validations(df_load)
    df_temp = df_load.drop(
        columns=["CODE_STIF_TRNS", "CODE_STIF_RES", "CODE_STIF_ARRET", "ID_REFA_LDA"]
    )

    logger.info("Preprocess Data")
    df_fusion = aifluence.preprocess_validation_titre(df_temp)
    df_aifluence = aifluence.preprocess_station(df_fusion, p_data_station)

    if weather:
        df_aifluence = add_weather(df_aifluence, weather)

    df_rename = df_aifluence.rename_axis("DATE")
    df_aifluence = df_rename.sort_values(by=["STATION", "DATE"])
    df_aifluence = df_aifluence.rename_axis(None)
    df_aifluence = aifluence.cut_start_end_ts(df_aifluence, start=start_date, end=end_date)
    return df_aifluence


def gluonts_dataset(dataset_name: str) -> TrainDatasets:
    return get_gluonts_dataset(dataset_name)


def huggingface_dataset(
    repository_name: str,
    dataset_name: str,
    freq: str,
    target: str,
    weather=None,
) -> HuggingFaceDataset:
    dataset = get_huggingface_dataset(repository_name, dataset_name)
    dataset["train"].set_transform(partial(transform_start_field, freq=freq))
    dataset["test"].set_transform(partial(transform_start_field, freq=freq))
    dataset = HuggingFaceDataset(train=dataset["train"], test=dataset["test"])

    if weather:
        dataset = add_weather(dataset, weather)

    return dataset
