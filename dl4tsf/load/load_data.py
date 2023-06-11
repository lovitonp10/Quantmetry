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
from typing import Dict, List, Optional
import logging
from load.load_data_enedis import Enedis
from utils.utils_gluonts import generate_item_ids_static_features

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
    prediction_length: int = 7,
    name_feats: Dict[str, List[str]] = None,
    weather: Dict[str, any] = {
        "path_weather": "data/all_weather/",
        "dynamic_features": ["temperature", "rainfall", "pressure"],
        "cat_features": ["barometric_trend"],
        "station_name": "ORLY",
    },
) -> pd.DataFrame:
    logger.info("Loading Data")
    enedis = Enedis(path, target)
    enedis.load_data()

    logger.info("Preprocess Data")
    df_enedis = enedis.get_preprocessed_data()

    df_enedis["item_id"] = generate_item_ids_static_features(
        df=df_enedis, key_columns=name_feats["feat_static_cat"] + name_feats["feat_static_real"]
    )

    if weather:
        df_enedis, df_forecast = add_weather(df_enedis, weather, prediction_length)
        # If you have dynamic_feat (known in the future):
        # df_forecast = pd.merge(forecast_dynamic_feat, df_forecast,
        # left_index=True, right_index=True, how="left")
        return df_enedis, df_forecast

    # Dummy generated dynamic_cat
    # import numpy as np
    # df_enedis['test_dynamic_cat'] = np.random.randint(0, 4, size=865387)

    return df_enedis, None


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
    logger.info("Loading Data")
    aifluence = Aifluence(path)
    aifluence.load_validations()

    logger.info("Preprocess Data")
    df_aifluence = aifluence.get_preprocessed_data(p_data_station=p_data_station)

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
