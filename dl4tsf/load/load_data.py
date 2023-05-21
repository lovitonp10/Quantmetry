import glob
import logging
import pandas as pd
from gluonts.dataset.common import TrainDatasets
from gluonts.dataset.repository.datasets import get_dataset
from load.load_data_aifluence import load_validations
from load.load_exo_data import add_weather
from typing import Dict

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
    target: str = "nb_vald_total",
    weather: Dict[str, any] = {
        "path_weather": "data/all_weather/",
        "dynamic_features": ["temperature", "rainfall", "pressure"],
        "cat_features": ["barometric_trend"],
        "station_name": "ORLY",
    },
) -> pd.DataFrame:

    """load a json file for download data from url and save it to local

    Args:
        path (path of json file): loaded json
        target: target features
        weather: weather feature for the dataset
    Returns:
        df (DataFrame): public data frame from IDF-mobilités
    """

    # PART 1 : Read data on the website "data.iledefrance-mobilites.fr"
    df_load = load_validations(path)
    df_load.replace(to_replace="Moins de 5", value=3, inplace=True)
    df_load["NB_VALD"] = df_load["NB_VALD"].astype(int)
    df_load = df_load[df_load["ID_REFA_LDA"] != "?"]

    df_load.rename(
        columns={
            "LIBELLE_ARRET": "arret",
            "JOUR": "date",
            "CODE_STIF_TRNS": "code_transport",
            "CODE_STIF_RES": "code_reseau",
            "CODE_STIF_ARRET": "code_arret",
            "ID_REFA_LDA": "id_arret",
            "CATEGORIE_TITRE": "cat_titre",
            "NB_VALD": "nb_vald",
            "NB_VALD_TOTAL": "nb_vald_total",
        },
        inplace=True,
    )
    df_load["cat_titre"] = df_load["cat_titre"].replace("?", "INCONNU")

    df_validations = df_load.groupby(
        ["date", "arret", "code_transport", "code_reseau", "code_arret", "id_arret"],
        as_index=False,
    ).agg({"nb_vald": "sum"})
    df_validations.rename(columns={"nb_vald": "nb_vald_total"}, inplace=True)

    df_aifluence = df_validations.copy()

    for titre in df_load["cat_titre"].unique():
        df_temp = df_load[df_load["cat_titre"] == titre].rename(
            columns={"nb_vald": "nb_vald_" + str(titre)}
        )
        df_aifluence = df_aifluence.merge(
            df_temp[
                [
                    "date",
                    "arret",
                    "code_transport",
                    "code_reseau",
                    "code_arret",
                    "id_arret",
                    "nb_vald_" + str(titre),
                ]
            ],
            on=["date", "arret", "code_transport", "code_reseau", "code_arret", "id_arret"],
            how="left",
        )

    df_aifluence.sort_values(by=["arret", "date"])
    df_aifluence.index = df_aifluence["date"]
    df_aifluence = df_aifluence.drop(["date"], axis=1)

    # if weather:
    #     df_aifluence = add_weather(df_aifluence, weather)

    return df_aifluence


def gluonts_dataset(dataset_name: str) -> TrainDatasets:
    return get_dataset(dataset_name)
