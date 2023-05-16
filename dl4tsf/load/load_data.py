import glob
import os
import tempfile
import logging
import zipfile
import wget
import pandas as pd
from collections.abc import Mapping
from gluonts.dataset.common import TrainDatasets
from gluonts.dataset.repository.datasets import get_dataset
from load.load_exo_data import add_weather
from typing import Dict

logger = logging.getLogger(__name__)


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


def aifluence_public_histo_vrf(
    path: str = "data/idf_mobilites/",
    target: str = "nb_validation",
    weather: Dict[str, any] = {
        "path_weather": "data/all_weather/",
        "dynamic_features": ["t", "rr3", "pmer"],
        "cat_features": ["cod_tend"],
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

    # PART 1 : Download data on the website "data.iledefrance-mobilites.fr"
    prefix = "https://data.iledefrance-mobilites.fr/explore/dataset/histo-validations/files/"
    list_df = []  # list of datas download
    files = "histo-validations-reseau-ferre.json"  # Json file with url for download data
    history = pd.read_json(path + files, typ="series")  # Read  the json file

    temp_dir = tempfile.TemporaryDirectory(prefix="AIFL_")  # Create a temporal link
    logger.info(f"Creation of temporary directory: {temp_dir.name}")

    for dict_year in history:  # Read each link by years
        fields = dict_year["fields"]  # Save the information of url
        year = int(fields["annee"].split(" ")[-1])  # Save the year of the url
        for name_sem, semester in fields.items():  # Read each information in url
            if isinstance(semester, Mapping):
                if (year < 2019) or (
                    year == 2019 and name_sem == "semestre_1"
                ):  # Takes into account the particularity of the years after 2019

                    url = prefix + semester["id"] + "/download/"  # Create the complete name of url
                    logger.info(f"Importing file {semester['filename']} from: \n {url}")
                    path_file = os.path.join(temp_dir.name, semester["filename"])

                    try:
                        wget.download(url=url, out=path_file, bar=None)  # Download the file
                    except Exception as e:
                        logger.warning(f"Could not download file {url}")
                        logger.error(e)

                    # PART 2 : Unzip the file for take the data
                    zip_ref = zipfile.ZipFile(path_file, "r")
                    file_nb_fer = [
                        file for file in zip_ref.namelist() if "nb_fer" in file.lower()
                    ][0]
                    zip_ref.extract(file_nb_fer, path=temp_dir.name)

                    # All files are ".txt" format with "\t" separation
                    # Except 2015 files are ".csv" format with "";" separation
                    if year == 2015:
                        sep = ";"
                    else:
                        sep = "\t"

                    df_temp = pd.read_csv(os.path.join(temp_dir.name, file_nb_fer), sep=sep)
                    list_df.append(df_temp)
    df_aifluence = pd.concat(list_df)
    temp_dir.cleanup()

    # PART 3 : Modification on the final DataFrame
    df_aifluence.rename(
        columns={
            "LIBELLE_ARRET": "arret",
            "JOUR": "date",
            "CATEGORIE_TITRE": "cat_titre",
            "CODE_STIF_TRNS": "code_transport",
            "CODE_STIF_RES": "code_reseau",
            "CODE_STIF_ARRET": "code_arret",
            "ID_REFA_LDA": "id_arret",
            "NB_VALD": target,
        },
        inplace=True,
    )
    df_aifluence.index = pd.to_datetime(df_aifluence.date, dayfirst=True)
    df_aifluence = df_aifluence[
        ["cat_titre", "arret", "code_transport", "code_reseau", "code_arret", "id_arret", target]
    ]

    df_na = df_aifluence[df_aifluence.nb_validation.isna()]
    groups_with_nan = (
        df_na.groupby(
            ["cat_titre", "arret", "code_transport", "code_reseau", "code_arret", "id_arret"],
            group_keys=False,
        )
        .apply(lambda x: x.any())
        .index.tolist()
    )
    df_aifluence = df_aifluence[
        ~df_aifluence.set_index(
            ["cat_titre", "arret", "code_transport", "code_reseau", "code_arret", "id_arret"]
        ).index.isin(groups_with_nan)
    ]

    return df_aifluence


def gluonts_dataset(dataset_name: str) -> TrainDatasets:
    return get_dataset(dataset_name)
