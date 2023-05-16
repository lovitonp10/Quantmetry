import glob

import pandas as pd
from gluonts.dataset.common import TrainDatasets
from gluonts.dataset.repository.datasets import get_dataset as get_gluonts_dataset
from datasets import load_dataset as get_huggingface_dataset
from functools import partial
from utils.custom_objects_pydantic import HuggingFaceDataset
from domain.transformations_pd import transform_start_field


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
    return get_gluonts_dataset(dataset_name)


def huggingface_dataset(
    repository_name: str, dataset_name: str, freq: str, target: str
) -> HuggingFaceDataset:
    dataset = get_huggingface_dataset(repository_name, dataset_name)
    dataset["train"].set_transform(partial(transform_start_field, freq=freq))
    dataset["test"].set_transform(partial(transform_start_field, freq=freq))
    dataset = HuggingFaceDataset(train=dataset["train"], test=dataset["test"])
    return dataset


def enedis(
    path: str = "data/enedis/",
    target: str = "total_energy",
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

    # Dummy generated dynamic_cat
    # import numpy as np
    # df_enedis['test_dynamic_cat'] = np.random.randint(0, 4, size=865387)

    return df_enedis
