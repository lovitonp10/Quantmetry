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
    df_energy.rename(columns={"heure": "hour", "libelle_region": "region"}, inplace=True)
    df_energy = df_energy[["date", "hour", "region", target]]
    df_energy = df_energy[df_energy.region != "Grand Est"]
    df_energy["date_hour"] = pd.to_datetime(
        df_energy.date + " " + df_energy.hour, format="%Y-%m-%d %H:%M"
    )
    df_energy = df_energy.sort_values(by=["date_hour", "region"], ascending=True).reset_index(
        drop=True
    )
    # df_energy.index = df_energy['date_hour']
    # df_energy = df_energy[['region',target]]
    df_energy = pd.pivot_table(df_energy, values=target, index=["date_hour"], columns=["region"])
    df_energy = df_energy.resample("15T").interpolate(method="linear")

    dynamic_feat = []
    static_feat = ["region"]
    return df_energy, dynamic_feat, static_feat


def enedis(path: str = "data/enedis/", target: str = "total_energy"):
    list_csv = glob.glob(path + "*.csv")
    df_enedis = pd.DataFrame()
    for file in list_csv:
        df_tmp = pd.read_csv(file, sep=";")
        df_enedis = pd.concat([df_enedis, df_tmp], axis=0)
    df_enedis.rename(
        columns={"horodate": "date", "total_energie_soutiree_wh": target}, inplace=True
    )
    df_enedis = df_enedis.sort_values(by=["region", "profil", "date", "nb_points_soutirage"])
    df_enedis.index = pd.to_datetime(df_enedis.date)
    df_enedis = df_enedis[["region", "profil", target, "nb_points_soutirage"]]

    # df_enedis['profil2'] = df_enedis['profil']
    # df_enedis['soutirage2'] = df_enedis['nb_points_soutirage']
    # df_enedis['soutirage3'] = df_enedis['nb_points_soutirage']
    # df_enedis['static_real_1'] = np.repeat(np.random.randn(24), 416)
    # df_enedis['static_real_2'] = np.repeat(np.random.randn(24), 416)
    # df_enedis['static_real_3'] = np.repeat(np.random.randn(24), 416)

    dynamic_real = ["nb_points_soutirage"]
    static_cat = ["region", "profil"]
    static_real = []

    return df_enedis, dynamic_real, static_cat, static_real


def gluonts_dataset(dataset_name: str) -> TrainDatasets:
    return get_dataset(dataset_name)
