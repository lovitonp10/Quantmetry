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


def gluonts_dataset(dataset_name: str) -> TrainDatasets:
    return get_dataset(dataset_name)
