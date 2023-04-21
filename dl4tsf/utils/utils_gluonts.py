from typing import List

import pandas as pd
from pandas import Period
import numpy as np
import copy


def sample_df(samples: np.ndarray, start_date: Period, periods, freq) -> List[pd.DataFrame]:
    # samples = forecast.samples
    # ns, h = samples.shape
    dates = pd.date_range(start_date.to_timestamp(), freq=freq, periods=periods)
    return pd.DataFrame(samples.T, index=dates)


def transform_huggingface_to_pandas(gluonts_dataset, freq: str):
    df_pandas = pd.DataFrame()
    periods = len(gluonts_dataset[0]["target"])
    i = 0

    for item in list(gluonts_dataset)[:10]:
        print(i)
        i = i + 1
        df_tmp = pd.DataFrame()

        df_tmp["target"] = item["target"]
        df_tmp["date"] = pd.date_range(
            start=item["start"].to_timestamp(), periods=periods, freq=freq
        )
        df_tmp["item_id"] = item["item_id"]
        df_tmp["feat_static_cat"] = (
            item["feat_static_cat"][0]
            if isinstance(item["feat_static_cat"], list) and len(item["feat_static_cat"]) == 1
            else item["feat_static_cat"]
        )
        df_tmp["feat_dynamic_real"] = (
            item["feat_dynamic_real"][0]
            if isinstance(item["feat_dynamic_real"], list) and len(item["feat_dynamic_real"]) == 1
            else item["feat_dynamic_real"]
        )
        df_pandas = pd.concat([df_pandas, df_tmp], axis=0)
    return df_pandas


def transform_huggingface_to_dict(gluonts_dataset, freq: str):
    periods = len(gluonts_dataset[0]["target"])
    i = 0

    list_dataset = []
    for item in list(gluonts_dataset)[:10]:
        print(i)
        i = i + 1
        df_tmp = pd.DataFrame()
        df_tmp["target"] = item["target"]
        df_tmp["date"] = pd.date_range(
            start=item["start"].to_timestamp(), periods=periods, freq=freq
        )
        df_tmp["item_id"] = item["item_id"]
        df_tmp["feat_static_cat"] = (
            item["feat_static_cat"][0]
            if isinstance(item["feat_static_cat"], list) and len(item["feat_static_cat"]) == 1
            else item["feat_static_cat"]
        )
        df_tmp["feat_dynamic_real"] = (
            item["feat_dynamic_real"][0]
            if isinstance(item["feat_dynamic_real"], list) and len(item["feat_dynamic_real"]) == 1
            else item["feat_dynamic_real"]
        )
        dict_dataset = {}
        dict_dataset["item_id"] = item["item_id"]
        dict_dataset["df"] = df_tmp.copy()

        list_dataset.append(copy.deepcopy(dict_dataset))
    return list_dataset


def get_test_length(freq: str, test_length: str) -> int:
    """Calculates the number of of rows for the test set give time frequency and test length.


    Parameters
    ----------
    freq : str
        A string representing the frequency of the dataset in the format 'Xunit'.
    test_length : str
        A string representing the desired duration of a the test test. In the format 'Xunit'.

    Returns
    -------
    int
        The number of rows of the test set.
    """
    freq_minutes = pd.Timedelta(freq).total_seconds() / 60
    test_length_minutes = pd.Timedelta(test_length).total_seconds() / 60
    return int(test_length_minutes / freq_minutes)
