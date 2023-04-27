from typing import List

import pandas as pd
from gluonts.model.forecast import Forecast
from gluonts.itertools import Map
from gluonts.dataset import Dataset
from gluonts.dataset.common import (
    ProcessDataEntry,
    TrainDatasets,
    MetaData,
    CategoricalFeatureInfo,
    BasicFeatureInfo,
)
from typing import cast
import numpy as np


def sample_df(forecast: Forecast) -> List[pd.DataFrame]:
    samples = forecast.samples
    ns, h = samples.shape
    dates = pd.date_range(forecast.start_date.to_timestamp(), freq=forecast.freq, periods=h)
    return pd.DataFrame(samples.T, index=dates)


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


def create_ts_with_features(
    df, target, dynamic_real, static_cat, static_real, freq, prediction_length, cardinality
):
    # static features
    if len(static_cat) != 0:
        df["item_id"] = df[static_cat].apply(lambda x: "_".join(x.astype(str)), axis=1)
        df["item_id"] = df["item_id"].astype("category").cat.codes
        static_features_cat = df.groupby(static_cat).sum().reset_index()[static_cat]
        for col in static_features_cat:
            static_features_cat[col] = static_features_cat[col].astype("category").cat.codes
    else:
        df["item_id"] = 0

    if len(static_real) != 0:
        static_features_real = df.groupby(static_real).sum().reset_index()[static_real]
    else:
        static_features_real = static_real

    # dynamic features
    df_pivot = (
        pd.pivot_table(df, values=[target] + dynamic_real, index=df.index, columns=["item_id"])
        .resample(freq)
        .interpolate(method="linear")
    )

    df_train = df_pivot[:-prediction_length]
    train = [
        {
            "target": np.array(df_train[target][i].to_list()),
            "start": df_train.index[0],
            **(
                {
                    "feat_dynamic_real": np.array(
                        [df_train[dynamic_real[j]][i].to_list() for j in range(len(dynamic_real))]
                    )
                }
                if len(dynamic_real) != 0
                else {}
            ),
            **(
                {"feat_static_cat": np.array(static_features_cat)[i]}
                if len(static_cat) != 0
                else {}
            ),
            **(
                {"feat_static_real": np.array(static_features_real)[i]}
                if len(static_real) != 0
                else {}
            ),
            "item_id": str(i),
        }
        for i in df["item_id"].unique()
    ]

    test = [
        {
            "target": np.array(df_pivot[target][i].to_list()),
            "start": df_pivot.index[0],
            **(
                {
                    "feat_dynamic_real": np.array(
                        [df_pivot[dynamic_real[j]][i].to_list() for j in range(len(dynamic_real))]
                    )
                }
                if len(dynamic_real) != 0
                else {}
            ),
            **(
                {"feat_static_cat": np.array(static_features_cat)[i]}
                if len(static_cat) != 0
                else {}
            ),
            **(
                {"feat_static_real": np.array(static_features_real)[i]}
                if len(static_real) != 0
                else {}
            ),
            "item_id": str(i),
        }
        for i in df["item_id"].unique()
    ]

    # gluonts dataset format
    meta = MetaData(
        freq=freq,
        prediction_length=prediction_length,
    )

    meta.feat_dynamic_real = [BasicFeatureInfo(name=name) for name in dynamic_real]

    meta.feat_static_cat = [
        CategoricalFeatureInfo(name=name, cardinality=str(cardinality))
        for name, cardinality in zip(static_cat, cardinality)
    ]

    meta.feat_static_real = [BasicFeatureInfo(name=name) for name in static_real]

    process = ProcessDataEntry(freq, one_dim_target=True, use_timestamp=False)

    train_df = cast(Dataset, Map(process, train))
    test_df = cast(Dataset, Map(process, test))

    dataset = TrainDatasets(metadata=meta, train=train_df, test=test_df)

    return dataset
