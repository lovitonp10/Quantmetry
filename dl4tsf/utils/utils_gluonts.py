import logging
import shutil
from functools import partial
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, cast

import configs
import datasets
import numpy as np
import pandas as pd
import pydantic
from gluonts import json
from gluonts.dataset import Dataset, DatasetWriter
from gluonts.dataset.common import (
    BasicFeatureInfo,
    CategoricalFeatureInfo,
    ProcessDataEntry,
)
from gluonts.itertools import Map
from pandas import Period
from utils.custom_objects_pydantic import HuggingFaceDataset

logger = logging.getLogger(__name__)


class MetaData(pydantic.BaseModel):
    freq: str = pydantic.Field(..., alias="time_granularity")  # type: ignore
    target: Optional[BasicFeatureInfo] = None

    feat_static_cat: List[CategoricalFeatureInfo] = []
    feat_static_real: List[BasicFeatureInfo] = []
    feat_dynamic_real: List[BasicFeatureInfo] = []
    feat_dynamic_cat: List[CategoricalFeatureInfo] = []
    past_feat_dynamic_real: List[BasicFeatureInfo] = []

    prediction_length: Optional[int] = None

    class Config(pydantic.BaseConfig):
        allow_population_by_field_name = True


class TrainDatasets(NamedTuple):
    """
    A dataset containing two subsets, one to be used for training purposes, and
    the other for testing purposes, as well as metadata.
    """

    metadata: MetaData
    train: Dataset
    validation: Optional[Dataset] = None
    test: Optional[Dataset] = None

    def save(
        self,
        path_str: str,
        writer: DatasetWriter,
        overwrite=False,
    ) -> None:
        """
        Saves an TrainDatasets object to a JSON Lines file.

        Parameters
        ----------
        path_str
            Where to save the dataset.
        overwrite
            Whether to delete previous version in this folder.
        """
        path = Path(path_str)

        if overwrite:
            shutil.rmtree(path, ignore_errors=True)

        path.mkdir(parents=True)
        with open(path / "metadata.json", "wb") as out_file:
            json.bdump(self.metadata.dict(), out_file, nl=True)

        train = path / "train"
        train.mkdir(parents=True)
        writer.write_to_folder(self.train, train)

        if self.validation is not None:
            validation = path / "validation"
            validation.mkdir(parents=True)
            writer.write_to_folder(self.validation, validation)

        if self.test is not None:
            test = path / "test"
            test.mkdir(parents=True)
            writer.write_to_folder(self.test, test)


def sample_df(
    samples: np.ndarray,
    start_date: Period,
    periods: int,
    freq: str,
    ts_length: int,
    pred_length: int,
    validation: bool,
) -> List[pd.DataFrame]:
    """Creates a list of DataFrame samples.

    Parameters
    ----------
    samples : np.ndarray
        Array of samples.
    start_date : Period
        The start date of the samples.
    periods : int
        The number of periods.
    freq : str
        The frequency of the samples. It should be a valid pandas frequency string ('1H', '1D',..).
    ts_length : int
        The length of the time series.
    pred_length : int
        The prediction length.
    validation : bool
        Specifies whether the samples are for validation.

    Returns
    -------
    List[pd.DataFrame]
        A list of DataFrame samples.
    """
    # samples = forecast.samples
    # ns, h = samples.shape
    if validation is True:
        ts_length = ts_length - pred_length
    dates = pd.date_range(start_date.to_timestamp(), freq=freq, periods=periods).shift(ts_length)
    return pd.DataFrame(samples.T, index=dates)


def get_ts_length(df_pandas: pd.DataFrame) -> int:
    """Returns the length of the time series.

    Parameters
    ----------
    df_pandas : pd.DataFrame
        DataFrame representing the time series.

    Returns
    -------
    int
        The length of the time series.
    """
    ts_length = df_pandas.shape[0]
    return ts_length


def get_mean_metrics(metrics: dict) -> dict:
    for key, value in metrics.items():
        metrics[key] = np.mean(value)
    return metrics


def transform_huggingface_to_dict(dataset: pd.DataFrame, freq: str):
    """Transforms a HuggingFace dataset to a list of pandas DataFrames.

    Parameters
    ----------
    dataset : pd.DataFrame
        HuggingFace dataset to be transformed.
    freq : str
        Frequency string representing the frequency of the time series.

    Returns
    -------
    List[pd.DataFrame]
        Transformed list of pandas DataFrames.
    """
    list_dataset = []
    for item in list(dataset):
        list_dataset.append(pd.DataFrame(item["target"]))
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


def create_ts_with_features(
    dataset_type: str,
    df: pd.DataFrame,
    target: str,
    name_feats: configs.Feats,
    freq: str,
    test_length_rows: int,
    prediction_length: int,
    static_cardinality: List[int],
    dynamic_cardinality: List[int],
    df_forecast: pd.DataFrame,
) -> TrainDatasets:
    """
    Create time series datasets with different features.

    Parameters
    ----------
    dataset_type : str
        Type of dataset to create (e.g., "gluonts", "hugging_face").
    df : pd.DataFrame
        Input DataFrame containing the time series data.
    target : str
        Name of the target variable column in the DataFrame.
    name_feats : configs.Feats
        List of names of columns.
    freq : str
        Frequency of the time series data (e.g., "D" for daily, "H" for hourly).
    test_length_rows : int
        Number of rows to use for the test dataset.
    prediction_length : int
        Length of the prediction horizon.
    static_cardinality : List[int]
        List of cardinalities for the static categorical features.
    dynamic_cardinality : List[int]
        List of cardinalities for the dynamic categorical features.
    df_forecast : pd.DataFrame
        DataFrame containing the forecasted values for the dynamic features.

    Returns
    -------
    TrainDatasets
        The created dataset in gluonts or hugging face format.

    """
    dynamic_real = name_feats.feat_dynamic_real
    static_cat = name_feats.feat_static_cat
    static_real = name_feats.feat_static_real
    past_dynamic_real = name_feats.past_feat_dynamic_real
    dynamic_cat = name_feats.feat_dynamic_cat

    logger.info("get static features")
    df_static_features = df.drop_duplicates(subset=["item_id"]).set_index("item_id")[
        static_cat + static_real
    ]
    for col in static_cat + static_real:
        df_static_features[col] = df_static_features[col].astype("category").cat.codes

    logger.info("Resample Pivot table")
    df_pivot = pivot_df(
        df=df,
        cols=[target] + dynamic_real + past_dynamic_real + dynamic_cat,
        index_col="item_id",
        freq=freq,
    )

    logger.info("Add dynamic features")
    df_dynamic_feat_forecast = pivot_df(
        df=df_forecast,
        cols=list(df_forecast.columns.difference(["item_id"])),
        index_col="item_id",
        freq=freq,
    )

    logger.info("Create train/val/test dfs")
    train, val, test = train_val_test_split(
        dataset_type=dataset_type,
        df=df,
        df_pivot=df_pivot,
        target=target,
        name_feats=name_feats,
        df_static_features=df_static_features,
        test_length_rows=test_length_rows,
        prediction_length=prediction_length,
        df_dynamic_feat_forecast=df_dynamic_feat_forecast,
    )

    if dataset_type == "gluonts":
        # gluonts dataset format
        dataset = gluonts_format(
            df_train=train,
            df_validation=val,
            df_test=test,
            name_feats=name_feats,
            static_cardinality=static_cardinality,
            dynamic_cardinality=dynamic_cardinality,
            freq=freq,
            prediction_length=prediction_length,
        )

    elif dataset_type == "hugging_face":
        dataset = hugging_face_format(
            df_train=train,
            df_validation=val,
            df_test=test,
            freq=freq,
        )

    return dataset


def gluonts_format(
    df_train: pd.DataFrame,
    df_validation: pd.DataFrame,
    df_test: pd.DataFrame,
    name_feats: configs.Feats,
    static_cardinality: List[int],
    dynamic_cardinality: List[int],
    freq: str,
    prediction_length: int,
) -> TrainDatasets:
    """Formats the data into a GluonTS-compatible format.

    Parameters
    ----------
    df_train : pd.DataFrame
        DataFrame containing the training data.
    df_validation : pd.DataFrame
        DataFrame containing the validation data.
    df_test : pd.DataFrame
        DataFrame containing the test data.
    name_feats : configs.Feats
        An object specifying the names of different features used in the dataset.
    static_cardinality : List[int]
        A list of integers representing the cardinality of each static categorical feature.
    dynamic_cardinality : List[int]
        A list of integers representing the cardinality of each dynamic categorical feature.
    freq : str
        The frequency of the time series data. It should be a valid pandas frequency string.
    prediction_length : int
        The number of time steps to predict into the future.

    Returns
    -------
    TrainDatasets
        A TrainDatasets object containing the formatted train, validation, and test datasets.
    """
    dynamic_real = name_feats.feat_dynamic_real
    static_cat = name_feats.feat_static_cat
    static_real = name_feats.feat_static_real
    past_dynamic_real = name_feats.past_feat_dynamic_real
    dynamic_cat = name_feats.feat_dynamic_cat

    meta = MetaData(
        freq=freq,
        prediction_length=prediction_length,
    )

    meta.feat_dynamic_real = [BasicFeatureInfo(name=name) for name in dynamic_real]
    meta.past_feat_dynamic_real = [BasicFeatureInfo(name=name) for name in past_dynamic_real]

    meta.feat_static_cat = [
        CategoricalFeatureInfo(name=name, cardinality=str(cardinality))
        for name, cardinality in zip(static_cat, static_cardinality)
    ]

    meta.feat_static_real = [BasicFeatureInfo(name=name) for name in static_real]

    meta.feat_dynamic_cat = [
        CategoricalFeatureInfo(name=name, cardinality=str(cardinality))
        for name, cardinality in zip(dynamic_cat, dynamic_cardinality)
    ]

    process = ProcessDataEntry(freq, one_dim_target=True, use_timestamp=False)
    train = cast(Dataset, Map(process, df_train))
    validation = cast(Dataset, Map(process, df_validation))
    test = cast(Dataset, Map(process, df_test))

    dataset = TrainDatasets(metadata=meta, train=train, validation=validation, test=test)
    return dataset


def hugging_face_format(
    df_train: pd.DataFrame,
    df_validation: pd.DataFrame,
    df_test: pd.DataFrame,
    freq: str,
) -> HuggingFaceDataset:
    """Formats the data into a Hugging Face-compatible format.

    Parameters
    ----------
    df_train : pd.DataFrame
        DataFrame containing the training data.
    df_validation : pd.DataFrame
        DataFrame containing the validation data.
    df_test : pd.DataFrame
        DataFrame containing the test data.
    freq : str
        The frequency of the time series data. It should be a valid pandas frequency string.

    Returns
    -------
    HuggingFaceDataset
        A HuggingFaceDataset object containing the formatted train, validation, and test datasets.
    """
    train = pd.DataFrame(df_train)
    validation = pd.DataFrame(df_validation)
    test = pd.DataFrame(df_test)

    train_dataset = datasets.Dataset.from_dict(train)
    validation_dataset = datasets.Dataset.from_dict(validation)
    test_dataset = datasets.Dataset.from_dict(test)
    # dataset = datasets.DatasetDict({"train":train_dataset,"test":test_dataset})

    train_dataset.set_transform(partial(transform_start_field, freq=freq))
    validation_dataset.set_transform(partial(transform_start_field, freq=freq))
    test_dataset.set_transform(partial(transform_start_field, freq=freq))
    dataset = HuggingFaceDataset
    dataset.train = train_dataset
    dataset.validation = validation_dataset
    dataset.test = test_dataset
    return dataset


def transform_start_field(
    batch: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    """Transforms the 'start' field in the batch DataFrame to the specified frequency.

    Parameters
    ----------
    batch : pd.DataFrame
        DataFrame containing the batch of data.
    freq : str
        The frequency to which the 'start' field should be transformed (e.g., '1H', '1D').

    Returns
    -------
    pd.DataFrame
        The transformed batch DataFrame with the 'start' field updated to the specified frequency.
    """
    batch["start"] = [pd.Period(date, freq) for date in batch["start"]]
    return batch


def generate_item_ids_static_features(
    df: pd.DataFrame,
    key_columns: List[str],
) -> pd.Series:
    """Creates item IDs based on the key columns in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    key_columns : List[str]
        List of columns that must be taken to create item_it

    Returns
    -------
    pd.Series
        A pandas Series containing the item IDs.
    """

    df_u_items = (
        df[key_columns]
        .drop_duplicates(ignore_index=True)
        .reset_index()
        .rename(columns={"index": "item_id"})
    )
    lst_item_ids = df.merge(df_u_items, how="left", on=key_columns)["item_id"].values
    return lst_item_ids


def pivot_df(
    df: pd.DataFrame,
    cols: List[str],
    index_col: str,
    freq: str,
) -> pd.DataFrame:
    """Resamples the DataFrame with the specified frequency and fills missing values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    cols: List[str]
        List of columns to be pivoted
    index_col: str
        column name of the index
    freq : str
        The frequency to which the DataFrame should be resampled (e.g., '1H', '1D').

    Returns
    -------
    pd.DataFrame
        The resampled DataFrame with filled missing values.
    """
    df_pivot = (
        pd.pivot_table(
            df,
            values=cols,
            index=df.index,
            columns=[index_col],
        )
        .resample(freq)
        .interpolate(method="linear")
    )

    # The method interpolate doesn't fillna on the first row
    # We use Backward Fill to fill the NaN values with the next valid observation
    df_pivot.fillna(method="bfill", inplace=True)

    # if df_pivot.iloc[0].isna().any():
    #     df_pivot = df_pivot.drop(labels=df_pivot.index[0], axis=0)

    # dynamic cat as type int
    # for feat in dynamic_cat:
    #     df_pivot[feat] = df_pivot[feat].astype(int)

    return df_pivot


def train_val_test_split(
    dataset_type: str,
    df: pd.DataFrame,
    df_pivot: pd.DataFrame,
    target: str,
    name_feats: configs.Feats,
    df_static_features: pd.DataFrame,
    test_length_rows: int,
    prediction_length: int,
    df_dynamic_feat_forecast: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Split the data into train, validation, and test sets for time series forecasting.

    Parameters
    ----------
    dataset_type : str
        The type of dataset being created ("gluonts", "hugging_face").
    df : pd.DataFrame
        The input DataFrame containing the time series data.
    df_pivot : pd.DataFrame
        The pivot DataFrame containing the transformed time series data.
    target : str
        The name of the target variable column.
    name_feats : configs.Feats
        List of names of columns.
    df_static_features : pd.DataFrame
        The DataFrame containing static features.
    test_length_rows : int
        The number of rows to use for the test dataset.
    prediction_length : int
        The length of the prediction horizon.
    df_dynamic_feat_forecast : pd.DataFrame
        The DataFrame containing the forecasted values for the dynamic features.

    Returns
    -------
    List[Dict[str, Any]]
        The split train, validation, or test datasets.

    """

    # train
    df_train = df_pivot[: -test_length_rows * 2].copy()
    item_ids = df["item_id"].unique()
    train = create_dict_dataset(
        target=df_train[target],
        start_date=df_train.index[0],
        df_feat_dynamic_real=df_train[name_feats.feat_dynamic_real],
        df_feat_static_cat=df_static_features[name_feats.feat_static_cat],
        df_feat_static_real=df_static_features[name_feats.feat_static_real],
        df_past_feat_dynamic_real=df_train[name_feats.past_feat_dynamic_real],
        df_feat_dynamic_cat=df_train[name_feats.feat_dynamic_cat],
        item_ids=item_ids,
    )

    # validation
    df_val = df_pivot[: -test_length_rows * 2].copy()
    item_ids = df["item_id"].unique()
    val = create_dict_dataset(
        target=df_val[target],
        start_date=df_val.index[0],
        df_feat_dynamic_real=df_val[name_feats.feat_dynamic_real],
        df_feat_static_cat=df_static_features[name_feats.feat_static_cat],
        df_feat_static_real=df_static_features[name_feats.feat_static_real],
        df_past_feat_dynamic_real=df_val[name_feats.past_feat_dynamic_real],
        df_feat_dynamic_cat=df_val[name_feats.feat_dynamic_cat],
        item_ids=item_ids,
    )

    # test
    df_test = df_pivot.copy()
    df_feat_dynamic_real = pd.concat(
        [
            df_test[name_feats.feat_dynamic_real],
            df_dynamic_feat_forecast[name_feats.feat_dynamic_real],
        ],
        axis=0,
    )
    df_feat_dynamic_cat = pd.concat(
        [
            df_test[name_feats.feat_dynamic_cat],
            df_dynamic_feat_forecast[name_feats.feat_dynamic_cat],
        ],
        axis=0,
    )

    df_feat_static_cat = df_static_features[name_feats.feat_static_cat]
    df_feat_static_real = df_static_features[name_feats.feat_static_real]
    target = add_target_forecast(
        df_pivot=df_test,
        target=target,
        prediction_length=prediction_length,
        item_id=df["item_id"],
        dataset_type=dataset_type,
    )

    df_past_feat_dynamic_real = add_past_forecast(
        df_pivot=df_test,
        past_dynamic_real=name_feats.past_feat_dynamic_real,
        prediction_length=prediction_length,
        item_id=df["item_id"],
        dataset_type=dataset_type,
    )

    test = create_dict_dataset(
        target=target,
        start_date=df_test.index[0],
        df_feat_dynamic_real=df_feat_dynamic_real,
        df_feat_static_cat=df_feat_static_cat,
        df_feat_static_real=df_feat_static_real,
        df_past_feat_dynamic_real=df_past_feat_dynamic_real,
        df_feat_dynamic_cat=df_feat_dynamic_cat,
        item_ids=df["item_id"].unique(),
    )

    return train, val, test


def create_dict_dataset(
    target: str,
    start_date: Any,
    df_feat_dynamic_real: pd.DataFrame,
    df_feat_static_real: pd.DataFrame,
    df_feat_static_cat: pd.DataFrame,
    df_past_feat_dynamic_real: pd.DataFrame,
    df_feat_dynamic_cat: pd.DataFrame,
    item_ids: List[int],
) -> List[Dict[str, Any]]:
    """
    Create a dictionary-based dataset for time series forecasting.

    Parameters
    ----------
    target : str
        The name of the target variable column.
    start_date : Datetime
        the first starting date.
    df_feat_dynamic_real : pd.DataFrame
        The DataFrame containing dynamic real-valued features.
    df_feat_static_real : pd.DataFrame
        The DataFrame containing static real features.
    df_feat_static_cat : pd.DataFrame
        The DataFrame containing static categorical features.
    df_past_feat_dynamic_real : pd.DataFrame
        The DataFrame containing past values of dynamic real-valued features.
    df_feat_dynamic_cat : pd.DataFrame
        The DataFrame containing dynamic categorical features.
    item_ids : pd.DataFrame
        The input DataFrame containing the item_ids.

    Returns
    -------
    List[Dict[str, Any]]
        The dictionary-based dataset.

    """
    data = [
        {
            "target": np.array(target[i].to_list()),
            "start": start_date,
            **(
                {
                    "feat_dynamic_real": np.array(
                        [
                            df_feat_dynamic_real[feat][i].to_list()
                            for feat in df_feat_dynamic_real.columns.get_level_values(0).unique()
                        ]
                    )
                }
                if len(df_feat_dynamic_real.columns) != 0
                else {}
            ),
            **(
                {"feat_static_cat": np.array(df_feat_static_cat)[i]}
                if df_feat_static_cat.shape[1] != 0
                else {}
            ),
            **(
                {
                    "feat_static_real": np.array(df_feat_static_real)[i]
                }  # np.array(df[df["item_id"] == i][static_real].iloc[0])}
                if df_feat_static_real.shape[1] != 0
                else {}
            ),
            **(
                {
                    "past_feat_dynamic_real": np.array(
                        [
                            df_past_feat_dynamic_real[feat][i].to_list()
                            for feat in df_past_feat_dynamic_real.columns.get_level_values(
                                0
                            ).unique()
                        ]
                    )
                }
                if df_past_feat_dynamic_real.shape[1] != 0
                else {}
            ),
            **(
                {
                    "feat_dynamic_cat": np.array(
                        [
                            df_feat_dynamic_cat[feat][i].to_list()
                            for feat in df_feat_dynamic_cat.columns.get_level_values(0).unique()
                        ]
                    )
                }
                if df_feat_dynamic_cat.shape[1] != 0
                else {}
            ),
            "item_id": str(i),
        }
        for i in item_ids
    ]

    return data


def add_target_forecast(
    df_pivot: pd.DataFrame,
    target: str,
    prediction_length: int,
    item_id: pd.Series,
    dataset_type: str,
) -> pd.DataFrame:
    """Add 0 to target for prediction length forecast.

    Parameters
    ----------
    df_pivot : DataFrame
        The pivot DataFrame containing the data.
    target : str
        The name of the target variable.
    prediction_length : int
        The length of the forecast.
    item_id : Series
        Series containing the item IDs.

    Returns
    -------
    DataFrame
        The DataFrame with the added 0 to target.
    """
    if dataset_type == "hugging_face":
        return df_pivot[target]
    elif dataset_type == "gluonts":
        target_forecast = pd.DataFrame()
        num_item_id = len(item_id.unique())
        for feat in [target]:
            index = pd.MultiIndex.from_tuples([(feat, i) for i in range(num_item_id)])
            target_data = np.array(list(repeat(df_pivot[feat], num_item_id))).T
            target_data = pd.DataFrame(
                np.zeros((prediction_length, df_pivot[feat].shape[1])), columns=index
            )
            target_forecast = pd.concat([target_forecast, target_data], axis=1)

        target = pd.concat([df_pivot[target], target_forecast], axis=0)

        return target


def add_past_forecast(
    df_pivot: pd.DataFrame,
    past_dynamic_real: List[str],
    prediction_length: int,
    item_id: pd.Series,
    dataset_type: str,
) -> pd.DataFrame:
    """Add 0 to past_dynamic_real for prediction length forecast.

    Parameters
    ----------
    df_pivot : DataFrame
        The pivot DataFrame containing the data.
    target : str
        The name of the target variable.
    dynamic_real : list
        List of names of dynamic real variables.
    past_dynamic_real : list
        List of names of past dynamic real variables.
    dynamic_cat : list
        List of names of dynamic categorical variables.
    prediction_length : int
        The length of the forecast.
    item_id : Series
        Series containing the item IDs.

    Returns
    -------
    DataFrame
        The DataFrame with the added 0 to past_dynamic_real.
    """
    if len(past_dynamic_real) == 0:
        return pd.DataFrame
    if dataset_type == "hugging_face":
        return df_pivot[past_dynamic_real]
    elif dataset_type == "gluonts":
        target_forecast = pd.DataFrame()
        num_item_id = len(item_id.unique())
        for feat in past_dynamic_real:
            index = pd.MultiIndex.from_tuples([(feat, i) for i in range(num_item_id)])
            past_data = np.array(list(repeat(df_pivot[feat], num_item_id))).T
            past_data = pd.DataFrame(
                np.zeros((prediction_length, df_pivot[feat].shape[1])), columns=index
            )
            past_forecast = pd.concat([target_forecast, past_data], axis=1)

        df_past_feat_dynamic_real = pd.concat([df_pivot[past_dynamic_real], past_forecast], axis=0)

        return df_past_feat_dynamic_real
