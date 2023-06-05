from typing import List, Tuple, Dict, Any, Optional, NamedTuple
from utils.custom_objects_pydantic import HuggingFaceDataset
import pandas as pd
from pandas import Period
import numpy as np
import configs


from gluonts.itertools import Map
from pathlib import Path
import shutil
from gluonts import json
from gluonts.dataset import Dataset, DatasetWriter
import datasets
from gluonts.dataset.common import (
    ProcessDataEntry,
    # TrainDatasets,
    CategoricalFeatureInfo,
    BasicFeatureInfo,
)

from typing import cast
import pydantic
from functools import partial
from itertools import repeat


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
    # samples = forecast.samples
    # ns, h = samples.shape
    if validation is False:
        dates = pd.date_range(start_date.to_timestamp(), freq=freq, periods=periods).shift(
            ts_length
        )
    else:
        dates = pd.date_range(start_date.to_timestamp(), freq=freq, periods=periods).shift(
            ts_length - pred_length
        )
    return pd.DataFrame(samples.T, index=dates)


def get_ts_length(df_pandas: pd.DataFrame) -> int:
    ts_length = df_pandas.shape[0]
    return ts_length


def transform_huggingface_to_pandas(gluonts_dataset: pd.DataFrame, freq: str):
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

        if "feat_static_cat" in gluonts_dataset.features:
            df_tmp["feat_static_cat"] = (
                item["feat_static_cat"][0]
                if isinstance(item["feat_static_cat"], list) and len(item["feat_static_cat"]) == 1
                else item["feat_static_cat"]
            )
        if "feat_dynamic_real" in gluonts_dataset.features:
            df_tmp["feat_dynamic_real"] = (
                item["feat_dynamic_real"][0]
                if isinstance(item["feat_dynamic_real"], list)
                and len(item["feat_dynamic_real"]) == 1
                else item["feat_dynamic_real"]
            )
        df_pandas = pd.concat([df_pandas, df_tmp], axis=0)
    return df_pandas


def transform_huggingface_to_dict(dataset: pd.DataFrame, freq: str):
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
    dynamic_real : List[str]
        List of names of columns containing dynamic real-valued features.
    static_cat : List[str]
        List of names of columns containing static categorical features.
    static_real : List[str]
        List of names of columns containing static real-valued features.
    past_dynamic_real : List[str]
        List of names of columns containing past values of dynamic real-valued features.
    dynamic_cat : List[str]
        List of names of columns containing dynamic categorical features.
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

    # calculate item_id and get an efficient df static_features for the next format
    df["item_id"], df_static_features = utils_item_id(df, static_cat, static_real)

    # resample, fill na and create pivot table with all dynamic features and target
    df_pivot = utils_missing_values(df, target, dynamic_real, past_dynamic_real, dynamic_cat, freq)

    # create df with all features known in the future, length = prediction_length
    df_dynamic_feat_forecast = create_df_dynamic_forecast(
        df_forecast,
        dynamic_real,
        dynamic_cat,
        df["item_id"],
    )

    # create train df
    df_train = train_val_test_split(
        "train",
        dataset_type,
        df,
        df_pivot,
        target,
        name_feats,
        df_static_features,
        test_length_rows,
        prediction_length,
        df_dynamic_feat_forecast,
    )

    # create train df
    df_validation = train_val_test_split(
        "validation",
        dataset_type,
        df,
        df_pivot,
        target,
        name_feats,
        df_static_features,
        test_length_rows,
        prediction_length,
        df_dynamic_feat_forecast,
    )

    # create test df
    df_test = train_val_test_split(
        "test",
        dataset_type,
        df,
        df_pivot,
        target,
        name_feats,
        df_static_features,
        test_length_rows,
        prediction_length,
        df_dynamic_feat_forecast,
    )

    if dataset_type == "gluonts":
        # gluonts dataset format
        dataset = gluonts_format(
            df_train=df_train,
            df_validation=df_validation,
            df_test=df_test,
            name_feats=name_feats,
            static_cardinality=static_cardinality,
            dynamic_cardinality=dynamic_cardinality,
            freq=freq,
            prediction_length=prediction_length,
        )

    elif dataset_type == "hugging_face":
        dataset = hugging_face_format(
            df_train=df_train,
            df_validation=df_validation,
            df_test=df_test,
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


def transform_start_field(batch: pd.DataFrame, freq: str):
    batch["start"] = [pd.Period(date, freq) for date in batch["start"]]
    return batch


def utils_item_id(
    df: pd.DataFrame,
    static_cat: List[str],
    static_real: List[str],
) -> Tuple[pd.Series, pd.DataFrame]:
    if len(static_cat) != 0:
        static_feat = static_cat
        if len(static_real) != 0:
            static_feat = static_feat + static_real
        lst_item = df[static_feat].apply(lambda x: "_".join(x.astype(str)), axis=1)
        lst_item = lst_item.astype("category").cat.codes
        static_features_df = df.groupby(static_feat).sum().reset_index()[static_feat]
        for col in static_features_df[static_cat]:
            static_features_df[col] = static_features_df[col].astype("category").cat.codes

    elif len(static_real) != 0:
        static_feat = static_real
        lst_item = df[static_feat].apply(lambda x: "_".join(x.astype(str)), axis=1)
        lst_item = lst_item.astype("category").cat.codes
        static_features_df = df.groupby(static_feat).sum().reset_index()[static_feat]

    else:
        lst_item = 0
        static_features_df = pd.DataFrame()

    return lst_item, static_features_df


def utils_missing_values(
    df: pd.DataFrame,
    target: str,
    dynamic_real: List[str],
    past_dynamic_real: List[str],
    dynamic_cat: List[str],
    freq: str,
) -> pd.DataFrame:

    # create pivot table with all dynamic feat and target
    # resample with the right frequency
    # fill na with method linear interpolate
    df_pivot = (
        pd.pivot_table(
            df,
            values=[target] + dynamic_real + past_dynamic_real + dynamic_cat,
            index=df.index,
            columns=["item_id"],
        )
        .resample(freq)
        .interpolate(method="linear")
    )

    # The method interpolate doesn't fillna on the first row
    # We use Backward Fill to fill the NaN values with the next valid observation
    df_pivot.fillna(method="bfill", inplace=True)

    """if df_pivot.iloc[0].isna().any():
        df_pivot = df_pivot.drop(labels=df_pivot.index[0], axis=0)"""

    # dynamic cat as type int
    for feat in dynamic_cat:
        df_pivot[feat] = df_pivot[feat].astype(int)

    return df_pivot


def train_val_test_split(
    part: str,
    dataset_type: str,
    df: pd.DataFrame,
    df_pivot: pd.DataFrame,
    target: str,
    name_feats: configs.Feats,
    df_static_features: pd.DataFrame,
    test_length_rows: int,
    prediction_length: int,
    dynamic_feat_forecast: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Split the data into train, validation, and test sets for time series forecasting.

    Parameters
    ----------
    part : str
        The part of the data to split ("train", "validation", "test").
    dataset_type : str
        The type of dataset being created ("gluonts", "hugging_face").
    df : pd.DataFrame
        The input DataFrame containing the time series data.
    df_pivot : pd.DataFrame
        The pivot DataFrame containing the transformed time series data.
    target : str
        The name of the target variable column.
    dynamic_real : List[str]
        A list of names of columns containing dynamic real-valued features.
    df_static_features : pd.DataFrame
        The DataFrame containing static features.
    static_cat : List[str]
        A list of names of columns containing static categorical features.
    static_real : List[str]
        A list of names of columns containing static real-valued features.
    past_dynamic_real : List[str]
        A list of names of columns containing past values of dynamic real-valued features.
    dynamic_cat : List[str]
        A list of names of columns containing dynamic categorical features.
    test_length_rows : int
        The number of rows to use for the test dataset.
    prediction_length : int
        The length of the prediction horizon.
    dynamic_feat_forecast : pd.DataFrame
        The DataFrame containing the forecasted values for the dynamic features.

    Returns
    -------
    List[Dict[str, Any]]
        The split train, validation, or test datasets.

    """

    dynamic_real = name_feats.feat_dynamic_real
    static_cat = name_feats.feat_static_cat
    static_real = name_feats.feat_static_real
    past_dynamic_real = name_feats.past_feat_dynamic_real
    dynamic_cat = name_feats.feat_dynamic_cat

    if part == "train":
        df_pivot = df_pivot[: -test_length_rows * 2]
        df_feat_dynamic_real = df_pivot[dynamic_real]
        df_feat_dynamic_cat = df_pivot[dynamic_cat]
        target = df_pivot[target]
        df_past_feat_dynamic_real = df_pivot[past_dynamic_real]

    elif part == "validation":
        df_pivot = df_pivot[:-test_length_rows]
        df_feat_dynamic_real = df_pivot[dynamic_real]
        df_feat_dynamic_cat = df_pivot[dynamic_cat]
        target = df_pivot[target]
        df_past_feat_dynamic_real = df_pivot[past_dynamic_real]

    elif part == "test" and dataset_type == "hugging_face":
        df_feat_dynamic_real = pd.concat(
            [df_pivot[dynamic_real], dynamic_feat_forecast[dynamic_real]], axis=0
        )
        df_feat_dynamic_cat = pd.concat(
            [df_pivot[dynamic_cat], dynamic_feat_forecast[dynamic_cat]], axis=0
        )
        target = df_pivot[target]
        df_past_feat_dynamic_real = df_pivot[past_dynamic_real]

    elif part == "test" and dataset_type == "gluonts":
        df_feat_dynamic_real = pd.concat(
            [df_pivot[dynamic_real], dynamic_feat_forecast[dynamic_real]], axis=0
        )
        df_feat_dynamic_cat = pd.concat(
            [df_pivot[dynamic_cat], dynamic_feat_forecast[dynamic_cat]], axis=0
        )
        target = add_target_forecast(
            df_pivot,
            target,
            prediction_length,
            df["item_id"],
        )
        df_past_feat_dynamic_real = add_past_forecast(
            df_pivot,
            past_dynamic_real,
            prediction_length,
            df["item_id"],
        )

    data = create_dict_dataset(
        target,
        df_pivot,
        df_feat_dynamic_real,
        dynamic_real,
        df_static_features,
        static_cat,
        static_real,
        df_past_feat_dynamic_real,
        past_dynamic_real,
        df_feat_dynamic_cat,
        dynamic_cat,
        df,
    )

    return data


def create_dict_dataset(
    target: str,
    df_pivot: pd.DataFrame,
    df_feat_dynamic_real: pd.DataFrame,
    dynamic_real: List[str],
    df_static_features: pd.DataFrame,
    static_cat: List[str],
    static_real: List[str],
    df_past_feat_dynamic_real: pd.DataFrame,
    past_dynamic_real: List[str],
    df_feat_dynamic_cat: pd.DataFrame,
    dynamic_cat: List[str],
    df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Create a dictionary-based dataset for time series forecasting.

    Parameters
    ----------
    target : str
        The name of the target variable column.
    df_pivot : pd.DataFrame
        The pivot DataFrame containing the transformed time series data.
    df_feat_dynamic_real : pd.DataFrame
        The DataFrame containing dynamic real-valued features.
    dynamic_real : List[str]
        A list of names of columns containing dynamic real-valued features.
    df_static_features : pd.DataFrame
        The DataFrame containing static features.
    static_cat : List[str]
        A list of names of columns containing static categorical features.
    static_real : List[str]
        A list of names of columns containing static real-valued features.
    df_past_feat_dynamic_real : pd.DataFrame
        The DataFrame containing past values of dynamic real-valued features.
    past_dynamic_real : List[str]
        A list of names of columns containing past values of dynamic real-valued features.
    df_feat_dynamic_cat : pd.DataFrame
        The DataFrame containing dynamic categorical features.
    dynamic_cat : List[str]
        A list of names of columns containing dynamic categorical features.
    df : pd.DataFrame
        The input DataFrame containing the time series data.

    Returns
    -------
    List[Dict[str, Any]]
        The dictionary-based dataset.

    """
    data = [
        {
            "target": np.array(target[i].to_list()),
            "start": df_pivot.index[0],
            **(
                {
                    "feat_dynamic_real": np.array(
                        [
                            df_feat_dynamic_real[dynamic_real[j]][i].to_list()
                            for j in range(len(dynamic_real))
                        ]
                    )
                }
                if len(dynamic_real) != 0
                else {}
            ),
            **(
                {"feat_static_cat": np.array(df_static_features[static_cat])[i]}
                if len(static_cat) != 0
                else {}
            ),
            **(
                {
                    "feat_static_real": np.array(df_static_features[static_real])[i]
                }  # np.array(df[df["item_id"] == i][static_real].iloc[0])}
                if len(static_real) != 0
                else {}
            ),
            **(
                {
                    "past_feat_dynamic_real": np.array(
                        [
                            df_past_feat_dynamic_real[past_dynamic_real[j]][i].to_list()
                            for j in range(len(past_dynamic_real))
                        ]
                    )
                }
                if len(past_dynamic_real) != 0
                else {}
            ),
            **(
                {
                    "feat_dynamic_cat": np.array(
                        [
                            df_feat_dynamic_cat[dynamic_cat[j]][i].to_list()
                            for j in range(len(dynamic_cat))
                        ]
                    )
                }
                if len(dynamic_cat) != 0
                else {}
            ),
            "item_id": str(i),
        }
        for i in df["item_id"].unique()
    ]

    return data


def create_df_dynamic_forecast(
    df_forecast: pd.DataFrame,
    dynamic_real: List[str],
    dynamic_cat: List[str],
    item_id: pd.Series,
) -> pd.DataFrame:
    """Create a DataFrame for dynamic forecast data with MultiIndex.

    Parameters
    ----------
    df_forecast : DataFrame
        The forecast DataFrame containing the data.
    dynamic_real : list
        List of names of dynamic real variables.
    dynamic_cat : list
        List of names of dynamic categorical variables.
    item_id : Series
        Series containing the item IDs.

    Returns
    -------
    DataFrame
        The DataFrame with the added dynamic forecast data.
    """

    dynamic_feat_forecast = pd.DataFrame()
    if df_forecast is not None:
        weather_dynamic_feat_real = ["temperature", "rainfall", "pressure"]
        weather_dynamic_feat_cat = ["barometric_trend"]
        original_dynamic_real = [x for x in dynamic_real if x not in weather_dynamic_feat_real]
        original_dynamic_cat = [x for x in dynamic_cat if x not in weather_dynamic_feat_cat]
        num_item_id = len(item_id.unique())
        if len(original_dynamic_real) > 0 or len(original_dynamic_cat) > 0:
            dynamic_feat_forecast = df_forecast
        else:
            for feat in weather_dynamic_feat_real + weather_dynamic_feat_cat:
                index = pd.MultiIndex.from_tuples([(feat, i) for i in range(num_item_id)])
                weather_data = np.array(
                    list(repeat(df_forecast[feat], num_item_id))
                ).T  # np.tile(weather_forecast[feat], (1, num_item_id))
                weather_data = pd.DataFrame(weather_data, columns=index)
                dynamic_feat_forecast = pd.concat([dynamic_feat_forecast, weather_data], axis=1)

    return dynamic_feat_forecast


def add_target_forecast(
    df_pivot: pd.DataFrame,
    target: str,
    prediction_length: int,
    item_id: pd.Series,
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
