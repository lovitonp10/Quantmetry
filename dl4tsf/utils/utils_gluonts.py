from typing import List, Tuple, Dict, Any, Optional, NamedTuple
from utils.custom_objects_pydantic import HuggingFaceDataset
import pandas as pd
from pandas import Period
import numpy as np

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
    samples: np.ndarray, start_date: Period, periods, freq, ts_length, pred_length, validation
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


def transform_huggingface_to_dict(dataset, freq: str):
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
    dynamic_real: List[str],
    static_cat: List[str],
    static_real: List[str],
    past_dynamic_real: List[str],
    dynamic_cat: List[str],
    freq: str,
    test_length_rows: int,
    prediction_length: int,
    static_cardinality: List[int],
    dynamic_cardinality: List[int],
    df_forecast: pd.DataFrame,
) -> TrainDatasets:
    # static features
    df["item_id"], static_features_df = utils_static_features(df, static_cat, static_real)

    # dynamic features
    df_pivot = utils_dynamic_features(
        df, target, dynamic_real, past_dynamic_real, dynamic_cat, freq
    )

    dynamic_feat_forecast = create_df_dynamic_forecast(
        df_forecast,
        dynamic_real,
        dynamic_cat,
        df["item_id"],
    )

    train = train_val_test_split(
        "train",
        dataset_type,
        df,
        df_pivot,
        target,
        dynamic_real,
        static_features_df,
        static_cat,
        static_real,
        past_dynamic_real,
        dynamic_cat,
        test_length_rows,
        prediction_length,
        dynamic_feat_forecast,
    )

    validation = train_val_test_split(
        "validation",
        dataset_type,
        df,
        df_pivot,
        target,
        dynamic_real,
        static_features_df,
        static_cat,
        static_real,
        past_dynamic_real,
        dynamic_cat,
        test_length_rows,
        prediction_length,
        dynamic_feat_forecast,
    )

    test = train_val_test_split(
        "test",
        dataset_type,
        df,
        df_pivot,
        target,
        dynamic_real,
        static_features_df,
        static_cat,
        static_real,
        past_dynamic_real,
        dynamic_cat,
        test_length_rows,
        prediction_length,
        dynamic_feat_forecast,
    )

    if dataset_type == "gluonts":
        # gluonts dataset format
        dataset = gluonts_format(
            train=train,
            validation=validation,
            test=test,
            dynamic_real=dynamic_real,
            past_dynamic_real=past_dynamic_real,
            static_cat=static_cat,
            static_cardinality=static_cardinality,
            static_real=static_real,
            dynamic_cat=dynamic_cat,
            dynamic_cardinality=dynamic_cardinality,
            freq=freq,
            prediction_length=prediction_length,
        )

    elif dataset_type == "hugging_face":
        dataset = hugging_face_format(
            train=train,
            validation=validation,
            test=test,
            freq=freq,
        )

    return dataset


def gluonts_format(
    train,
    validation,
    test,
    dynamic_real,
    past_dynamic_real,
    static_cat,
    static_cardinality,
    static_real,
    dynamic_cat,
    dynamic_cardinality,
    freq,
    prediction_length,
):
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
    train_df = cast(Dataset, Map(process, train))
    validation_df = cast(Dataset, Map(process, validation))
    test_df = cast(Dataset, Map(process, test))

    dataset = TrainDatasets(metadata=meta, train=train_df, validation=validation_df, test=test_df)
    return dataset


def hugging_face_format(train, validation, test, freq):
    train_df = pd.DataFrame(train)
    validation_df = pd.DataFrame(validation)
    test_df = pd.DataFrame(test)

    train_dataset = datasets.Dataset.from_dict(train_df)
    validation_dataset = datasets.Dataset.from_dict(validation_df)
    test_dataset = datasets.Dataset.from_dict(test_df)
    # dataset = datasets.DatasetDict({"train":train_dataset,"test":test_dataset})

    train_dataset.set_transform(partial(transform_start_field, freq=freq))
    validation_dataset.set_transform(partial(transform_start_field, freq=freq))
    test_dataset.set_transform(partial(transform_start_field, freq=freq))
    dataset = HuggingFaceDataset
    dataset.train = train_dataset
    dataset.validation = validation_dataset
    dataset.test = test_dataset
    return dataset


def transform_start_field(batch, freq):
    batch["start"] = [pd.Period(date, freq) for date in batch["start"]]
    return batch


def utils_static_features(
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


def utils_dynamic_features(
    df: pd.DataFrame,
    target: str,
    dynamic_real: List[str],
    past_dynamic_real: List[str],
    dynamic_cat: List[str],
    freq: str,
) -> pd.DataFrame:
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

    for feat in dynamic_cat:
        df_pivot[feat] = df_pivot[feat].astype(int)

    return df_pivot


def train_val_test_split(
    part: str,
    dataset_type: str,
    df: pd.DataFrame,
    df_pivot: pd.DataFrame,
    target: str,
    dynamic_real: List[str],
    static_features_df: pd.DataFrame,
    static_cat: List[str],
    static_real: List[str],
    past_dynamic_real: List[str],
    dynamic_cat: List[str],
    test_length_rows: int,
    prediction_length: int,
    dynamic_feat_forecast: pd.DataFrame,
) -> List[Dict[str, Any]]:
    if part == "train":
        df_pivot = df_pivot[: -test_length_rows * 2]
        feat_dynamic_real = df_pivot[dynamic_real]
        feat_dynamic_cat = df_pivot[dynamic_cat]
        target = df_pivot[target]
        df_past_feat_dynamic_real = df_pivot[past_dynamic_real]

    if part == "validation":
        df_pivot = df_pivot[:-test_length_rows]
        feat_dynamic_real = df_pivot[dynamic_real]
        feat_dynamic_cat = df_pivot[dynamic_cat]
        target = df_pivot[target]
        df_past_feat_dynamic_real = df_pivot[past_dynamic_real]

    if part == "test" and dataset_type == "hugging_face":
        feat_dynamic_real = pd.concat(
            [df_pivot[dynamic_real], dynamic_feat_forecast[dynamic_real]], axis=0
        )
        feat_dynamic_cat = pd.concat(
            [df_pivot[dynamic_cat], dynamic_feat_forecast[dynamic_cat]], axis=0
        )
        target = df_pivot[target]
        df_past_feat_dynamic_real = df_pivot[past_dynamic_real]

    if part == "test" and dataset_type == "gluonts":
        feat_dynamic_real = pd.concat(
            [df_pivot[dynamic_real], dynamic_feat_forecast[dynamic_real]], axis=0
        )
        feat_dynamic_cat = pd.concat(
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

    data = [
        {
            "target": np.array(target[i].to_list()),
            "start": df_pivot.index[0],
            **(
                {
                    "feat_dynamic_real": np.array(
                        [
                            feat_dynamic_real[dynamic_real[j]][i].to_list()
                            for j in range(len(dynamic_real))
                        ]
                    )
                }
                if len(dynamic_real) != 0
                else {}
            ),
            **(
                {"feat_static_cat": np.array(static_features_df[static_cat])[i]}
                if len(static_cat) != 0
                else {}
            ),
            **(
                {
                    "feat_static_real": np.array(static_features_df[static_real])[i]
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
                            feat_dynamic_cat[dynamic_cat[j]][i].to_list()
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
    df_forecast,
    dynamic_real,
    dynamic_cat,
    item_id,
):
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
    df_pivot,
    target,
    prediction_length,
    item_id,
):
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
    df_pivot,
    past_dynamic_real,
    prediction_length,
    item_id,
):
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
