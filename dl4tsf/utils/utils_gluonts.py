from typing import List, Tuple, Dict, Any, Optional, Iterator, NamedTuple
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

from gluonts.transform import InstanceSplitter
from gluonts.transform.sampler import InstanceSampler
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName


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
    samples: np.ndarray, start_date: Period, periods, freq, ts_length, pred_length, test_step
) -> List[pd.DataFrame]:
    # samples = forecast.samples
    # ns, h = samples.shape
    if test_step is True:
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
    weather_forecast: pd.DataFrame,
) -> TrainDatasets:
    # static features
    df["item_id"], static_features_cat = utils_static_features(df, static_cat)

    # dynamic features
    df_pivot = utils_dynamic_features(
        df, target, dynamic_real, past_dynamic_real, dynamic_cat, freq
    )

    df_pivot, dynamic_feat_forecast = add_dynamic_forecast(
        df_pivot,
        weather_forecast,
        target,
        dynamic_real,
        past_dynamic_real,
        dynamic_cat,
        prediction_length,
    )

    train = train_val_test_split(
        "train",
        dataset_type,
        df,
        df_pivot,
        target,
        dynamic_real,
        static_features_cat,
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
        static_features_cat,
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
        static_features_cat,
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
    df: pd.DataFrame, static_cat: List[str]
) -> Tuple[pd.Series, pd.DataFrame]:
    if len(static_cat) != 0:
        lst_item = df[static_cat].apply(lambda x: "_".join(x.astype(str)), axis=1)
        lst_item = lst_item.astype("category").cat.codes
        static_features_cat = df.groupby(static_cat).sum().reset_index()[static_cat]
        for col in static_features_cat:
            static_features_cat[col] = static_features_cat[col].astype("category").cat.codes
    else:
        lst_item = 0
        static_features_cat = pd.DataFrame()

    return lst_item, static_features_cat


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
    if df_pivot.iloc[0].isna().any():
        df_pivot = df_pivot.drop(labels=df_pivot.index[0], axis=0)

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
    static_features_cat: pd.DataFrame,
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
        target_forecast = add_target_forecast(
            df_pivot,
            target,
            dynamic_real,
            past_dynamic_real,
            dynamic_cat,
            prediction_length,
        )
        target = pd.concat([df_pivot[target], target_forecast], axis=0)
        past_forecast = add_past_forecast(
            df_pivot,
            target,
            dynamic_real,
            past_dynamic_real,
            dynamic_cat,
            prediction_length,
        )
        df_past_feat_dynamic_real = pd.concat([df_pivot[past_dynamic_real], past_forecast], axis=0)

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
                {"feat_static_cat": np.array(static_features_cat)[i]}
                if len(static_cat) != 0
                else {}
            ),
            **(
                {"feat_static_real": np.array(df[df["item_id"] == i][static_real].iloc[0])}
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


def add_dynamic_forecast(
    df_pivot,
    weather_forecast,
    target,
    dynamic_real,
    past_dynamic_real,
    dynamic_cat,
    prediction_length,
):
    dynamic_feat_forecast = pd.DataFrame()
    if weather_forecast is not None:
        weather_dynamic_feat_real = ["temperature", "rainfall", "pressure"]
        weather_dynamic_feat_cat = ["barometric_trend"]
        original_dynamic_real = [x for x in dynamic_real if x not in weather_dynamic_feat_real]
        original_dynamic_cat = [x for x in dynamic_cat if x not in weather_dynamic_feat_cat]
        num_item_id = int(
            df_pivot.shape[1] / len([target] + dynamic_real + past_dynamic_real + dynamic_cat)
        )
        for feat in weather_dynamic_feat_real + weather_dynamic_feat_cat:
            index = pd.MultiIndex.from_tuples([(feat, i) for i in range(num_item_id)])
            weather_data = np.array(
                list(repeat(weather_forecast[feat], num_item_id))
            ).T  # np.tile(weather_forecast[feat], (1, num_item_id))
            weather_data = pd.DataFrame(weather_data, columns=index)
            dynamic_feat_forecast = pd.concat([dynamic_feat_forecast, weather_data], axis=1)
        if len(original_dynamic_real) > 0 or len(original_dynamic_cat) > 0:
            original_data = df_pivot[original_dynamic_real + original_dynamic_cat][
                -prediction_length:
            ]
            df_pivot = df_pivot[:-prediction_length]
            dynamic_feat_forecast = pd.concat([original_data, dynamic_feat_forecast], axis=1)

    elif weather_forecast is None and len(dynamic_real) > 0 or len(dynamic_cat) > 0:
        dynamic_feat_forecast = df_pivot[dynamic_real + dynamic_real][-prediction_length:]
        df_pivot = df_pivot[:-prediction_length]

    return df_pivot, dynamic_feat_forecast


def add_target_forecast(
    df_pivot,
    target,
    dynamic_real,
    past_dynamic_real,
    dynamic_cat,
    prediction_length,
):
    target_forecast = pd.DataFrame()
    num_item_id = int(
        df_pivot.shape[1] / len([target] + dynamic_real + past_dynamic_real + dynamic_cat)
    )
    for feat in [target]:
        index = pd.MultiIndex.from_tuples([(feat, i) for i in range(num_item_id)])
        target_data = np.array(list(repeat(df_pivot[feat], num_item_id))).T
        target_data = pd.DataFrame(
            np.zeros((prediction_length, df_pivot[feat].shape[1])), columns=index
        )
        target_forecast = pd.concat([target_forecast, target_data], axis=1)

    return target_forecast


def add_past_forecast(
    df_pivot,
    target,
    dynamic_real,
    past_dynamic_real,
    dynamic_cat,
    prediction_length,
):
    target_forecast = pd.DataFrame()
    num_item_id = int(
        df_pivot.shape[1] / len([target] + dynamic_real + past_dynamic_real + dynamic_cat)
    )
    for feat in past_dynamic_real:
        index = pd.MultiIndex.from_tuples([(feat, i) for i in range(num_item_id)])
        past_data = np.array(list(repeat(df_pivot[feat], num_item_id))).T
        past_data = pd.DataFrame(
            np.zeros((prediction_length, df_pivot[feat].shape[1])), columns=index
        )
        past_forecast = pd.concat([target_forecast, past_data], axis=1)

    return past_forecast


class CustomTFTInstanceSplitter(InstanceSplitter):
    """Instance splitter used by the Temporal Fusion Transformer model.

    Unlike ``InstanceSplitter``, this class returns known dynamic features as
    a single tensor of shape [..., context_length + prediction_length, ...]
    without splitting it into past & future parts. Moreover, this class supports
    dynamic features that are known in the past.
    """

    @validated()
    def __init__(
        self,
        instance_sampler: InstanceSampler,
        past_length: int,
        future_length: int,
        target_field: str = FieldName.TARGET,
        is_pad_field: str = FieldName.IS_PAD,
        start_field: str = FieldName.START,
        forecast_start_field: str = FieldName.FORECAST_START,
        observed_value_field: str = FieldName.OBSERVED_VALUES,
        lead_time: int = 0,
        output_NTC: bool = True,
        time_series_fields: List[str] = [],
        past_time_series_fields: List[str] = [],
        dummy_value: float = 0.0,
    ) -> None:
        super().__init__(
            target_field=target_field,
            is_pad_field=is_pad_field,
            start_field=start_field,
            forecast_start_field=forecast_start_field,
            instance_sampler=instance_sampler,
            past_length=past_length,
            future_length=future_length,
            lead_time=lead_time,
            output_NTC=output_NTC,
            time_series_fields=time_series_fields,
            dummy_value=dummy_value,
        )

        assert past_length > 0, "The value of `past_length` should be > 0"

        self.observed_value_field = observed_value_field
        self.past_ts_fields = past_time_series_fields

    def flatmap_transform(self, data: DataEntry, is_train: bool) -> Iterator[DataEntry]:
        pl = self.future_length
        lt = self.lead_time
        target = data[self.target_field]

        sampled_indices = self.instance_sampler(target)

        slice_cols = (
            self.ts_fields + self.past_ts_fields + [self.target_field, self.observed_value_field]
        )
        for i in sampled_indices:
            pad_length = max(self.past_length - i, 0)
            d = data.copy()

            for field in slice_cols:
                if i >= self.past_length:
                    past_piece = d[field][..., i - self.past_length : i]
                else:
                    pad_block = np.full(
                        shape=d[field].shape[:-1] + (pad_length,),
                        fill_value=self.dummy_value,
                        dtype=d[field].dtype,
                    )
                    past_piece = np.concatenate([pad_block, d[field][..., :i]], axis=-1)
                future_piece = d[field][..., (i + lt) : (i + lt + pl)]
                if self.output_NTC:
                    past_piece = past_piece.transpose()
                    future_piece = future_piece.transpose()
                if field not in self.past_ts_fields:
                    d[self._past(field)] = past_piece
                    d[self._future(field)] = future_piece
                    del d[field]
                else:
                    d[field] = past_piece
            pad_indicator = np.zeros(self.past_length)
            if pad_length > 0:
                pad_indicator[:pad_length] = 1
            d[self._past(self.is_pad_field)] = pad_indicator
            d[self.forecast_start_field] = d[self.start_field] + i + lt

            yield d
