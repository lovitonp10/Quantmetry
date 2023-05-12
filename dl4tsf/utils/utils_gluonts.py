from typing import List, Tuple, Dict, Any, Optional, Iterator

import pandas as pd
from gluonts.model.forecast import Forecast
from gluonts.itertools import Map
from gluonts.dataset import Dataset
from gluonts.dataset.common import (
    ProcessDataEntry,
    TrainDatasets,
    CategoricalFeatureInfo,
    BasicFeatureInfo,
)
from gluonts.transform import InstanceSplitter
from gluonts.transform.sampler import InstanceSampler
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName

from typing import cast
import numpy as np
import pydantic


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


def sample_df(forecast: Forecast) -> List[pd.DataFrame]:
    samples = forecast.samples
    ns, h = samples.shape
    dates = pd.date_range(forecast.start_date.to_timestamp(), freq=forecast.freq, periods=h)
    return pd.DataFrame(samples.T, index=dates)


def get_ts_length(df_pandas: pd.DataFrame) -> int:
    ts_length = df_pandas.shape[0]
    return ts_length


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
    df: pd.DataFrame,
    target: str,
    dynamic_real: List[str],
    static_cat: List[str],
    static_real: List[str],
    past_dynamic_real: List[str],
    dynamic_cat: List[str],
    freq: str,
    prediction_length: int,
    static_cardinality: List[int],
    dynamic_cardinality: List[int],
) -> TrainDatasets:
    # static features
    df["item_id"], static_features_cat = utils_static_features(df, static_cat)

    # dynamic features
    df_pivot = utils_dynamic_features(
        df, target, dynamic_real, past_dynamic_real, dynamic_cat, freq
    )

    train = train_test_split(
        "train",
        df,
        df_pivot,
        target,
        dynamic_real,
        static_features_cat,
        static_cat,
        static_real,
        past_dynamic_real,
        dynamic_cat,
        prediction_length,
    )

    test = train_test_split(
        "test",
        df,
        df_pivot,
        target,
        dynamic_real,
        static_features_cat,
        static_cat,
        static_real,
        past_dynamic_real,
        dynamic_cat,
        prediction_length,
    )

    # gluonts dataset format
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
    test_df = cast(Dataset, Map(process, test))

    dataset = TrainDatasets(metadata=meta, train=train_df, test=test_df)

    return dataset


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

    return df_pivot


def train_test_split(
    part: str,
    df: pd.DataFrame,
    df_pivot: pd.DataFrame,
    target: str,
    dynamic_real: List[str],
    static_features_cat: pd.DataFrame,
    static_cat: List[str],
    static_real: List[str],
    past_dynamic_real: List[str],
    dynamic_cat: List[str],
    prediction_length: int,
) -> List[Dict[str, Any]]:
    if part == "train":
        df_pivot = df_pivot[:-prediction_length]

    data = [
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
                {"feat_static_real": np.array(df[df["item_id"] == i][static_real].iloc[0])}
                if len(static_real) != 0
                else {}
            ),
            **(
                {
                    "past_feat_dynamic_real": np.array(
                        [
                            df_pivot[past_dynamic_real[j]][i].to_list()
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
                        [df_pivot[dynamic_cat[j]][i].to_list() for j in range(len(dynamic_cat))]
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
