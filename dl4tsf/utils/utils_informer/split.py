from typing import List, Iterator
from gluonts.transform import FlatMapTransformation, InstanceSampler
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
import numpy as np


"""
Description: This file is a modified version of split.py from gluonts that add
the support of past features split.
Original File:
https://github.com/awslabs/gluonts/blob/dev/src/gluonts/transform/split.py
"""


class CustomInformerInstanceSplitter(FlatMapTransformation):
    """
    Selects training instances, by slicing the target and other time series
    like arrays at random points in training mode or at the last time point in
    prediction mode. Assumption is that all time like arrays start at the same
    time point.

    The target and each time_series_field is removed and instead two
    corresponding fields with prefix `past_` and `future_` are included. E.g.

    If the target array is one-dimensional, the resulting instance has shape
    (len_target). In the multi-dimensional case, the instance has shape (dim,
    len_target).

    target -> past_target and future_target

    The transformation also adds a field 'past_is_pad' that indicates whether
    values where padded or not.

    Convention: time axis is always the last axis.

    Parameters
    ----------

    target_field
        field containing the target
    is_pad_field
        output field indicating whether padding happened
    start_field
        field containing the start date of the time series
    forecast_start_field
        output field that will contain the time point where the forecast starts
    instance_sampler
        instance sampler that provides sampling indices given a time series
    past_length
        length of the target seen before making prediction
    future_length
        length of the target that must be predicted
    lead_time
        gap between the past and future windows (default: 0)
    output_NTC
        whether to have time series output in (time, dimension) or in
        (dimension, time) layout (default: True)
    time_series_fields
        fields that contains time series, they are split in the same interval
        as the target (default: None)
    dummy_value
        Value to use for padding. (default: 0.0)
    """

    @validated()
    def __init__(
        self,
        target_field: str,
        is_pad_field: str,
        start_field: str,
        forecast_start_field: str,
        instance_sampler: InstanceSampler,
        past_length: int,
        future_length: int,
        lead_time: int = 0,
        output_NTC: bool = True,
        time_series_fields: List[str] = [],
        past_time_series_fields: List[str] = [],
        dummy_value: float = 0.0,
    ) -> None:
        super().__init__()

        assert future_length > 0, "The value of `future_length` should be > 0"

        self.instance_sampler = instance_sampler
        self.past_length = past_length
        self.future_length = future_length
        self.lead_time = lead_time
        self.output_NTC = output_NTC
        self.ts_fields = time_series_fields
        self.target_field = target_field
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field
        self.dummy_value = dummy_value

        self.past_ts_fields = past_time_series_fields

    def _past(self, col_name):
        return f"past_{col_name}"

    def _future(self, col_name):
        return f"future_{col_name}"

    def flatmap_transform(self, data: DataEntry, is_train: bool) -> Iterator[DataEntry]:
        pl = self.future_length
        lt = self.lead_time
        slice_cols = self.ts_fields + self.past_ts_fields + [self.target_field]
        target = data[self.target_field]

        sampled_indices = self.instance_sampler(target)

        for i in sampled_indices:
            pad_length = max(self.past_length - i, 0)
            d = data.copy()
            for ts_field in slice_cols:
                if i > self.past_length:
                    # truncate to past_length
                    past_piece = d[ts_field][..., i - self.past_length : i]
                elif i < self.past_length:
                    pad_block = (
                        np.ones(
                            d[ts_field].shape[:-1] + (pad_length,),
                            dtype=d[ts_field].dtype,
                        )
                        * self.dummy_value
                    )
                    past_piece = np.concatenate([pad_block, d[ts_field][..., :i]], axis=-1)
                else:
                    past_piece = d[ts_field][..., :i]
                if ts_field not in self.past_ts_fields:
                    d[self._past(ts_field)] = past_piece
                    d[self._future(ts_field)] = d[ts_field][..., i + lt : i + lt + pl]
                    del d[ts_field]
                else:
                    d[ts_field] = past_piece
            pad_indicator = np.zeros(self.past_length, dtype=target.dtype)
            if pad_length > 0:
                pad_indicator[:pad_length] = 1

            if self.output_NTC:
                for ts_field in slice_cols:
                    if ts_field not in self.past_ts_fields:
                        d[self._past(ts_field)] = d[self._past(ts_field)].transpose()
                        d[self._future(ts_field)] = d[self._future(ts_field)].transpose()
                    else:
                        d[ts_field] = d[ts_field].transpose()

            d[self._past(self.is_pad_field)] = pad_indicator
            d[self.forecast_start_field] = d[self.start_field] + i + lt
            yield d
