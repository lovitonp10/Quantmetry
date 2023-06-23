from typing import Any, Dict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts.dataset.field_names import FieldName


def plot_timeseries(
    ts_index, uni_variate_dataset: Dict[str, Any], prediction_length: int, forecasts: np
):
    fig, ax = plt.subplots()

    index = pd.period_range(
        start=uni_variate_dataset[ts_index][FieldName.START],
        periods=len(uni_variate_dataset[ts_index][FieldName.TARGET]),
        freq=uni_variate_dataset[ts_index][FieldName.START].freq,
    ).to_timestamp()

    ax.xaxis.set_minor_locator(mdates.HourLocator())
    ax.plot(
        index[-2 * prediction_length :],
        uni_variate_dataset[ts_index]["target"][-2 * prediction_length :],
        label="actual",
    )

    ax.plot(
        index[-prediction_length:],
        forecasts[ts_index, ...].mean(axis=0),
        label="mean",
    )
    ax.fill_between(
        index[-prediction_length:],
        forecasts[ts_index, ...].mean(0) - forecasts[ts_index, ...].std(axis=0),
        forecasts[ts_index, ...].mean(0) + forecasts[ts_index, ...].std(axis=0),
        alpha=0.2,
        interpolate=True,
        label="+/- 1-std",
    )
    ax.legend()
    fig.autofmt_xdate()

    plt.savefig("test.png")
    plt.close()
