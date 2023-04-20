from typing import List

import pandas as pd
from gluonts.model.forecast import Forecast


def sample_df(forecast: Forecast) -> List[pd.DataFrame]:
    samples = forecast.samples
    ns, h = samples.shape
    dates = pd.date_range(forecast.start_date.to_timestamp(), freq=forecast.freq, periods=h)
    return pd.DataFrame(samples.T, index=dates)


def get_test_length(freq: str, test_length: str) -> int:
    """_summary_

    Parameters
    ----------
    freq : str
        _description_
    test_length : str
        _description_

    Returns
    -------
    int
        _description_
    """
    freq_minutes = pd.Timedelta(freq).total_seconds() / 60
    test_length_minutes = pd.Timedelta(test_length).total_seconds() / 60
    return int(test_length_minutes / freq_minutes)
