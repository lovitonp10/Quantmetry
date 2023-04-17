from typing import List

import pandas as pd
from gluonts.model.forecast import Forecast


def sample_df(forecast: Forecast) -> List[pd.DataFrame]:
    samples = forecast.samples
    ns, h = samples.shape
    dates = pd.date_range(forecast.start_date.to_timestamp(), freq=forecast.freq, periods=h)
    return pd.DataFrame(samples.T, index=dates)
