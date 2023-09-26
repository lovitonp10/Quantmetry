from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LagsFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        column_name: str,
        date_col: str,
        lags: str = ["7D"],
        groupers: List[str] = ["item_id"],
    ):
        self.column_name = column_name
        self.date_col = date_col
        self.groupers = groupers
        self.lags: List[str] = lags

    def fit(self, X, y=None):
        return self

    def get_shifted_value(self, series: pd.Series, freq, groupers) -> pd.Series:
        """
        Get the shifted value according to frequency, for different groups
        Index must be a multiindex with at least one datetime type
        """
        return series.unstack(level=groupers).shift(freq=freq).stack(level=groupers)

    def transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        df_out = df.set_index([self.date_col] + self.groupers).copy()
        for lag in self.lags:
            if self.groupers and any(grouper != 'item_id' for grouper in self.groupers):
                df_out[f"{self.column_name}_{'_'.join([grouper for grouper in self.groupers if grouper != 'item_id'])}_prev={lag}"] = self.get_shifted_value(
                    df_out[self.column_name], freq=lag, groupers=self.groupers
                )
            else : 
                df_out[f"{self.column_name}_prev={lag}"] = self.get_shifted_value(
                    df_out[self.column_name], freq=lag, groupers=self.groupers)

        return df_out.reset_index()


class CumSumFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        column_name: str,
        date_col: str,
        groupers: List[str] = ["code_airport", "flight_type"],
    ):
        self.column_name = column_name
        self.date_col = date_col
        self.groupers = groupers

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        df_out = df.copy()
        df_out[f"{self.column_name}_cumsum_daily"] = df_out.groupby(
            self.groupers + [df_out["datetime_h"].dt.date]
        )[self.column_name].cumsum()
        return df_out.reset_index()


class RollingFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        column_name: str,
        date_col: str,
        groupers: List[str],
        lags: str = ["1D"],
        closed: str = "left",
        min_periods: int = 1,
        aggregate_func_name="mean",
    ):
        self.column_name = column_name
        self.date_col = date_col
        self.groupers = groupers
        self.lags = lags
        self.closed = closed
        self.min_periods = min_periods
        self.aggregate_func_name = aggregate_func_name
        self.aggregate_func = getattr(pd.core.window.rolling.Rolling, self.aggregate_func_name)

    def fit(self, X, y=None):
        return self

    def apply_rolling(self, grp, column, freq, on, closed, min_periods):
        return self.aggregate_func(
            grp.rolling(freq, on=on, closed=closed, min_periods=min_periods)[column]
        )

    def transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        df_out = df.copy()
        # index_original = list(df.index.names)
        # df_out = df.reset_index()
        for lag in self.lags:
            df_out[f"{self.column_name}_rolling_{self.aggregate_func_name}={lag}"] = (
                df_out.groupby(self.groupers, group_keys=False).apply(
                    lambda x: self.apply_rolling(
                        x,
                        freq=lag,
                        column=self.column_name,
                        on=self.date_col,
                        closed=self.closed,
                        min_periods=self.min_periods,
                    )
                )
                # .set_index(index_original)
            )

        # df_out = df_out.set_index(index_original)
        return df_out


def get_mean_rolling_amount(
    df: pd.DataFrame,
    column: str = "PMR",
    freq: str = "1D",
    on: str = "datetime_h",
    closed: str = "left",
    min_periods: int = None,
    groupers: List[str] = ["code_airport", "hour"],
) -> pd.Series:
    """
    Get rolling mean according to date
    """
    index_original = list(df.index.names)
    df_out = df.reset_index()

    def apply_rolling(grp, column, freq, on, closed, min_periods):
        return grp.rolling(freq, on=on, closed=closed, min_periods=min_periods)[column].mean()

    df_out["new_col"] = df_out.groupby(groupers, group_keys=False).apply(
        lambda x: apply_rolling(
            x, freq=freq, column=column, on=on, closed=closed, min_periods=min_periods
        )
    )

    df_out = df_out.set_index(index_original)
    return df_out["new_col"]


# to review
# def get_mean_rolling_amount_2(df,
#                        column='PMR',
#                        freq='1D', on='datetime_h',
#                        closed='left',
#                        groupers= ['code_airport', 'hour']):
#     """
#     Get rolling mean according to date
#     """
#     index_original = list(df.index.names)
#     df_out = df.reset_index()

#     df_out['new_col'] = df_out.groupby(groupers,
#                                        group_keys=False).rolling(freq,
#                                        on=on, closed=closed, min_periods=1
#                                        )[column].mean().reset_index()[column]

#     df_out = df_out.reset_index()
#     df_out = df_out.set_index(index_original)

#     return df_test
