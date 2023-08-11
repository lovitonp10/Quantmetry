from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RemoveFirstObservations(BaseEstimator, TransformerMixin):
    def __init__(self, date_col: str, period: int):
        self.date_col = date_col
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        df_out = df.copy()

        new_min_date = df_out[self.date_col].min() + pd.DateOffset(days=self.period)
        df_out = df_out.query(
            f"{self.date_col}>=@new_min_date", local_dict={"new_min_date": new_min_date}
        )
        return df_out


class Add_FlightsPlan(BaseEstimator, TransformerMixin):
    def __init__(self, pdt_col: List[str], key_cols: List[str], df_pdt: pd.DataFrame):
        self.pdt_col = pdt_col
        self.key_cols = key_cols
        self.df_pdt = df_pdt

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        df_out = df.copy()
        df_out = df_out.merge(
            self.df_pdt[self.key_cols + self.pdt_col], how="left", on=self.key_cols
        )

        return df_out
