from copy import deepcopy
from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class GetDummies(BaseEstimator, TransformerMixin):
    def __init__(self, column_names: List[str], drop_original: bool):
        self.column_names = column_names
        self.max_categories = None
        self.drop_original = drop_original

    def fit(self, df: pd.DataFrame, y=None):
        self.encoders = {}

        for column_name in self.column_names:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False,
                max_categories=self.max_categories,
                drop=None,
            )
            encoder.fit(df[[column_name]])

            self.encoders[column_name] = deepcopy(encoder)
        return self

    def transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        df_out = df.copy()

        for column_name in self.column_names:
            X_encoded = pd.DataFrame(
                self.encoders[column_name].transform(df_out[[column_name]]), dtype=int
            )
            X_encoded.columns = self.encoders[column_name].get_feature_names_out()
            X_encoded.columns = X_encoded.columns.str.replace(f"{column_name}_", f"{column_name}=")
            df_out = pd.concat([df_out, X_encoded], axis=1)

        if self.drop_original:
            df_out = df_out.drop(columns=self.column_names)

        return df_out
