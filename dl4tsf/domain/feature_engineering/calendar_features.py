from typing import Any, Dict, List

import holidays
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from vacances_scolaires_france import SchoolHolidayDates


class CalendarFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def get_season(self, dfSeries: pd.Series, hemisphere: str = "northern") -> pd.Series:

        if hemisphere == "northern":
            labels = ["Winter", "Spring", "Summer", "Fall"]

        elif hemisphere == "southern":
            labels = ["Summer", "Fall", "Winter", "Spring"]
        return pd.cut(
            (dfSeries.dt.dayofyear + 11) % 366,
            [0, 91, 183, 275, 366],
            include_lowest=True,
            labels=labels,
        ).astype(str)

    def transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        df_out = df.copy()
        df_out["quarter"] = df_out[self.column_name].dt.quarter
        df_out["month"] = df_out[self.column_name].dt.month
        df_out["day"] = df_out[self.column_name].dt.day
        df_out["dayofweek"] = df_out[self.column_name].dt.dayofweek
        df_out["hour"] = df_out[self.column_name].dt.hour
        df_out["season_north"] = self.get_season(df_out[self.column_name], hemisphere="northern")
        df_out["week"] = df_out[self.column_name].dt.isocalendar().week.astype(int)
        df_out["season_south"] = self.get_season(df_out[self.column_name], hemisphere="southern")
        df_out["isweekend"] = df_out["dayofweek"].isin([5, 6]).astype(int)

        return df_out


class AddHolidays(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        date_col: str,
        # mapping: Dict[str, List[Any]],
        config_country_holidays: Dict[str, Any],
        replace_values: Dict[str, Any],
        year_start: int,
        year_end: int,
        features: List[str],
    ):
        self.date_col = date_col
        # self.mapping = mapping
        self.config_country_holidays = config_country_holidays
        self.replace_values = replace_values
        self.year_start = year_start
        self.year_end = year_end
        self.features = features

    def fit(self, df, y=None):
        return self

    def create_holidays(self):
        tmp_holidays = holidays.country_holidays(
            years=range(self.year_start, self.year_end),
            language="en",
            **self.config_country_holidays
        )
        df_holidays = pd.DataFrame.from_dict(tmp_holidays.items())
        df_holidays.columns = ["date_tmp", "holiday_name"]

        df_holidays = (
            df_holidays.assign(holiday_name=df_holidays["holiday_name"].str.split(";"))
            .explode("holiday_name")
            .reset_index(drop=True)
        )

        df_holidays["holiday_name"] = df_holidays["holiday_name"].str.strip()
        for v, k in self.replace_values.items():
            df_holidays["holiday_name"] = df_holidays["holiday_name"].replace(k, v)

        df_holidays = df_holidays.groupby(["date_tmp"]).max().reset_index()
        df_holidays["isholiday"] = 1
        df_holidays = df_holidays.drop(columns=["holiday_name"])
        return df_holidays

    def create_vacances_scolaires(self):
        df_vac_scolaires = pd.DataFrame()

        schoolholidays = SchoolHolidayDates()
        for year in range(self.year_start, self.year_end):
            for zone in ["A", "B", "C"]:
                df_tmp = (
                    pd.DataFrame.from_dict(schoolholidays.holidays_for_year_and_zone(year, zone)).T
                ).reset_index(drop=True)
                df_tmp = df_tmp[["date"]]
                df_tmp["zone"] = zone
                df_vac_scolaires = pd.concat([df_vac_scolaires, df_tmp], axis=0)
        df_vac_scolaires["is_vacances"] = 1
        df_vac_scolaires = df_vac_scolaires.rename(columns={"date": "date_tmp"})
        df_vac_scolaires = df_vac_scolaires.set_index(["date_tmp", "zone"]).unstack()
        df_vac_scolaires.columns = df_vac_scolaires.columns.map("_zone=".join).str.strip("=")
        df_vac_scolaires = df_vac_scolaires.reset_index()
        return df_vac_scolaires

    def transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        df_out = df.copy()
        df_out["date_tmp"] = df_out[self.date_col].dt.date

        df_holidays = self.create_holidays()
        df_out = df_out.merge(df_holidays.drop_duplicates(), how="left", on="date_tmp")

        df_vacances_scolaires = self.create_vacances_scolaires()
        df_out = df_out.merge(df_vacances_scolaires.drop_duplicates(), how="left", on="date_tmp")

        for feature in self.features:
            df_out[feature] = df_out[feature].fillna(0)

        df_out = df_out.drop(columns=["date_tmp"])
        new_cols = list(set(df_out.columns) - set(df.columns))
        features_to_rm = list(set(new_cols) - set(self.features))

        df_out = df_out.drop(columns=features_to_rm)

        return df_out
