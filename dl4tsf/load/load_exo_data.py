import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import relativedelta
from typing import Dict
import math


def get_station_id(path_weather: str = "data/all_weather/", name: str = "ORLY") -> int:
    df_stations = pd.read_csv(path_weather + "stations.txt", sep=";")
    dict_stations = df_stations[["ID", "Nom"]].set_index("Nom")["ID"].to_dict()
    id_station = dict_stations[name]
    # id_station = df_stations[df_stations["Nom"] == name]["ID"].iloc[0]

    return id_station


def load_weather(
    path_weather: str = "data/all_weather/",
    start: str = "30-01-2022",
    end: str = "1-02-2022",
    dynamic_features: list = ["t", "rr3", "pmer"],
    cat_features: list = ["cod_tend"],
    station_name: str = "ORLY",
    freq: str = "30T",
) -> pd.DataFrame:
    station = get_station_id(path_weather=path_weather, name=station_name)

    start_start_month = start.replace(day=1)
    end_start_month = end.replace(day=1)

    columns = ["numer_sta", "date"] + dynamic_features + cat_features

    diff = relativedelta.relativedelta(end_start_month, start_start_month)

    month_difference = diff.years * 12 + diff.months

    df = pd.DataFrame(columns=columns)

    for i in range(month_difference + 1):
        date = start + relativedelta.relativedelta(months=i)

        YM_str = date.strftime("%Y%m")
        path_ = path_weather + "synop." + YM_str + ".csv.gz"

        df2 = pd.read_csv(path_, sep=";", compression="gzip")
        df2.rename(
            columns={
                "t": "temperature",
                "pmer": "pressure",
                "cod_tend": "barometric_trend",
                "rr3": "rainfall",
            },
            inplace=True,
        )
        df2["date"] = df2["date"].apply(lambda x: datetime.strptime(str(x), "%Y%m%d%H%M%S"))
        df2 = df2[columns]
        df2 = df2[df2["numer_sta"] == station]

        if i == 0:
            df2 = df2[df2["date"] >= start]

        if i == month_difference:
            df2 = df2[df2["date"] <= end]

        df = pd.concat([df, df2], ignore_index=True)

    df.drop(["numer_sta"], axis=1, inplace=True)

    for feat in dynamic_features:
        df[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(method="ffill").astype(float)
    df.replace("mq", np.nan, inplace=True)
    df[cat_features] = df[cat_features].fillna(method="ffill").astype(int)

    df.set_index("date", inplace=True)

    duplicates = df.index.duplicated()
    df = df[~duplicates].copy()
    df = df.resample(freq)
    df = df.ffill()

    return df


def add_weather(
    df: pd.DataFrame,
    weather: Dict[str, any] = {
        "path_weather": "data/all_weather/",
        "dynamic_features": ["t", "rr3", "pmer"],
        "cat_features": ["cod_tend"],
        "station_name": "ORLY",
    },
    prediction_length: int = 7,
) -> pd.DataFrame:

    path_weather = weather["path_weather"]
    dynamic_features = weather["dynamic_features"]
    cat_features = weather["cat_features"]
    station_name = weather["station_name"]

    unique_dates = df.index.unique()
    sorted_dates = unique_dates.sort_values(ascending=True)
    first_date = sorted_dates.min()
    last_date = sorted_dates.max()

    first_date_str = first_date.strftime("%d-%m-%Y")
    last_date_str = last_date.strftime("%d-%m-%Y")

    frequency = pd.infer_freq(sorted_dates)

    forecast_days = count_days_for_pred(freq=frequency, pred_length=prediction_length)

    start = datetime.strptime(first_date_str, "%d-%m-%Y")
    end = datetime.strptime(last_date_str, "%d-%m-%Y") + timedelta(days=1 + forecast_days)

    weather = load_weather(
        path_weather=path_weather,
        start=start,
        end=end,
        dynamic_features=dynamic_features,
        cat_features=cat_features,
        station_name=station_name,
        freq=frequency,
    )

    df.index = df.index.tz_localize(None)
    weather.index = weather.index.tz_localize(None)

    forecast_date_range = pd.date_range(
        start=last_date, periods=prediction_length + 1, freq=frequency
    )[1:].tz_localize(None)
    weather_forecast = weather.loc[forecast_date_range][dynamic_features + cat_features]

    merge = pd.merge(df, weather, left_index=True, right_index=True, how="left")

    return merge, weather_forecast


def count_days_for_pred(freq, pred_length):
    freq = pd.Timedelta(freq)
    # Calculer la durée totale en minutes
    total_minutes = freq.total_seconds() / 60 * pred_length
    # Calculer le nombre de jours
    days = math.ceil(total_minutes / (24 * 60))
    return days
