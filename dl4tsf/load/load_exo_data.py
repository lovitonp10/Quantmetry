import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import relativedelta


def get_station_id(path_weather: str = "data/all_weather/", name: str = "ORLY") -> int:
    df_stations = pd.read_csv(path_weather + "stations.txt", sep=";")

    id_station = df_stations[df_stations["Nom"] == name]["ID"].iloc[0]

    return id_station


def load_weather(
    path_weather: str = "data/all_weather/",
    start: str = "30-01-2022",
    end: str = "1-02-2022",
    dyn_features: list = ["t", "rr3", "pmer"],
    cat_features: list = ["cod_tend"],
    station_name: str = "ORLY",
    freq: str = "30T",
) -> pd.DataFrame:

    station = get_station_id(path_weather=path_weather, name=station_name)

    start = datetime.strptime(start, "%d-%m-%Y")
    end = datetime.strptime(end, "%d-%m-%Y") + timedelta(days=1)

    start_start_month = start.replace(day=1)
    end_start_month = end.replace(day=1)

    columns = ["numer_sta", "date"] + dyn_features + cat_features

    diff = relativedelta.relativedelta(end_start_month, start_start_month)

    month_difference = diff.years * 12 + diff.months

    df = pd.DataFrame(columns=columns)

    for i in range(month_difference + 1):
        date = start + relativedelta.relativedelta(months=i)

        YM_str = date.strftime("%Y%m")
        path_ = path_weather + "synop." + YM_str + ".csv.gz"

        df2 = pd.read_csv(path_, sep=";", compression="gzip")

        df2["date"] = df2["date"].apply(lambda x: datetime.strptime(str(x), "%Y%m%d%H%M%S"))
        df2 = df2[columns]
        df2 = df2[df2["numer_sta"] == station]

        if i == 0:
            df2 = df2[df2["date"] >= start]

        if i == month_difference:
            df2 = df2[df2["date"] <= end]

        df = pd.concat([df, df2], ignore_index=True)

    df.drop(["numer_sta"], axis=1, inplace=True)

    df[dyn_features] = df[dyn_features].fillna(np.median)
    df[cat_features] = df[cat_features].fillna(method="ffill")

    df.set_index("date", inplace=True)
    df = df.resample(freq).ffill()

    return df


def add_weather(
    df: pd.DataFrame,
    path_weather: str = "data/all_weather/",
    dyn_features: list = ["t", "rr3", "pmer"],
    cat_features: list = ["cod_tend"],
    station_name: str = "ORLY",
) -> pd.DataFrame:

    unique_dates = df.index.unique()
    sorted_dates = unique_dates.sort_values(ascending=True)
    first_date = sorted_dates.min()
    last_date = sorted_dates.max()

    first_date_str = first_date.strftime("%d-%m-%Y")
    last_date_str = last_date.strftime("%d-%m-%Y")

    frequence = pd.infer_freq(sorted_dates)

    weather = load_weather(
        path_weather=path_weather,
        start=first_date_str,
        end=last_date_str,
        dyn_features=["t", "rr3", "pmer"],
        cat_features=["cod_tend"],
        station_name="ORLY",
        freq=frequence,
    )

    df.index = df.index.tz_localize(None)
    weather.index = weather.index.tz_localize(None)

    merge = pd.merge(df, weather, left_index=True, right_index=True, how="left")

    return merge
