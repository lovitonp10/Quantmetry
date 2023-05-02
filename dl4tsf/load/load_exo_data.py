import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import relativedelta





def add_meteo(
    path: str = "data/all_weather/",
    start: str = "15-01-2022",
    end: str = "25-02-2022",
    features: list = ["t", "rr3", "pmer"],
    station: int = 7149,
    target: str = "t",
) -> pd.DataFrame:

    start = datetime.strptime(start, "%d-%m-%Y")
    end = datetime.strptime(end, "%d-%m-%Y")

    columns = ["numer_sta", "date"] + features

    diff = relativedelta.relativedelta(end, start)
    month_difference = diff.years * 12 + diff.months

    df = pd.DataFrame(columns=columns)

    for i in range(month_difference + 1):
        date = start + relativedelta.relativedelta(months=i)

        YM_str = date.strftime("%Y%m")
        path_ = path + "synop." + YM_str + ".csv.gz"

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

    df[features] = df[features].fillna(np.median)
    df.set_index("date", inplace=True)
    df = df.resample("15T").ffill()

    return df