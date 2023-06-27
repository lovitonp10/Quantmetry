import pandas as pd


def resample_df_by_group(
    df: pd.DataFrame, grouper: str = "station_name", freq: str = "1D"
) -> pd.DataFrame:
    groups = df.groupby(grouper)

    resampled_df = pd.DataFrame()
    for name, group in groups:
        resampled_group = group.fillna(method="ffill").resample(freq).ffill()
        resampled_group[grouper] = name
        resampled_group = resampled_group.bfill()
        resampled_df = resampled_df.append(resampled_group)
    return resampled_df
