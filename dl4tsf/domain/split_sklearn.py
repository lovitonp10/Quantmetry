from typing import Tuple

import pandas as pd
from configs import TrainTestSplitConfig


def split_train_test(
    df: pd.DataFrame, cfg: TrainTestSplitConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    date_col = cfg.date_col
    date_start_train = cfg.date_start_train
    date_train = cfg.date_split_train_test
    date_test = cfg.date_split_test_pred
    df_train = df.query(f"({date_col} >= '{date_start_train}') & {date_col} <= '{date_train}'")
    df_test = df.query(f"({date_col} > '{date_train}') & ({date_col}<='{date_test}')")
    df_pred = df.query(f"{date_col} > '{date_test}'")
    return df_train, df_test, df_pred
