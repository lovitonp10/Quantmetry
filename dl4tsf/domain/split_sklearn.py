from typing import Tuple

import pandas as pd
from configs import TrainTestSplitConfig
from datetime import datetime, timedelta



def calculate_date(date,prediction_length,prediction_period):
    base_date = datetime.strptime(date, '%Y-%m-%d')
    new_date = base_date - timedelta(days=prediction_length * prediction_period)
    return new_date.strftime('%Y-%m-%d')

def split_train_test(
    df: pd.DataFrame, cfg: TrainTestSplitConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    date_col = cfg.date_col
    date_start_train = cfg.date_start_train
    date_train = calculate_date(cfg.date_split_train_test, cfg.prediction_length, 9)
    date_test = calculate_date(cfg.date_split_test_pred, cfg.prediction_length, 9)
    df_train = df.query(f"({date_col} >= '{date_start_train}') & {date_col} <= '{date_train}'")
    df_test = df.query(f"({date_col} > '{date_train}') & ({date_col}<='{date_test}')")
    df_pred = df.query(f"{date_col} > '{date_test}'")
    return df_train, df_test, df_pred
