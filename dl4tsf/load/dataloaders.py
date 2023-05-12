import configs
import hydra
import pandas as pd
from gluonts.dataset.common import TrainDatasets
from utils.utils_gluonts import get_test_length, get_ts_length, create_ts_with_features
from typing import List


class CustomDataLoader:
    def __init__(
        self,
        cfg_dataset: configs.Dataset,
        target: str,
        feats: List[str],
        cfg_model: configs.Model,
        test_length: str,
    ) -> None:
        self.tmp = hydra.utils.instantiate(cfg_dataset.load, _convert_="all")
        self.register_data()
        self.prediction_length = cfg_model.model_config.prediction_length
        self.freq = cfg_dataset.freq
        self.cfg_dataset = cfg_dataset
        self.target = target if target else "target"
        self.dynamic_real = feats["feat_dynamic_real"]
        self.static_cat = feats["feat_static_cat"]
        self.static_real = feats["feat_static_real"]
        self.past_dynamic_real = feats["past_feat_dynamic_real"]
        self.dynamic_cat = feats["feat_dynamic_cat"]
        self.test_length = test_length
        self.static_cardinality = cfg_model.model_config.static_cardinality
        self.dynamic_cardinality = cfg_model.model_config.dynamic_cardinality

    def register_data(self):
        if isinstance(self.tmp, pd.DataFrame):
            self.df_pandas = self.tmp
        if isinstance(self.tmp, TrainDatasets):
            self.df_gluonts = self.tmp

    def get_pandas_format(self) -> pd.DataFrame:
        if self.df_pandas:
            return self.df_pandas
        # to implement

    def get_gluonts_format(self) -> TrainDatasets:
        test_length_rows = get_test_length(self.freq, self.test_length)
        self.cfg_dataset.ts_length = get_ts_length(self.df_pandas)

        self.df_gluonts = create_ts_with_features(
            self.df_pandas,
            self.target,
            self.dynamic_real,
            self.static_cat,
            self.static_real,
            self.past_dynamic_real,
            self.dynamic_cat,
            self.freq,
            test_length_rows,
            self.static_cardinality,
            self.dynamic_cardinality,
        )

        return self.df_gluonts

    def get_huggingface_format(self):
        # to implement
        pass
