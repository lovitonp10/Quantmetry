import configs
import hydra
import pandas as pd
from gluonts.dataset.common import TrainDatasets
from utils.custom_objects_pydantic import HuggingFaceDataset
from utils.utils_gluonts import create_ts_with_features, get_test_length, get_ts_length


class CustomDataLoader:
    def __init__(
        self,
        cfg_dataset: configs.Dataset,
        target: str,
        feats: configs.Feats,
        cfg_model: configs.Model,
        test_length: str,
    ) -> None:
        self.tmp, self.df_forecast = hydra.utils.instantiate(cfg_dataset.load, _convert_="all")
        self.register_data()
        self.prediction_length = cfg_model.model_config.prediction_length
        self.freq = cfg_dataset.freq
        self.cfg_dataset = cfg_dataset
        self.target = target if target else "target"
        self.name_feats = feats
        self.test_length = test_length
        self.static_cardinality = cfg_model.model_config.static_cardinality
        self.dynamic_cardinality = cfg_model.model_config.dynamic_cardinality
        self.model_name = cfg_model.model_name

    def get_map_item_id(self):
        df_map = self.tmp[["item_id"] + self.name_feats.feat_for_item_id].drop_duplicates()
        return df_map.reset_index(drop=True)

    def register_data(self):
        if isinstance(self.tmp, pd.DataFrame):
            self.df_pandas = self.tmp
        if isinstance(self.tmp, TrainDatasets):
            self.df_gluonts = self.tmp
        if isinstance(self.tmp, HuggingFaceDataset):
            self.df_huggingface = self.tmp

    def create_huggingface_from_pandas(self):
        # Not yet ready for Informer model
        # self.df_huggingface = Dataset.from_pandas(self.df_pandas)
        test_length_rows = get_test_length(self.freq, self.test_length)
        self.cfg_dataset.ts_length = get_ts_length(self.df_pandas)

        self.df_huggingface = create_ts_with_features(
            "hugging_face",
            self.df_pandas,
            self.target,
            self.name_feats,
            self.freq,
            test_length_rows,
            self.prediction_length,
            self.static_cardinality,
            self.dynamic_cardinality,
            self.df_forecast,
        )

    def create_gluonts_from_pandas(self):
        test_length_rows = get_test_length(self.freq, self.test_length)
        self.cfg_dataset.ts_length = get_ts_length(self.df_pandas)

        self.df_gluonts = create_ts_with_features(
            "gluonts",
            self.df_pandas,
            self.target,
            self.name_feats,
            self.freq,
            test_length_rows,
            self.prediction_length,
            self.static_cardinality,
            self.dynamic_cardinality,
            self.df_forecast,
        )

    def get_dataset(self):
        if self.model_name == "TFTForecaster":
            return self.get_gluonts_format()
        elif self.model_name == "InformerForecaster":
            return self.get_huggingface_format()

    def get_pandas_format(self) -> pd.DataFrame:
        if self.df_pandas:
            return self.df_pandas

    def get_gluonts_format(self) -> TrainDatasets:
        if hasattr(self, "df_gluonts") and (self.df_gluonts is not None):
            return self.df_gluonts
        elif hasattr(self, "df_pandas") and (self.df_pandas is not None):
            self.create_gluonts_from_pandas()
            return self.get_gluonts_format()
        elif hasattr(self, "df_huggingface") and (self.df_huggingface is not None):
            self.create_pandas_from_hugging_face()
            return self.get_gluonts_format()

    def get_huggingface_format(self):
        if hasattr(self, "df_huggingface") and (self.df_huggingface):
            return self.df_huggingface
        elif hasattr(self, "df_pandas") and (self.df_pandas is not None):
            self.create_huggingface_from_pandas()
            return self.get_huggingface_format()
        # not priority
        # elif hasattr(self, "df_gluonts") and (self.df_gluonts is not None):
        #     self.create_pandas_from_gluonts()
        #     return self.get_huggingface_format()
