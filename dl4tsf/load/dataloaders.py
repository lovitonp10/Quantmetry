import configs
import hydra
import pandas as pd
from gluonts.dataset.common import MetaData, TrainDatasets
from gluonts.dataset.pandas import PandasDataset


class CustomDataLoader:
    def __init__(
        self, cfg_dataset: configs.Dataset, target: str, cfg_model: configs.Model
    ) -> None:
        self.tmp = hydra.utils.instantiate(cfg_dataset.load, _convert_="all")
        self.register_data()
        self.prediction_length = cfg_model.model_config.prediction_length
        self.freq = cfg_dataset.freq
        self.target = target if target else "target"
        self.test_length = cfg_dataset.test_length

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
        if hasattr(self, "df_gluonts") and (self.df_gluonts):
            return self.df_gluonts
        meta = MetaData(freq=self.freq, prediction_length=self.prediction_length)
        train = PandasDataset(self.df_pandas[: -(self.test_length)], target=self.target)
        test = PandasDataset(self.df_pandas, target=self.target)
        self.df_gluonts = TrainDatasets(metadata=meta, train=train, test=test)
        return self.df_gluonts

    def get_huggingface_format(self):
        # to implement
        pass
