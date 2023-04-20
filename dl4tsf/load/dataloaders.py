import configs
import hydra
import pandas as pd
from gluonts.dataset.common import MetaData, TrainDatasets
from gluonts.dataset.pandas import PandasDataset
from utils.utils_gluonts import get_test_length
from utils.custom_objects_pydantic import HuggingFaceDataset


class CustomDataLoader:
    def __init__(
        self,
        cfg_dataset: configs.Dataset,
        target: str,
        cfg_model: configs.Model,
        test_length: str,
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
        if isinstance(self.tmp, HuggingFaceDataset):
            self.df_huggingface = self.tmp

    def create_pandas_from_hugging_face(self):
        pass

    def create_gluonts_from_pandas(self):
        test_length_rows = get_test_length(self.freq, self.test_length)

        train_df = pd.melt(self.df_pandas[:-(test_length_rows)], ignore_index=False)
        test_df = pd.melt(self.df_pandas, ignore_index=False)

        train_data = PandasDataset.from_long_dataframe(
            train_df, target="value", item_id="variable", freq=self.freq
        )
        test_data = PandasDataset.from_long_dataframe(
            test_df, target="value", item_id="variable", freq=self.freq
        )

        meta = MetaData(
            cardinality=len(self.df_pandas[:-(test_length_rows)]),
            freq=self.freq,
            prediction_length=self.prediction_length,
        )

        self.df_gluonts = TrainDatasets(metadata=meta, train=train_data, test=test_data)

        
    def get_pandas_format(self) -> pd.DataFrame:
        if self.df_pandas:
            return self.df_pandas
        # to implement

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
