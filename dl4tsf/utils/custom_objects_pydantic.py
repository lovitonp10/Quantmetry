from typing import Union

import pandas as pd
from datasets.arrow_dataset import Dataset
from pydantic import BaseModel as PydanticBaseModel


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


class HuggingFaceDataset(BaseModel):
    train: Union[Dataset, pd.DataFrame]
    validation: Union[Dataset, pd.DataFrame]
    test: Union[Dataset, pd.DataFrame]
