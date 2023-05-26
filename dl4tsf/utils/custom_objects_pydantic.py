from pydantic import BaseModel as PydanticBaseModel
from datasets.arrow_dataset import Dataset


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


class HuggingFaceDataset(BaseModel):
    train: Dataset
    validation: Dataset
    test: Dataset
