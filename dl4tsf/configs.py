from typing import Any, Dict

from pydantic import BaseModel


class DataSet(BaseModel):
    repository_name: str
    dataset_name: str
    freq: str


class Model(BaseModel):
    model_config: Dict[str, Any]
    optimizer_config: Dict[str, Any]


class Train(BaseModel):
    epochs: int
    batch_size_train: int
    nb_batch_per_epoch: int


class Configs(BaseModel):
    dataset: DataSet
    model: Model
    train: Train
