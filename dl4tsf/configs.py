from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class DataSet(BaseModel):
    repository_name: str
    dataset_name: str
    freq: str
    feats: Dict[str, int]


class ModelConfig(BaseModel):
    input_size: int
    prediction_length: int
    context_length: Optional[int]
    lags_sequence: List[int]
    dropout: float
    embed_dim: int
    variable_dim: Optional[int]
    num_heads: int
    encoder_layers: int
    decoder_layers: int
    d_models: int
    cardinality: Optional[List[int]]
    num_parallel_samples: int
    scaling: Union[str, bool]


class Model(BaseModel):
    model_config: ModelConfig
    optimizer_config: Dict[str, Any]


class Train(BaseModel):
    epochs: int
    batch_size_train: int
    nb_batch_per_epoch: int


class Configs(BaseModel):
    dataset: DataSet
    model: Model
    train: Train
