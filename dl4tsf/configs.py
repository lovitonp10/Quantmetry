from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class Dataset(BaseModel):
    load: Dict[str, Any]
    freq: str
    feats: Dict[str, int]


class ModelConfig(BaseModel):
    input_size: int
    prediction_length: int
    context_length: Optional[int]
    lags_sequence: List[int]
    dropout: float
    variable_dim: Optional[int]
    num_heads: int
    encoder_layers: int
    decoder_layers: int
    d_models: int
    cardinality: Optional[List[int]]
    num_parallel_samples: int
    scaling: Union[str, bool]


class Model(BaseModel):
    model_name: str
    model_config: ModelConfig
    optimizer_config: Dict[str, Any]


class Train(BaseModel):
    epochs: int
    batch_size_train: int
    nb_batch_per_epoch: int
    train_sampler: Any
    validation_sampler: Any
    time_features: Any
    trainer_kwargs: Dict[str, Any]


class Configs(BaseModel):
    dataset: Dataset
    model: Model
    train: Train
