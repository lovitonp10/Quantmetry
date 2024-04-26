from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class Feats(BaseModel):
    feat_for_item_id: List[str]
    feat_dynamic_real: List[str]
    feat_static_cat: List[str]
    feat_static_real: List[str]
    past_feat_dynamic_real: List[str]
    feat_dynamic_cat: List[str]


class dates_split(BaseModel):
    date_split_train: str
    date_split_val_start: str
    date_split_val_end: str
    date_split_test: str


class Dataset(BaseModel):
    dataset_name: str
    load: Dict[str, Any]
    test_length: str
    ts_length: Optional[int]
    freq: str
    static_cardinality: Optional[List[int]]
    dynamic_cardinality: Optional[List[int]]
    name_feats: Feats
    train_test_val_split: dates_split


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
    static_cardinality: Optional[List[int]]
    dynamic_cardinality: Optional[List[int]]
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
    batch_size_test: int
    train_sampler: Any
    validation_sampler: Any
    time_features: Any
    trainer_kwargs: Dict[str, Any]
    callback: Dict[str, Any]


class Configs(BaseModel):
    dataset: Dataset
    model: Model
    train: Train
