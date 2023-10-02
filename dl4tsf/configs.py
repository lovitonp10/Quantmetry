from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel


class Metrics(str, Enum):
    mae = "mae"
    mse = "mse"
    rmse = "rmse"
    mape = "mape"
    smape = "smape"
    wmape = "wmape"


class Feats(BaseModel):
    feat_for_item_id: List[str]
    feat_dynamic_real: List[str]
    feat_static_cat: List[str]
    feat_static_real: List[str]
    past_feat_dynamic_real: List[str]
    feat_dynamic_cat: List[str]


class Dataset(BaseModel):
    dataset_name: str
    load: Dict[str, Any]
    test_length: str
    ts_length: Optional[int]
    freq: str
    freq_increm: str
    static_cardinality: Optional[List[int]]
    dynamic_cardinality: Optional[List[int]]
    name_feats: Feats


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


class TrainTestSplitConfig(BaseModel):
    date_col: str
    date_start_train: str
    date_split_train_test: str
    date_split_test_pred: str
    split_variable: Optional[str]


class CrossValConfig(BaseModel):
    unit: Literal["M", "D", "W"]
    training_minimum_window: int
    test_window: int


class SKLearnModel(BaseModel):
    model_name: str
    model_config: Dict[str, Any]
    target: str
    date_col: str
    error_metrics: List[Metrics]
    features: Optional[List[str]]
    cv_config: Optional[CrossValConfig]


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


class TrainSklean(BaseModel):
    # train_test_pred: TrainTestSplitConfig
    name: str


class Preprocess(BaseModel):
    pipeline: Dict[str, Any]
    train_test_pred: TrainTestSplitConfig
    item_id_list: Optional[List[str]]


class Configs(BaseModel):
    dataset: Dataset
    feature_engineering: Preprocess
    model: Union[Model, SKLearnModel]
    train: Union[Train, TrainSklean]
