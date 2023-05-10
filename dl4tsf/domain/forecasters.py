from typing import Any, Dict, Iterable, List, Optional, Tuple

import configs
import hydra
import pandas as pd
import torch
import domain.metrics
from domain.lightning_module import TFTLightningModule
from domain.module import TFTModel
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.pandas import PandasDataset as gluontsPandasDataset
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.evaluation.backtest import backtest_metrics
from gluonts.itertools import Cyclic, IterableSlice, PseudoShuffled
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import IterableDataset
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from utils import utils_gluonts

PREDICTION_INPUT_NAMES = [
    "feat_static_cat",
    "feat_static_real",
    "past_time_feat",
    "past_target",
    "past_observed_values",
    "future_time_feat",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class Forecaster:
    def __init__(
        self,
        cfg_model: configs.Model,
        cfg_train: configs.Train,
        cfg_dataset: configs.Dataset,
        from_pretrained: str = None,
        **kwargs,
    ) -> None:
        self.model_config = cfg_model.model_config
        self.optimizer_config = cfg_model.optimizer_config
        self.cfg_train = cfg_train
        self.cfg_dataset = cfg_dataset
        self.from_pretrained = from_pretrained


class TFTForecaster(Forecaster, PyTorchLightningEstimator):
    @validated()
    def __init__(
        self,
        cfg_model: configs.Model,
        cfg_train: configs.Train,
        cfg_dataset: configs.Dataset,
        # model
        distr_output: DistributionOutput = StudentTOutput(),
        loss: DistributionLoss = NegativeLogLikelihood(),
        from_pretrained: str = None,
    ) -> None:
        Forecaster.__init__(
            self,
            cfg_model=cfg_model,
            cfg_train=cfg_train,
            cfg_dataset=cfg_dataset,
            from_pretrained=from_pretrained,
        )
        self.callback = hydra.utils.instantiate(cfg_train.callback, _convert_="all")
        self.logger = TensorBoardLogger(
            "tensorboard_logs", name=cfg_dataset.dataset_name, sub_dir="TFT", log_graph=True
        )
        self.add_kwargs = {"callbacks": [self.callback], "logger": self.logger}
        trainer_kwargs = {**cfg_train.trainer_kwargs, **self.add_kwargs}
        PyTorchLightningEstimator.__init__(self, trainer_kwargs=trainer_kwargs)

        self.freq = self.cfg_dataset.freq
        self.num_feat_dynamic_real = self.cfg_dataset.feats["num_feat_dynamic_real"]
        self.num_feat_static_cat = self.cfg_dataset.feats["num_feat_static_cat"]
        self.num_feat_static_real = self.cfg_dataset.feats["num_feat_static_real"]

        self.model_config.context_length = (
            self.model_config.context_length
            if self.model_config.context_length is not None
            else self.model_config.prediction_length
        )

        self.model = None
        self.distr_output = distr_output
        self.loss = loss
        self.model_config.variable_dim = (
            self.model_config.variable_dim or self.model_config.d_models
        )
        self.model_config.cardinality = (
            self.model_config.cardinality
            if self.model_config.cardinality and self.num_feat_static_cat > 0
            else [1]
        )
        self.time_features = (
            self.cfg_train.time_features
            if self.cfg_train.time_features is not None
            else time_features_from_frequency_str(self.freq)
        )

        self.batch_size = self.cfg_train.batch_size_train
        self.nb_batch_per_epoch = self.cfg_train.nb_batch_per_epoch

        self.train_sampler = self.cfg_train.train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=self.model_config.prediction_length
        )
        self.validation_sampler = self.cfg_train.validation_sampler or ValidationSplitSampler(
            min_future=self.model_config.prediction_length
        )

    def train(self, input_data: gluontsPandasDataset):
        self.model = None
        self.model = super().train(training_data=input_data)

    def predict(self, test_data: gluontsPandasDataset) -> Tuple[List[pd.Series], List[pd.Series]]:
        forecast_it, ts_it = make_evaluation_predictions(dataset=test_data, predictor=self.model)
        forecasts_df = []
        for forecast in forecast_it:
            forecasts_df.append(utils_gluonts.sample_df(forecast))
        return list(ts_it), forecasts_df

    def get_callback_losses(self, type: str = "train") -> Dict[str, Any]:
        return self.callback.metrics["loss"][f"{type}_loss"]

    def evaluate(
        self, input_data: gluontsPandasDataset, prediction_length: float, freq: float
    ) -> Dict[str, Any]:
        true_ts, forecasts = self.predict(input_data)
        agg_metrics = {
            "mae": domain.metrics.estimate_mae(forecasts, true_ts, prediction_length),
            "rmse": domain.metrics.estimate_rmse(forecasts, true_ts, prediction_length),
            "mape": domain.metrics.estimate_mape(forecasts, true_ts, prediction_length),
            "smape": domain.metrics.estimate_smape(forecasts, true_ts, prediction_length),
            "wmape": domain.metrics.estimate_wmape(forecasts, true_ts, prediction_length),
            "mase": domain.metrics.estimate_mase(forecasts, true_ts, prediction_length, freq),
            "QuantileLoss[0.1]": domain.metrics.quantileloss(
                forecasts, true_ts, prediction_length, 0.1
            ),
            "QuantileLoss[0.2]": domain.metrics.quantileloss(
                forecasts, true_ts, prediction_length, 0.2
            ),
            "QuantileLoss[0.3]": domain.metrics.quantileloss(
                forecasts, true_ts, prediction_length, 0.3
            ),
            "QuantileLoss[0.4]": domain.metrics.quantileloss(
                forecasts, true_ts, prediction_length, 0.4
            ),
            "QuantileLoss[0.5]": domain.metrics.quantileloss(
                forecasts, true_ts, prediction_length, 0.5
            ),
            "QuantileLoss[0.6]": domain.metrics.quantileloss(
                forecasts, true_ts, prediction_length, 0.6
            ),
            "QuantileLoss[0.7]": domain.metrics.quantileloss(
                forecasts, true_ts, prediction_length, 0.7
            ),
            "QuantileLoss[0.8]": domain.metrics.quantileloss(
                forecasts, true_ts, prediction_length, 0.8
            ),
            "QuantileLoss[0.9]": domain.metrics.quantileloss(
                forecasts, true_ts, prediction_length, 0.9
            ),
        }
        return agg_metrics

    def evaluate_gluonts(self, input_data: gluontsPandasDataset) -> Dict[str, Any]:
        ev = Evaluator(num_workers=0)
        agg_metrics, _ = backtest_metrics(input_data, self.model, evaluator=ev)
        return agg_metrics

    def create_transformation(self) -> Transformation:
        remove_field_names = []
        if self.num_feat_static_real == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if self.num_feat_dynamic_real == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
            [RemoveFields(field_names=remove_field_names)]
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])]
                if not self.num_feat_static_cat > 0
                else []
            )
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_REAL, value=[0.0])]
                if not self.num_feat_static_real > 0
                else []
            )
            + [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                ),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    # in the following line, we add 1 for the time dimension
                    expected_ndim=1 + len(self.distr_output.event_shape),
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.model_config.prediction_length,
                ),
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.model_config.prediction_length,
                    log_scale=True,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                    + ([FieldName.FEAT_DYNAMIC_REAL] if self.num_feat_dynamic_real > 0 else []),
                ),
            ]
        )

    def _create_instance_splitter(self, module: TFTLightningModule, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=module.model._past_length,
            future_length=self.model_config.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
            dummy_value=self.distr_output.value_in_support,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: TFTLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(module, "training") + SelectFields(
            TRAINING_INPUT_NAMES
        )

        training_instances = transformation.apply(
            Cyclic(data)
            if shuffle_buffer_length is None
            else PseudoShuffled(Cyclic(data), shuffle_buffer_length=shuffle_buffer_length)
        )

        return IterableSlice(
            iter(
                DataLoader(
                    IterableDataset(training_instances),
                    batch_size=self.batch_size,
                    **kwargs,
                )
            ),
            self.nb_batch_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        module: TFTLightningModule,
        **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(module, "validation") + SelectFields(
            TRAINING_INPUT_NAMES
        )

        validation_instances = transformation.apply(data)

        return DataLoader(
            IterableDataset(validation_instances),
            batch_size=self.batch_size,
            **kwargs,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: TFTLightningModule,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module.model,
            batch_size=self.batch_size,
            prediction_length=self.model_config.prediction_length,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def create_lightning_module(self) -> TFTLightningModule:
        model = TFTModel(
            freq=self.freq,
            model_config=self.model_config,
            num_feat_dynamic_real=1 + self.num_feat_dynamic_real + len(self.time_features),  # age
            num_feat_static_real=max(1, self.num_feat_static_real),
            num_feat_static_cat=max(1, self.num_feat_static_cat),
            distr_output=self.distr_output,
        )

        return TFTLightningModule(model=model, loss=self.loss)
