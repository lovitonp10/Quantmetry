from typing import Any, Dict, Iterable, List, Optional, Tuple
from gluonts.evaluation import make_evaluation_predictions
import configs
import hydra
import numpy as np
import pandas as pd
import torch
import domain.metrics
from domain.lightning_module import TFTLightningModule
from domain.module import TFTModel
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.pandas import PandasDataset as gluontsPandasDataset
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
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)
from utils.utils_tft.split import CustomTFTInstanceSplitter
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from utils import utils_gluonts
import pickle
from pathlib import Path

import logging

from accelerate import Accelerator
from torch.optim import AdamW
from domain.transformations import (
    create_test_dataloader,
    create_train_dataloader,
    create_validation_dataloader,
)
from gluonts.time_feature import get_seasonality

import evaluate

from utils.utils_informer.configuration_informer import CustomInformerConfig
from utils.utils_informer.modeling_informer import CustomInformerForPrediction
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow_deploy.flavor import log_model


logger = logging.getLogger(__name__)

PREDICTION_INPUT_NAMES = [
    "feat_static_cat",
    "feat_static_real",
    "past_time_feat",
    "past_target",
    "past_observed_values",
    "future_time_feat",
    "past_feat_dynamic_real",
    "past_feat_dynamic_cat",
    "future_feat_dynamic_cat",
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
        from_mlflow: str = None,
        **kwargs,
    ) -> None:
        self.model_config = cfg_model.model_config
        self.optimizer_config = cfg_model.optimizer_config
        self.model_name = cfg_model.model_name
        self.cfg_train = cfg_train
        self.cfg_dataset = cfg_dataset
        self.from_mlflow = from_mlflow

    def get_model(self):
        return self.model

    def save(self, path):
        pickle.dump(self.get_model(), Path(path).open(mode="wb"))

    # @classmethod
    def load_model(self, from_mlflow, dst_path=None):
        from_mlflow = "runs:/" + str(from_mlflow) + "/model"
        local_model_path = _download_artifact_from_uri(
            artifact_uri=from_mlflow, output_path=dst_path
        )
        model_subpath = Path(local_model_path) / "model.pkl"
        return pickle.load(Path(model_subpath).open(mode="rb"))

    def save_mlflow_model(self, tracking_url_type_store, forecaster):
        registered_model_name = self.model_name if tracking_url_type_store != "file" else None
        log_model(
            model=forecaster, artifact_path="model", registered_model_name=registered_model_name
        )


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
        from_mlflow: str = None,
    ) -> None:
        Forecaster.__init__(
            self,
            cfg_model=cfg_model,
            cfg_train=cfg_train,
            cfg_dataset=cfg_dataset,
            from_mlflow=from_mlflow,
        )
        self.callback = hydra.utils.instantiate(cfg_train.callback, _convert_="all")
        self.logger = TensorBoardLogger(
            "tensorboard_logs",
            name=cfg_dataset.dataset_name,
            sub_dir="TFT",
            log_graph=True,
        )
        self.add_kwargs = {"callbacks": [self.callback], "logger": self.logger}
        trainer_kwargs = {**cfg_train.trainer_kwargs, **self.add_kwargs}
        PyTorchLightningEstimator.__init__(self, trainer_kwargs=trainer_kwargs)

        self.freq = self.cfg_dataset.freq
        self.num_feat_dynamic_real = len(self.cfg_dataset.name_feats.feat_dynamic_real)
        self.num_feat_static_cat = len(self.cfg_dataset.name_feats.feat_static_cat)
        self.num_feat_static_real = len(self.cfg_dataset.name_feats.feat_static_real)
        self.num_past_feat_dynamic_real = len(self.cfg_dataset.name_feats.past_feat_dynamic_real)
        self.num_feat_dynamic_cat = len(self.cfg_dataset.name_feats.feat_dynamic_cat)
        self.from_mlflow = from_mlflow
        self.model_config.context_length = (
            self.model_config.context_length
            if self.model_config.context_length is not None
            else self.model_config.prediction_length
        )

        if from_mlflow is not None:
            self.model = self.load_model(from_mlflow)
        else:
            self.model = None

        self.distr_output = distr_output
        self.loss = loss
        self.model_config.variable_dim = (
            self.model_config.variable_dim or self.model_config.d_models
        )
        self.model_config.static_cardinality = (
            self.model_config.static_cardinality
            if self.model_config.static_cardinality and self.num_feat_static_cat > 0
            else [1]
        )

        self.model_config.dynamic_cardinality = (
            self.model_config.dynamic_cardinality
            if self.model_config.dynamic_cardinality and self.num_feat_dynamic_cat > 0
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
        if self.from_mlflow is not None:
            logging.error("Model already trained, cannot be retrained from scratch")
            return
        self.model = None
        self.model = super().train(training_data=input_data)

    def predict(
        self, test_dataset: gluontsPandasDataset, validation=True
    ) -> Tuple[List[pd.Series], List[pd.Series]]:
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_dataset, predictor=self.model
        )
        forecasts_df = []
        for forecast in forecast_it:
            forecasts_df.append(
                utils_gluonts.sample_df(
                    forecast.samples,
                    periods=forecast.samples.shape[1],
                    start_date=forecast.start_date,
                    freq=forecast.freq,
                    ts_length=0,  # forecast.samples.shape[1],
                    pred_length=0,  # forecast.samples.shape[1],
                    validation=validation,
                )
            )
        if validation is False:
            list_it = []
            for ls in list(ts_it):
                list_it.append(ls[: -self.model_config.prediction_length])
        else:
            list_it = list(ts_it)

        return list_it, forecasts_df

    def get_callback_losses(self, type: str = "train") -> Dict[str, Any]:
        return self.callback.metrics["loss"][f"{type}_loss"]

    def evaluate(
        self,
        test_dataset: gluontsPandasDataset,
        mean: bool = False,
    ) -> Dict[str, Any]:
        true_ts, forecasts = self.predict(test_dataset)
        agg_metrics = {
            "mae": domain.metrics.estimate_mae(
                forecasts, true_ts, self.model_config.prediction_length
            ),
            "rmse": domain.metrics.estimate_rmse(
                forecasts, true_ts, self.model_config.prediction_length
            ),
            "mape": domain.metrics.estimate_mape(
                forecasts, true_ts, self.model_config.prediction_length
            ),
            "smape": domain.metrics.estimate_smape(
                forecasts, true_ts, self.model_config.prediction_length
            ),
            "wmape": domain.metrics.estimate_wmape(
                forecasts, true_ts, self.model_config.prediction_length
            ),
            "mase": domain.metrics.estimate_mase(
                forecasts, true_ts, self.model_config.prediction_length, self.freq
            ),
        }

        for i in range(1, 10):
            agg_metrics[f"QuantileLoss[{i/10}]"] = domain.metrics.quantileloss(
                forecasts, true_ts, self.model_config.prediction_length, i / 10
            )

        if mean:
            for name, value in agg_metrics.items():
                agg_metrics[name] = np.mean(value)

        return agg_metrics, true_ts, forecasts

    def create_transformation(self) -> Transformation:
        remove_field_names = []
        if self.num_feat_static_real == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if self.num_feat_dynamic_real == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        if self.num_past_feat_dynamic_real == 0:
            remove_field_names.append(FieldName.PAST_FEAT_DYNAMIC_REAL)
        if self.num_feat_dynamic_cat == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_CAT)

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
            + (
                [SetField(output_field=FieldName.PAST_FEAT_DYNAMIC_REAL, value=[0.0])]
                if not self.num_past_feat_dynamic_real > 0
                else []
            )
            + (
                [
                    SetField(
                        output_field=FieldName.FEAT_DYNAMIC_CAT,
                        value=np.zeros(self.cfg_dataset.ts_length, dtype=int),
                    )
                ]
                if not self.num_feat_dynamic_cat > 0
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

        ts_fields = [
            FieldName.FEAT_TIME,
            FieldName.FEAT_DYNAMIC_CAT,
        ]

        past_ts_fields = []
        if self.num_past_feat_dynamic_real > 0:
            past_ts_fields.append(FieldName.PAST_FEAT_DYNAMIC_REAL)

        return CustomTFTInstanceSplitter(
            instance_sampler=instance_sampler,
            past_length=module.model._past_length,
            future_length=self.model_config.prediction_length,
            time_series_fields=ts_fields,
            past_time_series_fields=past_ts_fields,
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
            num_past_feat_dynamic_real=self.num_past_feat_dynamic_real,
            num_feat_dynamic_cat=self.num_feat_dynamic_cat,
            distr_output=self.distr_output,
        )

        return TFTLightningModule(model=model, loss=self.loss)


class InformerForecaster(Forecaster):
    def __init__(
        self,
        cfg_model: configs.Model,
        cfg_train: configs.Train,
        cfg_dataset: configs.Dataset,
        from_mlflow: str = None,
    ) -> None:
        Forecaster.__init__(
            self,
            cfg_model=cfg_model,
            cfg_train=cfg_train,
            cfg_dataset=cfg_dataset,
            from_mlflow=from_mlflow,
        )
        self.from_mlflow = from_mlflow
        self.freq = self.cfg_dataset.freq
        time_features = time_features_from_frequency_str(self.freq)
        self.model_config_informer = CustomInformerConfig(
            num_time_features=len(time_features) + 1,
            cardinality=self.model_config.static_cardinality,
            num_dynamic_real_features=len(self.cfg_dataset.name_feats.feat_dynamic_real),
            num_static_categorical_features=len(self.cfg_dataset.name_feats.feat_static_cat),
            num_static_real_features=len(self.cfg_dataset.name_feats.feat_static_real),
            num_past_dynamic_real_features=len(self.cfg_dataset.name_feats.past_feat_dynamic_real),
            **self.model_config.dict(),
        )
        if from_mlflow is not None:
            self.model = self.load_model(from_mlflow)
            # CustomInformerForPrediction.from_pretrained(self.from_pretrained)
        else:
            self.model = CustomInformerForPrediction(self.model_config_informer)

    def get_train_dataloader(self, train_dataset: List[Dict[str, Any]]):
        logger.info("Create train dataloader")
        self.train_dataloader = create_train_dataloader(
            config=self.model_config_informer,
            freq=self.freq,
            data=train_dataset,
            batch_size=self.cfg_train.batch_size_train,
            num_batches_per_epoch=self.cfg_train.nb_batch_per_epoch,
            num_workers=2,
        )

    def get_test_dataloader(self, test_dataset: List[Dict[str, Any]], validation=True):
        logger.info("Create test dataloader")
        if validation is True:
            self.test_dataloader = create_validation_dataloader(
                config=self.model_config_informer,
                freq=self.freq,
                data=test_dataset,
                batch_size=self.cfg_train.batch_size_test,
            )
        else:
            self.test_dataloader = create_test_dataloader(
                config=self.model_config_informer,
                freq=self.freq,
                data=test_dataset,
                batch_size=self.cfg_train.batch_size_test,
            )

    def train(self, input_data: List[Dict[str, Any]]):
        if self.from_mlflow is not None:
            logger.error("Model already trained, cannot be retrained from scratch")
            return
        self.get_train_dataloader(input_data)
        accelerator = Accelerator()
        device = accelerator.device
        self.model.to(device)
        optimizer = AdamW(self.model.parameters(), **self.optimizer_config)

        self.model, optimizer, self.train_dataloader = accelerator.prepare(
            self.model,
            optimizer,
            self.train_dataloader,
        )

        self.loss_history = []
        self.model.train()

        for epoch in range(self.cfg_train.epochs):
            for idx, batch in enumerate(self.train_dataloader):
                optimizer.zero_grad()
                outputs = self.model(
                    static_categorical_features=batch["static_categorical_features"].to(device)
                    if self.model_config_informer.num_static_categorical_features > 0
                    else None,
                    static_real_features=batch["static_real_features"].to(device)
                    if self.model_config_informer.num_static_real_features > 0
                    else None,
                    past_time_features=batch["past_time_features"].to(torch.float32).to(device)
                    if device.type == "mps"
                    else batch["past_time_features"].to(device),
                    past_values=batch["past_values"].to(device),
                    future_time_features=batch["future_time_features"].to(torch.float32).to(device)
                    if device.type == "mps"
                    else batch["future_time_features"].to(device),
                    future_values=batch["future_values"].to(device),
                    past_observed_mask=batch["past_observed_mask"].to(device),
                    future_observed_mask=batch["future_observed_mask"].to(device),
                    past_dynamic_real_features=batch["past_dynamic_real_features"].to(device)
                    if self.model_config_informer.num_past_dynamic_real_features > 0
                    # if device.type == "mps"
                    # else batch["past_dynamic_real_features"].to(device),
                    else None,
                )
                loss = outputs.loss

                # Backpropagation
                accelerator.backward(loss)
                optimizer.step()

                self.loss_history.append(loss.item())
                # if idx % 100 == 0:
                # print(loss.item())

    def predict(
        self,
        test_dataset: List[Dict[str, Any]],
        transform_df=True,
        validation=True,
    ) -> Tuple[List[pd.Series], List[pd.Series]]:
        self.get_test_dataloader(test_dataset, validation)
        accelerator = Accelerator()
        device = accelerator.device

        self.model.to(device)
        self.model.eval()
        forecasts_ = []
        ts_it_ = []
        i = 0
        for batch in self.test_dataloader:
            outputs = self.model.generate(
                static_categorical_features=batch["static_categorical_features"].to(device)
                if self.model_config_informer.num_static_categorical_features > 0
                else None,
                static_real_features=batch["static_real_features"].to(device)
                if self.model_config_informer.num_static_real_features > 0
                else None,
                past_time_features=batch["past_time_features"].to(torch.float32).to(device)
                if device.type == "mps"
                else batch["past_time_features"].to(device),
                past_values=batch["past_values"].to(device),
                future_time_features=batch["future_time_features"].to(torch.float32).to(device)
                if device.type == "mps"
                else batch["future_time_features"].to(device),
                past_observed_mask=batch["past_observed_mask"].to(device),
                past_dynamic_real_features=batch["past_dynamic_real_features"].to(device)
                if self.model_config_informer.num_past_dynamic_real_features > 0
                else None,
                # if device.type == "mps"
                # else batch["past_dynamic_real_features"].to(device),
            )
            forecasts_.append(outputs.sequences.cpu().numpy())
            ts_it_.append(batch["past_values"].numpy())
            i = i + 1
        forecasts = np.vstack(forecasts_)
        ts_it = np.vstack(ts_it_)
        if not transform_df:
            return ts_it, forecasts
        # periods = len(test_dataset[0]["target"])
        df_ts = utils_gluonts.transform_huggingface_to_dict(test_dataset, freq=self.freq)
        forecasts_df = {}

        for i, forecast in enumerate(forecasts):
            forecasts_df[i] = utils_gluonts.sample_df(
                forecast,
                periods=forecast.shape[1],
                start_date=test_dataset[0]["start"],
                freq=self.freq,
                ts_length=len(test_dataset[0]["target"]),
                pred_length=self.model_config.prediction_length,
                validation=validation,
            )

        return df_ts, forecasts_df

    def get_callback_losses(self, type: str = "train") -> Dict[str, Any]:
        return self.loss_history

    def evaluate(
        self, test_dataset: List[Dict[str, Any]], forecasts=[]
    ) -> Tuple[Dict[str, Any], List[float]]:
        if len(forecasts) == 0:
            true_ts, forecasts = self.predict(test_dataset, transform_df=False)
        forecast_median = np.median(forecasts, 1)
        mase_metric = evaluate.load("evaluate-metric/mase")
        smape_metric = evaluate.load("evaluate-metric/smape")
        mase_metrics = []
        smape_metrics = []

        for item_id, ts in enumerate(test_dataset):
            training_data = ts["target"][: -self.model_config.prediction_length]
            ground_truth = ts["target"][-self.model_config.prediction_length :]
            mase = mase_metric.compute(
                predictions=forecast_median[item_id],
                references=np.array(ground_truth),
                training=np.array(training_data),
                periodicity=get_seasonality(self.freq),
            )
            mase_metrics.append(mase["mase"])

            smape = smape_metric.compute(
                predictions=forecast_median[item_id],
                references=np.array(ground_truth),
            )
            smape_metrics.append(smape["smape"])

        metrics = {}
        metrics["smape"] = smape_metrics
        metrics["mase"] = mase_metrics

        df_ts = pd.DataFrame(true_ts.T)
        forecasts_df = {}

        for i, forecast in enumerate(forecasts):
            forecasts_df[i] = utils_gluonts.sample_df(
                forecast,
                periods=forecast.shape[1],
                start_date=test_dataset[0]["start"],
                freq=self.freq,
                ts_length=len(test_dataset[0]["target"]),
                pred_length=self.model_config.prediction_length,
                validation=True,
            )

        return metrics, df_ts, forecasts_df
