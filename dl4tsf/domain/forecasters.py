from typing import Iterable, Optional

import configs
import torch

# import pandas as pd

from domain.lightning_module import TFTLightningModule
from domain.module import TFTModel
from gluonts.core.component import validated

# from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
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
from torch.utils.data import DataLoader

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
        trainer_kwargs = cfg_train.trainer_kwargs
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

    def train(self, input_data):
        # step 1 transform input data from type XXXX (List[Dict]) to Dataset

        # df_temp = input_data[0]
        # first_time = df_temp["start"]
        # last_time = df_temp["start"]+len(df_temp["target"])-1
        # ind = pd.date_range(
        #     start = first_time.start_time,
        #     end = last_time.start_time,
        #     freq=self.freq)
        # df_pd = pd.DataFrame(data=df_temp, columns=['target'], index=ind)
        # df = PandasDataset(df_pd, target ="target", freq=self.freq)

        self.model = PyTorchLightningEstimator.train(training_data=input_data)
        loss = 0  # to modify
        return self.model, loss

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
