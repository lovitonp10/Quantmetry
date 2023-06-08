from mlflow_deploy.flavor import log_model, load_model
from gluonts.evaluation import make_evaluation_predictions
from utils import utils_gluonts
from accelerate import Accelerator
import torch
import numpy as np
from domain.transformations import create_validation_dataloader
from gluonts.time_feature import time_features_from_frequency_str
from utils.utils_informer.configuration_informer import CustomInformerConfig


def save_mlflow_model(tracking_url_type_store, forecaster, cfg):
    if tracking_url_type_store != "file":
        log_model(
            model=forecaster, artifact_path="model", registered_model_name=cfg.model.model_name
        )

    else:
        log_model(
            model=forecaster,
            artifact_path="model",
            registered_model_name=None,
        )


class MLflowModel:
    def __init__(
        self,
        last_run: str,
    ) -> None:
        self.last_run = last_run

    def load_mlflow_model(self):
        logged_model = "runs:/" + str(self.last_run) + "/model"
        self.loaded_model = load_model(logged_model)

    def predict_mlflow_model(self, test_dataset, config):
        if not hasattr(self, "loaded_model"):
            self.load_mlflow_model()

        model_name = config.model.model_name
        if model_name == "TFTForecaster":
            return self.predict_tft(test_dataset)
        elif model_name == "InformerForecaster":
            return self.predict_informer(test_dataset, config)

    def predict_tft(self, test_dataset):
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_dataset, predictor=self.loaded_model
        )
        forecasts_df = []
        for forecast in forecast_it:
            forecasts_df.append(
                utils_gluonts.sample_df(
                    forecast.samples,
                    periods=forecast.samples.shape[1],
                    start_date=forecast.start_date,
                    freq=forecast.freq,
                    ts_length=0,
                    pred_length=0,
                )
            )
        return list(ts_it), forecasts_df

    def predict_informer(self, test_dataset, cfg):
        time_features = time_features_from_frequency_str(cfg.dataset.freq)
        model_config_informer = CustomInformerConfig(
            num_time_features=len(time_features) + 1,
            cardinality=cfg.model.model_config.static_cardinality,
            num_dynamic_real_features=len(cfg.dataset.name_feats["feat_dynamic_real"]),
            num_static_categorical_features=len(cfg.dataset.name_feats["feat_static_cat"]),
            num_static_real_features=len(cfg.dataset.name_feats["feat_static_real"]),
            num_past_dynamic_real_features=len(cfg.dataset.name_feats["past_feat_dynamic_real"]),
            **cfg.model.model_config.dict(),
        )

        test_dataloader = create_validation_dataloader(
            config=model_config_informer,
            freq=cfg.dataset.freq,
            data=test_dataset,
            batch_size=cfg.train.batch_size_test,
        )
        transform_df = True
        accelerator = Accelerator()
        device = accelerator.device

        self.loaded_model.to(device)
        self.loaded_model.eval()
        forecasts_ = []
        ts_it_ = []
        i = 0
        for batch in test_dataloader:
            outputs = self.loaded_model.generate(
                static_categorical_features=batch["static_categorical_features"].to(device)
                if model_config_informer.num_static_categorical_features > 0
                else None,
                static_real_features=batch["static_real_features"].to(device)
                if model_config_informer.num_static_real_features > 0
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
                if model_config_informer.num_past_dynamic_real_features > 0
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
        df_ts = utils_gluonts.transform_huggingface_to_dict(test_dataset, freq=cfg.dataset.freq)
        forecasts_df = {}

        for i, forecast in enumerate(forecasts):
            forecasts_df[i] = utils_gluonts.sample_df(
                forecast,
                periods=forecast.shape[1],
                start_date=test_dataset[0]["start"],
                freq=cfg.dataset.freq,
                ts_length=len(test_dataset[0]["target"]),
                pred_length=cfg.model.model_config.prediction_length,
            )

        return df_ts, forecasts_df


"""import mlflow

# Set the tracking URI to the appropriate MLflow server
mlflow.set_tracking_uri("http://your_mlflow_server")

# Get the latest run for the specified experiment
experiment_id = "your_experiment_id"
runs = mlflow.search_runs(experiment_ids=experiment_id, order_by=["-start_time"],
max_results=1)

# Extract the run ID of the latest run
latest_run_id = runs.loc[0]["run_id"]

print("Latest run ID:", latest_run_id)"""
