import logging

import hydra
import mlflow
from utils import logging_mlflow
from configs import Configs
from domain import forecasters

# from domain.plots import plot_timeseries
from omegaconf import DictConfig, OmegaConf
from load.dataloaders import CustomDataLoader

from urllib.parse import urlparse

from mlflow_deploy.flavor import log_model, load_model


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfgHydra: DictConfig):
    # Convert hydra config to dict
    cfg = OmegaConf.to_object(cfgHydra)
    cfg: Configs = Configs(**cfg)
    # print(cfgHydra["dataset"]["enedis"]["dataset_name"])
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000/"
    )  # cfgHydra["_paths"]["mlflow"]["tracking_uri"])
    mlflow.set_experiment("test_1")  # cfgHydra["_paths"]["mlflow"]["experiment_name"])

    mlflow.start_run()
    for pipeline_name in list(cfgHydra.keys())[:-1]:
        with mlflow.start_run(nested=True, run_name=pipeline_name + "_yaml"):
            logging_mlflow.log_params_from_omegaconf_dict(cfgHydra[pipeline_name])

    with mlflow.start_run(nested=True, run_name="train"):
        mlflow.log_param("hydra_output_dir", hydra_output_dir)
        logging_mlflow.log_params_from_omegaconf_dict(cfgHydra["train"])

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            filename="example.log",
            filemode="w",
        )

        logging.info("Prepare Data")
        loader_data = CustomDataLoader(
            cfg_dataset=cfg.dataset,
            target=cfg.dataset.load["target"],
            feats=cfg.dataset.name_feats,
            cfg_model=cfg.model,
            test_length=cfg.dataset.test_length,
        )
        dataset = loader_data.get_dataset()

        logging.info("Training")
        forecaster_inst = getattr(forecasters, cfg.model.model_name)
        forecaster = forecaster_inst(
            cfg_model=cfg.model, cfg_train=cfg.train, cfg_dataset=cfg.dataset
        )

        forecaster.train(input_data=dataset.train)
        losses = forecaster.get_callback_losses(type="train")
        logging.info("first 10 losses")
        logging.info(losses[:10])

        # mlflow.log_metric("loss", losses[-1], step=0)
        mlflow.log_metrics({"loss": losses[-1]})

        ts_it, forecast_it = forecaster.predict(test_dataset=dataset.test)

        # logging.info(ts_it[:10])
        # logging.info(forecast_it.shape)
        logging.info(ts_it[0].tail())
        logging.info(forecast_it[0].head())

        # metrics = forecaster.evaluate(test_dataset=dataset.test)
        # logging.info(metrics)

        # logging.info("Plot first TS predictions")
        # plot_timeseries(
        #     0,
        #     uni_variate_dataset=dataset.test,
        #     prediction_length=cfg.model.model_config.prediction_length,
        #     forecasts=forecast_it,
        # )

        # log model
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

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

    last_run = mlflow.last_active_run().info.run_id

    mlflow.end_run()

    # last_run = '1d8d6ebceb7f43deb74ef5c2ee603aa5'
    print(last_run)
    logged_model = "runs:/" + str(last_run) + "/model"

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

    loaded_model = load_model(logged_model)
    # ts_it, forecast_it = loaded_model.predict(dataset=dataset.test)
    if cfg.model.model_name == "TFTForecaster":
        from gluonts.evaluation import make_evaluation_predictions
        from utils import utils_gluonts

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset.test, predictor=loaded_model
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

        # logging.info(ts_it[:10])
        # logging.info(forecast_it.shape)
        logging.info(list(ts_it)[0].tail())
        logging.info(forecasts_df[0].head())

    else:
        from accelerate import Accelerator
        import torch
        import numpy as np
        from domain.transformations import create_validation_dataloader
        from gluonts.time_feature import time_features_from_frequency_str
        from utils.utils_informer.configuration_informer import CustomInformerConfig
        from utils import utils_gluonts

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
            data=dataset.test,
            batch_size=cfg.train.batch_size_test,
        )
        transform_df = True
        accelerator = Accelerator()
        device = accelerator.device

        loaded_model.to(device)
        loaded_model.eval()
        forecasts_ = []
        ts_it_ = []
        i = 0
        for batch in test_dataloader:
            outputs = loaded_model.generate(
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
        df_ts = utils_gluonts.transform_huggingface_to_dict(dataset.test, freq=cfg.dataset.freq)
        forecasts_df = {}

        for i, forecast in enumerate(forecasts):
            forecasts_df[i] = utils_gluonts.sample_df(
                forecast,
                periods=forecast.shape[1],
                start_date=dataset.test[0]["start"],
                freq=cfg.dataset.freq,
                ts_length=len(dataset.test[0]["target"]),
                pred_length=cfg.model.model_config.prediction_length,
            )

        logging.info(df_ts[0].tail())
        logging.info(forecasts_df[0].head())


if __name__ == "__main__":
    main()
