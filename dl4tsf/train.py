import logging
import os
from urllib.parse import urlparse

import hydra
import mlflow

from configs import Configs
from domain import forecasters
from load.dataloaders import CustomDataLoader
from mlflow_deploy import logging_mlflow
from utils.utils_gluonts import get_mean_metrics

# from domain.plots import plot_timeseries
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)
logger.info("Start")


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfgHydra: DictConfig):
    # Convert hydra config to dict
    cfg = OmegaConf.to_object(cfgHydra)
    cfg: Configs = Configs(**cfg)

    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]

    azure_logger = "azure.core.pipeline.policies.http_logging_policy"
    logging.getLogger(azure_logger).setLevel(logging.WARNING)

    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = cfgHydra["logging"]["mlflow"][
        "AZURE_STORAGE_CONNECTION_STRING"
    ]
    mlflow.set_tracking_uri(cfgHydra["logging"]["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfgHydra["logging"]["mlflow"]["experiment_name"])

    mlflow.start_run(nested=True, run_name="train_" + cfg.dataset.dataset_name)
    mlflow.log_param("hydra_output_dir", hydra_output_dir)
    logging_mlflow.log_params_from_omegaconf_dict(cfgHydra["train"])
    for pipeline_name in list(cfgHydra.keys())[:-1]:
        logging_mlflow.log_params_from_omegaconf_dict(cfgHydra[pipeline_name])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filename="example.log",
        filemode="w",
    )

    logger.info("Prepare Data")
    loader_data = CustomDataLoader(
        cfg_dataset=cfg.dataset,
        target=cfg.dataset.load["target"],
        feats=cfg.dataset.name_feats,
        cfg_model=cfg.model,
        test_length=cfg.dataset.test_length,
    )
    dataset = loader_data.get_dataset()
    logger.info("Prepare Completed")

    logger.info("Training")
    forecaster_inst = getattr(forecasters, cfg.model.model_name)
    forecaster = forecaster_inst(cfg_model=cfg.model, cfg_train=cfg.train, cfg_dataset=cfg.dataset)
    forecaster.train(input_data=dataset.train)
    logger.info("Training Completed")

    logger.info("Compute First 10 Losses")
    losses = forecaster.get_callback_losses(type="train")
    logger.info(losses[:10])
    mlflow.log_metrics({"loss": losses[-1]})

    logger.info("Compute Validation & Evaluation")
    # ts_it, forecast_it = forecaster.predict(test_dataset=dataset.validation, validation=True)
    metrics, ts_it, forecast_it = forecaster.evaluate(test_dataset=dataset.validation)
    logger.info(ts_it[0].tail())
    logger.info(forecast_it[0].head())

    mlflow.log_metrics(get_mean_metrics(metrics))
    # logger.info(metrics)

    logger.info("Compute Prediction")
    ts_it, forecast_it = forecaster.predict(test_dataset=dataset.test, validation=False)

    logger.info(ts_it[0].tail())
    logger.info(forecast_it[0].head())

    # logger.info("Plot first TS predictions")
    # plot_timeseries(
    #     0,
    #     uni_variate_dataset=dataset.test,
    #     prediction_length=cfg.model.model_config.prediction_length,
    #     forecasts=forecast_it,
    # )
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    forecaster.save_mlflow_model(tracking_url_type_store, forecaster)

    last_run = mlflow.last_active_run().info.run_id
    print(last_run)
    mlflow.end_run()

    # logging.info("MLflow test")
    # forecaster = forecaster_inst(
    #     cfg_model=cfg.model, cfg_train=cfg.train, cfg_dataset=cfg.dataset, from_mlflow=last_run
    # )

    # ts_it, forecast_it = forecaster.predict(test_dataset=dataset.test, validation=False)

    # logging.info(ts_it[0].tail())
    # logging.info(forecast_it[0].head())


if __name__ == "__main__":
    main()
