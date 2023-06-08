import logging

import hydra
import mlflow
from configs import Configs
from domain import forecasters

# from domain.plots import plot_timeseries
from omegaconf import DictConfig, OmegaConf
from load.dataloaders import CustomDataLoader

from urllib.parse import urlparse

from mlflow_deploy import logging_mlflow
from mlflow_deploy.utils_mlflow import (
    save_mlflow_model,
    MLflowModel,
)


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

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    save_mlflow_model(tracking_url_type_store, forecaster, cfg)

    last_run = mlflow.last_active_run().info.run_id

    mlflow.end_run()

    mlflow_model = MLflowModel(last_run)
    mlflow_model.load_mlflow_model()

    ts_it, forecast_it = mlflow_model.predict_mlflow_model(
        test_dataset=dataset.test,
        config=cfg,
    )

    logging.info(ts_it[0].tail())
    logging.info(forecast_it[0].head())


if __name__ == "__main__":
    main()
