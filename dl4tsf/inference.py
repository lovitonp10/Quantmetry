import logging

import hydra
from configs import Configs
from domain import forecasters
from load.dataloaders import CustomDataLoader

from omegaconf import DictConfig, OmegaConf
import os
import mlflow

logger = logging.getLogger(__name__)
logger.info("Start")


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfgHydra: DictConfig):
    # Convert hydra config to dict
    cfg = OmegaConf.to_object(cfgHydra)
    cfg: Configs = Configs(**cfg)
    # hydra_output_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]

    azure_logger = "azure.core.pipeline.policies.http_logging_policy"
    logging.getLogger(azure_logger).setLevel(logging.WARNING)

    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = cfgHydra["logging"]["mlflow"][
        "AZURE_STORAGE_CONNECTION_STRING"
    ]

    mlflow.set_tracking_uri(cfgHydra["logging"]["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfgHydra["logging"]["mlflow"]["experiment_name"])

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

    forecaster_inst = getattr(forecasters, cfg.model.model_name)

    logging.info("MLflow test")
    run_id = "f284c3af7eda4f8a95497df40cad016f"
    forecaster = forecaster_inst(
        cfg_model=cfg.model, cfg_train=cfg.train, cfg_dataset=cfg.dataset, from_mlflow=run_id
    )

    ts_it, forecast_it = forecaster.predict(test_dataset=dataset.test, validation=False)

    logging.info(ts_it[0].tail())
    logging.info(forecast_it[0].head())


if __name__ == "__main__":
    main()
