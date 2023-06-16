import logging

import hydra
from configs import Configs
from domain import forecasters

# from domain.plots import plot_timeseries
from omegaconf import DictConfig, OmegaConf
from load.dataloaders import CustomDataLoader

logger = logging.getLogger(__name__)
logger.info("Start")


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfgHydra: DictConfig):

    # Convert hydra config to dict
    cfg = OmegaConf.to_object(cfgHydra)
    cfg: Configs = Configs(**cfg)
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

    logger.info("Compute Validation & Evaluation")
    # ts_it, forecast_it = forecaster.predict(test_dataset=dataset.validation, validation=True)
    metrics, ts_it, forecast_it = forecaster.evaluate(test_dataset=dataset.validation)
    logger.info(ts_it[0].tail())
    logger.info(forecast_it[0].head())

    logger.info(metrics)

    # logger.info(ts_it[:10])
    # logger.info(forecast_it.shape)

    logger.info("Compute Prediction")
    ts_it, forecast_it = forecaster.predict(test_dataset=dataset.test, validation=False)

    # logger.info(ts_it[:10])
    # logger.info(forecast_it.shape)
    logger.info(ts_it[0].tail())
    logger.info(forecast_it[0].head())

    # logger.info("Plot first TS predictions")
    # plot_timeseries(
    #     0,
    #     uni_variate_dataset=dataset.test,
    #     prediction_length=cfg.model.model_config.prediction_length,
    #     forecasts=forecast_it,
    # )
    logger.info("End")


if __name__ == "__main__":
    main()
