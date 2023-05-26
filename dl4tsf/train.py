import logging

import hydra
from configs import Configs
from domain import forecasters

# from domain.plots import plot_timeseries
from omegaconf import DictConfig, OmegaConf
from load.dataloaders import CustomDataLoader


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
    forecaster = forecaster_inst(cfg_model=cfg.model, cfg_train=cfg.train, cfg_dataset=cfg.dataset)

    forecaster.train(input_data=dataset.train)
    losses = forecaster.get_callback_losses(type="train")
    logging.info("first 10 losses")
    logging.info(losses[:10])

    logging.info("Validation")
    ts_it, forecast_it = forecaster.predict(test_dataset=dataset.validation, test_step=False)

    # logging.info(ts_it[:10])
    # logging.info(forecast_it.shape)
    logging.info(ts_it[0].tail())
    logging.info(forecast_it[0].head())

    logging.info("Test")
    ts_it, forecast_it = forecaster.predict(test_dataset=dataset.test, test_step=True)

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


if __name__ == "__main__":
    main()
