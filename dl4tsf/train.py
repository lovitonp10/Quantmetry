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
        cfg_model=cfg.model,
        test_length=cfg.dataset.test_length,
    )
    # dataset = loader_data.get_gluonts_format()
    dataset = loader_data.get_huggingface_format()

    logging.info("Training")
    forecaster_inst = getattr(forecasters, cfg.model.model_name)
    forecaster = forecaster_inst(cfg_model=cfg.model, cfg_train=cfg.train, cfg_dataset=cfg.dataset)

    forecaster.train(input_data=dataset.train)
    logging.info(forecaster.get_callback_losses())

    ts_it, forecast_it = forecaster.predict(test_dataset=dataset.test)

    # logging.info(ts_it[:10])
    # logging.info(forecast_it.shape)
    metrics = forecaster.evaluate(test_dataset=dataset.test)
    logging.info(metrics)

    # logging.info("Plot first TS predictions")
    # plot_timeseries(
    #     0,
    #     uni_variate_dataset=dataset.test,
    #     prediction_length=cfg.model.model_config.prediction_length,
    #     forecasts=forecast_it,
    # )


if __name__ == "__main__":
    main()
