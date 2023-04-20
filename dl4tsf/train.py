import logging

import hydra
from configs import Configs
from domain import forecasters
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
        cfg_dataset=cfg.dataset, target=cfg.dataset.load["target"], cfg_model=cfg.model
    )
    data_gluonts = loader_data.get_gluonts_format()

    logging.info("Training")
    forecaster_inst = getattr(forecasters, cfg.model.model_name)
    forecaster = forecaster_inst(cfg_model=cfg.model, cfg_train=cfg.train, cfg_dataset=cfg.dataset)
    forecaster.train(input_data=data_gluonts.train)
    losses = forecaster.get_callback_losses(type="train")

    logging.info("first 10 losses")
    logging.info(losses[:10])

    forecast_it, ts_it = forecaster.predict(test_data=data_gluonts.test)

    logging.info(forecast_it[0].head())
    logging.info(ts_it[0].head())


if __name__ == "__main__":
    main()
