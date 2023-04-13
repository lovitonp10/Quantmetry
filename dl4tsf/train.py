import logging

import hydra
from configs import Configs
from domain import forecasters
from omegaconf import DictConfig, OmegaConf
from load.dataloaders import DataLoader


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
    loader_data = DataLoader(
        cfg_dataset=cfg.dataset,
        target="meantemp",
        cfg_model=cfg.model,
        test_length=114,
    )
    data_gluonts = loader_data.get_gluonts_format()

    logging.info("Training")
    model_inst = getattr(forecasters, cfg.model.model_name)
    model = model_inst(cfg_model=cfg.model, cfg_train=cfg.train, cfg_dataset=cfg.dataset)
    model.train(input_data=data_gluonts.train)

    forecast_it, ts_it = model.predict(test_data=data_gluonts.test)

    print(forecast_it[0].head())
    print(ts_it[0].head())


if __name__ == "__main__":
    main()
