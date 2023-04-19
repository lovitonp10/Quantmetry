import logging

import hydra
import numpy as np
from configs import Configs
from domain import forecasters
from domain.plots import plot_timeseries
from load.load_prepare import prepare_data
from omegaconf import DictConfig, OmegaConf


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
    train_dataset, test_dataset = prepare_data(cfg=cfg)

    logging.info("Training")
    model_inst = getattr(forecasters, cfg.model.model_name)
    model = model_inst(cfg_model=cfg.model, cfg_train=cfg.train, cfg_dataset=cfg.dataset)
    model_uni, losses = model.train(train_dataset=train_dataset)
    model_uni.save_pretrained("../models/informer_v1")

    logging.info("Inference")
    model_inst = getattr(forecasters, cfg.model.model_name)
    model = model_inst(
        cfg_model=cfg.model,
        cfg_train=cfg.train,
        cfg_dataset=cfg.dataset,
        from_pretrained="../models/informer_v1",
    )
    metrics, forecasts = model.score(test_dataset=test_dataset)
    print(np.mean(metrics["smape"]), np.mean(metrics["mase"]))

    logging.info("Plot first TS predictions")
    plot_timeseries(
        0,
        uni_variate_dataset=test_dataset,
        prediction_length=cfg.model.model_config["prediction_length"],
        forecasts=forecasts,
    )


if __name__ == "__main__":
    main()
