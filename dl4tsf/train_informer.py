import logging

import hydra
import numpy as np
from configs import Configs
from domain import forecasters
from domain.plots import plot_timeseries
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
        target="toto",
        cfg_model=cfg.model,
        test_length=114,
    )
    # data_gluonts = loader_data.get_gluonts_format()
    data_huggingface = loader_data.get_huggingface_format()
    print(data_huggingface.train[0]["start"])

    logging.info("Training")
    model_inst = getattr(forecasters, cfg.model.model_name)
    model = model_inst(cfg_model=cfg.model, cfg_train=cfg.train, cfg_dataset=cfg.dataset)
    model_uni, losses = model.train(train_dataset=data_huggingface.train)
    print(losses)
    model_uni.save_pretrained("../models/informer_v1")

    logging.info("Inference")
    model_inst = getattr(forecasters, cfg.model.model_name)
    model = model_inst(
        cfg_model=cfg.model,
        cfg_train=cfg.train,
        cfg_dataset=cfg.dataset,
        from_pretrained="../models/informer_v1",
    )
    metrics, forecasts = model.score(test_dataset=data_huggingface.test)
    print(np.mean(metrics["smape"]), np.mean(metrics["mase"]))

    logging.info("Plot first TS predictions")
    plot_timeseries(
        0,
        uni_variate_dataset=data_huggingface.test,
        prediction_length=cfg.model.model_config.prediction_length,
        forecasts=forecasts,
    )


if __name__ == "__main__":
    main()
