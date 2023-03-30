import logging

import hydra
from configs import Configs
from domain.plots import plot_timeseries
from domain.transformations import create_test_dataloader, create_train_dataloader

from gluonts.time_feature import time_features_from_frequency_str
from load.load_prepare import prepare_data
from omegaconf import DictConfig, OmegaConf
from transformers import InformerConfig, InformerForPrediction
from domain.metrics import estimate_mase_smape
from domain.train_model import train, inference


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

    batch_size_test = 3
    train_dataset, test_dataset = prepare_data(cfg=cfg)

    time_features = time_features_from_frequency_str(cfg.dataset.freq)
    config_uni = InformerConfig(num_time_features=len(time_features) + 1, **cfg.model.model_config)
    model_uni = InformerForPrediction(config_uni)

    # Create PyTorch DataLoaders
    logging.info("Create dataloaders")
    train_dataloader_uni = create_train_dataloader(
        config=config_uni,
        freq=cfg.dataset.freq,
        data=train_dataset,
        batch_size=cfg.train.batch_size_train,
        num_batches_per_epoch=cfg.train.nb_batch_per_epoch,
        num_workers=2,
    )

    test_dataloader_uni = create_test_dataloader(
        config=config_uni,
        freq=cfg.dataset.freq,
        data=test_dataset,
        batch_size=batch_size_test,
    )

    logging.info("Training")
    model_uni, losses = train(
        cfg=cfg, model=model_uni, config_model=config_uni, train_dataloader=train_dataloader_uni
    )
    model_uni.save_pretrained("../models/informer_v1")

    logging.info("Inference")
    model_uni = InformerForPrediction.from_pretrained("../models/informer_v1")

    forecasts = inference(
        model=model_uni, model_config=config_uni, test_dataloader=test_dataloader_uni
    )

    logging.info("Estimating MASE and SMAPE")
    smape_metrics, mase_metrics = estimate_mase_smape(
        cfg=cfg, forecasts=forecasts, test_dataset=test_dataset
    )
    print(smape_metrics)
    print(mase_metrics)

    logging.info("Plot first TS predictions")
    plot_timeseries(
        0,
        uni_variate_dataset=test_dataset,
        prediction_length=cfg.model.model_config["prediction_length"],
        forecasts=forecasts,
    )


if __name__ == "__main__":
    main()
