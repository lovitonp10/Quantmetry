import logging
from functools import partial

from configs import Configs
from datasets import load_dataset
from domain.transformations import UnivariateGrouper

from domain.transformations_pd import transform_start_field


def prepare_data(cfg: Configs):
    logging.info("Load dataset")
    dataset = load_dataset(cfg.dataset.repository_name, cfg.dataset.dataset_name)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    logging.info("Transform train and test")
    train_dataset.set_transform(partial(transform_start_field, freq=cfg.dataset.freq))
    test_dataset.set_transform(partial(transform_start_field, freq=cfg.dataset.freq))

    logging.info("Univariate grouper")
    # Convert to Univariate dict
    uni_variate_train_dataset = UnivariateGrouper(train_dataset)
    uni_variate_test_dataset = UnivariateGrouper(test_dataset)

    return uni_variate_train_dataset, uni_variate_test_dataset
