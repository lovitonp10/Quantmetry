from typing import Any, Dict

import hydra
import pandas as pd
from domain.split_sklearn import split_train_test
from sklearn.pipeline import Pipeline
from utils.custom_objects_pydantic import HuggingFaceDataset


def apply_processing(cfg: Dict[str, Any], df: pd.DataFrame) -> HuggingFaceDataset:
    df_out = df.reset_index().copy()
    print(df_out.columns)

    preprocessors = []
    cfg_fe = cfg.pipeline
    for name in cfg_fe.keys():
        preprocessor = hydra.utils.instantiate(cfg_fe[name], _convert_="all")
        preprocessors.append((name, preprocessor))

    pipeline_feature = Pipeline(steps=preprocessors)
    df_flux = pipeline_feature.fit_transform(df_out, y=None)
    df_flux = df_flux.set_index(["date", "item_id"]+cfg.item_id_list)

    train, validation, test = split_train_test(df_flux, cfg=cfg.train_test_pred)
    dataset = HuggingFaceDataset(train=train, validation=validation, test=test)
    return dataset
