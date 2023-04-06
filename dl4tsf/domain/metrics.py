import numpy as np
from evaluate import load
from gluonts.time_feature import get_seasonality
from configs import Configs
from typing import Any, Dict, List, Tuple


def estimate_mase_smape(
    cfg: Configs, forecasts, test_dataset: List[Dict[str, Any]]
) -> Tuple[List[float], List[float]]:
    forecast_median = np.median(forecasts, 1)
    mase_metric = load("evaluate-metric/mase")
    smape_metric = load("evaluate-metric/smape")
    mase_metrics = []
    smape_metrics = []

    for item_id, ts in enumerate(test_dataset):
        training_data = ts["target"][: -cfg.model.model_config["prediction_length"]]
        ground_truth = ts["target"][-cfg.model.model_config["prediction_length"] :]
        mase = mase_metric.compute(
            predictions=forecast_median[item_id],
            references=np.array(ground_truth),
            training=np.array(training_data),
            periodicity=get_seasonality(cfg.dataset.freq),
        )
        mase_metrics.append(mase["mase"])

        smape = smape_metric.compute(
            predictions=forecast_median[item_id],
            references=np.array(ground_truth),
        )
        smape_metrics.append(smape["smape"])
    return smape_metrics, mase_metrics
