import numpy as np
from evaluate import load
from gluonts.time_feature import get_seasonality


def estimate_mase_smape(cfg, forecasts, test_dataset):
    forecast_median = np.median(forecasts, 1)
    mase_metric = load("evaluate-metric/mase")
    smape_metric = load("evaluate-metric/smape")
    mase_metrics = []
    smape_metrics = []
    i = 0
    for item_id, ts in enumerate(test_dataset):
        print(i)
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
        i = i + 1
    return smape_metrics, mase_metrics
