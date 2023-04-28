import numpy as np
from evaluate import load
from gluonts.time_feature import get_seasonality


def estimate_mae(cfg, forecasts, true_ts):
    pred_length = cfg.model.model_config.prediction_length
    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-cfg.model.model_config.prediction_length :][idx])
        foracast_value = np.array(forecast.median(axis=1))
        mae_metrics = (np.sum(np.abs(true_value - foracast_value))) / pred_length
    return mae_metrics


def estimate_rmse(cfg, forecasts, true_ts):
    pred_length = cfg.model.model_config.prediction_length
    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-cfg.model.model_config.prediction_length :][idx])
        forecast_value = np.array(forecast.median(axis=1))
        rmse_metrics = (np.sum((true_value - forecast_value) ** 2) / pred_length) ** (0.5)
    return rmse_metrics


def estimate_mape(cfg, forecasts, true_ts):
    pred_length = cfg.model.model_config.prediction_length
    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-cfg.model.model_config.prediction_length :][idx])
        forecast_value = np.array(forecast.median(axis=1))
        mape_metrics = (100 / pred_length) * np.sum(
            np.abs((true_value - forecast_value) / true_value)
        )
    return mape_metrics


def estimate_smape(cfg, forecasts, true_ts):
    pred_length = cfg.model.model_config.prediction_length
    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-cfg.model.model_config.prediction_length :][idx])
        forecast_value = np.array(forecast.median(axis=1))
        smape_metrics = (200 / pred_length) * np.sum(
            np.abs(true_value - forecast_value) / (np.abs(true_value) + np.abs(forecast_value))
        )
    return smape_metrics


def estimate_wmape(cfg, forecasts, true_ts):
    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-cfg.model.model_config.prediction_length :][idx])
        forecast_value = np.array(forecast.median(axis=1))
        wmape_metrics = (
            100 * np.sum(np.abs(true_value - forecast_value)) / np.sum(np.abs(true_value))
        )
    return wmape_metrics


def estimate_mase(cfg, forecasts, true_ts):
    mase_metric = load("evaluate-metric/mase")
    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-cfg.model.model_config.prediction_length :][idx])
        forecast_value = np.array(forecast.median(axis=1))
        mase = mase_metric.compute(
            predictions=forecast_value,
            references=true_value,
            training=np.array(ts[idx]),
            periodicity=get_seasonality(cfg.dataset.freq),
        )
        mase_metrics = mase["mase"]
    return mase_metrics


def estimate_mase2(cfg, forecasts, true_ts):
    mae = estimate_mae(cfg=cfg, forecasts=forecasts, true_ts=true_ts)
    pred_length = cfg.model.model_config.prediction_length
    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        freq = get_seasonality(cfg.dataset.freq)
        true_value = np.array(ts[-cfg.model.model_config.prediction_length :][idx])
        true_value_seasonal = np.array(ts[-cfg.model.model_config.prediction_length - freq :][idx])

        mae_naive = (
            1 / (pred_length - freq) * np.sum(true_value[freq:] - true_value_seasonal[-freq:])
        )
        mase2_metrics = mae / mae_naive
    return mase2_metrics
