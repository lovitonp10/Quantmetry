import numpy as np
from gluonts.time_feature import get_seasonality
from gluonts.evaluation import metrics
from configs import Configs
from typing import Any, Dict, List, Tuple
from evaluate import load


def estimate_mae(forecasts: list, true_ts: list, prediction_length: float) -> list:

    """
    Compute the MAE metric:
    .. math::
        MAE = mean(|Y - hat{Y}|)
    """
    mae_metrics = []
    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-prediction_length:][0])
        forecast_value = np.array(forecast.median(axis=1))
        mae_metrics.append(metrics.abs_error(true_value, forecast_value) / prediction_length)

    return mae_metrics


def estimate_rmse(forecasts: list, true_ts: list, prediction_length: float) -> list:

    """
    Compute the RMSE metric:
    .. math::
        rmse = sqrt(mean((Y - hat{Y})^2))
    """

    rmse_metrics = []
    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-prediction_length:][0])
        forecast_value = np.array(forecast.mean(axis=1))
        mse_metrics = metrics.mse(true_value, forecast_value)
        rmse_metrics.append(mse_metrics ** (0.5))

    return rmse_metrics


def estimate_mape(
    forecasts: list, true_ts: list, prediction_length: float, pourcentage: bool
) -> list:

    """
    Compute the MAPE metric:
    .. math::
        mape = 100 * mean(|Y - hat{Y}| / |Y|))
    """
    mape_metrics = []
    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-prediction_length:][0])
        forecast_value = np.array(forecast.median(axis=1))
        if pourcentage:
            mape_metrics.append(100 * metrics.mape(true_value, forecast_value))
        else:
            mape_metrics.append(metrics.mape(true_value, forecast_value))

    return mape_metrics


def estimate_smape(
    forecasts: list,
    true_ts: list,
    prediction_length: float,
    pourcentage: bool,
) -> list:

    """
    Compute the SMAPE metric:
    .. math::
        smape = 200 * mean(|Y - hat{Y}| / (|Y| + |hat{Y}|))
    """

    smape_metrics = []
    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-prediction_length:][0])
        forecast_value = np.array(forecast.median(axis=1))
        if pourcentage:
            smape_metrics.append(100 * metrics.smape(true_value, forecast_value))
        else:
            smape_metrics.append(metrics.smape(true_value, forecast_value))

    return smape_metrics


def estimate_wmape(
    forecasts: list,
    true_ts: list,
    prediction_length: float,
    pourcentage: bool,
) -> list:

    """
    Compute the WMAPE metric:
    .. math::
        smape = 100 * sum(|Y - hat{Y}|) / sum(|Y|)
    """
    wmape_metrics = []
    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-prediction_length:][0])
        forecast_value = np.array(forecast.median(axis=1))
        if pourcentage:
            wmape_metrics.append(
                100 * np.sum(np.abs(true_value - forecast_value)) / np.sum(np.abs(true_value))
            )
        else:
            wmape_metrics.append(
                np.sum(np.abs(true_value - forecast_value)) / np.sum(np.abs(true_value))
            )

    return wmape_metrics


def estimate_mase(forecasts: list, true_ts: list, prediction_length: float, freq: str) -> list:

    """
    Compute the MASE metric:
    .. math::
        mase = mean(|Y - hat{Y}|) / seasonal_error
    """

    mase_metrics = []
    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-prediction_length:][0])
        forecast_value = np.array(forecast.median(axis=1))
        season_error = metrics.calculate_seasonal_error(
            past_data=np.array(ts[:-prediction_length][0]),
            freq=freq,
            seasonality=get_seasonality(freq),
        )
        mase_metrics.append(metrics.mase(true_value, forecast_value, season_error))

    return mase_metrics


def quantileloss(
    forecasts: list, true_ts: list, prediction_length: float, quantile: float
) -> list:

    """
    Compute the Quantile Loss metric:
    .. math::
        quantile_loss = 2 * sum(|(Y - hat{Y}) * (Y <= hat{Y}) - q|)
    """

    quantile_loss_metrics = []
    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-prediction_length:][0])
        forecast_quantile = np.array(forecast[int(quantile * 100)])
        quantile_loss_metrics.append(
            metrics.quantile_loss(true_value, forecast_quantile, quantile)
        )

    return quantile_loss_metrics


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
