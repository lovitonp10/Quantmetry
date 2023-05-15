import numpy as np
from gluonts.time_feature import get_seasonality
from gluonts.evaluation import metrics


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


def estimate_mape(forecasts: list, true_ts: list, prediction_length: float) -> list:

    """
    Compute the MAPE metric:
    .. math::
        mape = 100 * mean(|Y - hat{Y}| / |Y|))
    """
    mape_metrics = []
    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-prediction_length:][0])
        forecast_value = np.array(forecast.median(axis=1))
        mape_metrics.append(100 * metrics.mape(true_value, forecast_value))

    return mape_metrics


def estimate_smape(forecasts: list, true_ts: list, prediction_length: float) -> list:

    """
    Compute the SMAPE metric:
    .. math::
        smape = 200 * mean(|Y - hat{Y}| / (|Y| + |hat{Y}|))
    """

    smape_metrics = []
    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-prediction_length:][0])
        forecast_value = np.array(forecast.median(axis=1))
        smape_metrics.append(100 * metrics.smape(true_value, forecast_value))

    return smape_metrics


def estimate_wmape(forecasts: list, true_ts: list, prediction_length: float) -> list:

    """
    Compute the WMAPE metric:
    .. math::
        smape = 100 * sum(|Y - hat{Y}|) / sum(|Y|)
    """
    wmape_metrics = []
    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-prediction_length:][0])
        forecast_value = np.array(forecast.median(axis=1))
        wmape_metrics.append(
            100 * np.sum(np.abs(true_value - forecast_value)) / np.sum(np.abs(true_value))
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
