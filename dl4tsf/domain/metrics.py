import numpy as np
from gluonts.time_feature import get_seasonality
from gluonts.evaluation import metrics


def estimate_mae(cfg, forecasts, true_ts):

    """
    Compute the MAE metric:
    .. math::
        MAE = mean(|Y - hat{Y}|)
    """

    pred_length = cfg.model.model_config.prediction_length

    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-cfg.model.model_config.prediction_length :][idx])
        forecast_value = np.array(forecast.median(axis=1))
        mae_metrics = metrics.abs_error(true_value, forecast_value) / pred_length

    return mae_metrics


def estimate_rmse(cfg, forecasts, true_ts):

    """
    Compute the RMSE metric:
    .. math::
        rmse = sqrt(mean((Y - hat{Y})^2))
    """

    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-cfg.model.model_config.prediction_length :][idx])
        forecast_value = np.array(forecast.mean(axis=1))
        mse_metrics = metrics.mse(true_value, forecast_value)
        rmse_metrics = mse_metrics ** (0.5)

    return rmse_metrics


def estimate_mape(cfg, forecasts, true_ts):

    """
    Compute the MAPE metric:
    .. math::
        mape = 100 * mean(|Y - hat{Y}| / |Y|))
    """

    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-cfg.model.model_config.prediction_length :][idx])
        forecast_value = np.array(forecast.median(axis=1))
        mape_metrics = 100 * metrics.mape(true_value, forecast_value)

    return mape_metrics


def estimate_smape(cfg, forecasts, true_ts):

    """
    Compute the SMAPE metric:
    .. math::
        smape = 200 * mean(|Y - hat{Y}| / (|Y| + |hat{Y}|))
    """

    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-cfg.model.model_config.prediction_length :][idx])
        forecast_value = np.array(forecast.median(axis=1))
        smape_metrics = 100 * metrics.smape(true_value, forecast_value)

    return smape_metrics


def estimate_wmape(cfg, forecasts, true_ts):

    """
    Compute the WMAPE metric:
    .. math::
        smape = 100 * sum(|Y - hat{Y}|) / sum(|Y|)
    """

    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-cfg.model.model_config.prediction_length :][idx])
        forecast_value = np.array(forecast.median(axis=1))
        wmape_metrics = (
            100 * np.sum(np.abs(true_value - forecast_value)) / np.sum(np.abs(true_value))
        )
    return wmape_metrics


def estimate_mase(cfg, forecasts, true_ts):

    """
    Compute the MASE metric:
    .. math::
        mase = mean(|Y - hat{Y}|) / seasonal_error
    """

    for idx, (forecast, ts) in enumerate(zip(forecasts, true_ts)):
        true_value = np.array(ts[-cfg.model.model_config.prediction_length :][idx])
        forecast_value = np.array(forecast.median(axis=1))
        season_error = metrics.calculate_seasonal_error(
            past_data=np.array(ts[: -cfg.model.model_config.prediction_length][idx]),
            freq=cfg.dataset.freq,
            seasonality=get_seasonality(cfg.dataset.freq),
        )
        mase_metrics = metrics.mase(true_value, forecast_value, season_error)

    return mase_metrics
