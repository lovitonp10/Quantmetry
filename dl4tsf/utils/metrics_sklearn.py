import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


def rmse(true, pred):
    return mean_squared_error(true, pred, squared=False)


def mse(true, pred):
    return mean_squared_error(true, pred, squared=True)


def mae(true, pred):
    return mean_absolute_error(true, pred)


def rmsse(train, train_pred, test, test_pred):
    forecast_mse = mse(test, test_pred)
    train_mse = mse(train, train_pred)
    return np.sqrt(forecast_mse / train_mse)


def mape(true, pred):
    return mean_absolute_percentage_error(true, pred)


def smape(true, pred):
    return np.mean((np.abs(true - pred)) / (np.abs(true) + np.abs(pred)))
