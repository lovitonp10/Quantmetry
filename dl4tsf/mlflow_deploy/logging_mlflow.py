from typing import List

# import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import utils.metrics_sklearn as utils_metrics
from omegaconf import DictConfig, ListConfig


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig):
                _explore_recursive(f"{parent_name}.{k}", v)
            elif isinstance(v, ListConfig):
                if (k.startswith("feat_")) or (k.startswith("past_feat_")):
                    log_features(k, v)
                else:
                    _explore_recursive(
                        f"{parent_name}.{k}", ", ".join(str(element) for element in v)
                    )
            else:
                mlflow.log_param(f"{parent_name}.{k}", v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            if isinstance(v, DictConfig):
                _explore_recursive(f"{parent_name}.{i}", v)
            elif isinstance(v, ListConfig):
                _explore_recursive(f"{parent_name}.{i}", ", ".join(str(element) for element in v))
            else:
                mlflow.log_param(f"{parent_name}.{i}", v)
    else:
        if len(str(element)) > 199:
            element = "LENGTH exceeded"
        mlflow.log_param(parent_name, element)


def log_features(parent_name, v):
    for el in v:
        mlflow.log_param(f"{parent_name}.{el.replace('=', ' ')}", True)


def log_plots_lgbm(
    item_id: int,
    ts_it: List[pd.DataFrame],
    forecast_it: List[pd.DataFrame],
    map_item_id: pd.DataFrame,
    nb_past_pts: int,
    validation=True,
):
    pred_color = "green"

    forecast_it = forecast_it.groupby(["item_id", "date"])["y_pred"].mean()
    ts_it = ts_it.groupby(["item_id", "date"]).mean()
    ts_plot = ts_it.loc[item_id].tail(nb_past_pts)
    forecast_plot = forecast_it.loc[item_id].tail(nb_past_pts)

    title = ", ".join(
        str(value) for value in map_item_id[map_item_id["item_id"] == item_id].values
    )

    fig = plt.figure()
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.plot(ts_plot.index, ts_plot.values, color="blue", label="Target")
    plt.plot(forecast_plot.index, forecast_plot.values, color=pred_color, label="Forecast")
    plt.xticks(rotation=15)
    plt.suptitle(title)
    mae = round(utils_metrics.mae(ts_plot.values, forecast_plot.values), 2)
    wmape = round(utils_metrics.wmape(ts_plot.values, forecast_plot.values), 2)
    plt.title(f"MAE : {mae}, WMAPE: {wmape}")
    plt.legend()
    # plt.close()

    name = f"forecast_{item_id}.png"
    if validation:
        name = f"evaluation_{item_id}.png"

    plt.savefig(f"{name}_{item_id}.png")
    mlflow.log_figure(fig, name)
    plt.close()


def log_plots(
    item_id: int,
    ts_it: List[pd.DataFrame],
    forecast_it: List[pd.DataFrame],
    map_item_id: pd.DataFrame,
    nb_past_pts: int,
    validation=False,
):
    fig, ax = plt.subplots()
    ts_it = ts_it[item_id].tail(nb_past_pts)
    x0 = ts_it.index  # .to_timestamp()
    y0 = ts_it.values

    forecast_it = forecast_it.loc[item_id].mean(axis=1)
    if type(ts_it) == pd.DataFrame:
        ts_it = ts_it[0]
    last_value = ts_it[-1]  # .iloc[-1]
    last_index = x0[-1]
    pred_color = "red"

    if validation:
        last_value = ts_it[-len(forecast_it)]  # .iloc[-1]
        last_index = x0[-len(forecast_it)]
        pred_color = "green"

    last_value = pd.Series([last_value], index=[last_index])
    # Concatenate the new series with the second series
    forecast_it = pd.concat([last_value, forecast_it])

    x1, y1 = forecast_it.index, forecast_it.values

    ax.plot(x0, y0, color="blue", label="Target")
    ax.plot(x1, y1, color=pred_color, label="Forecast")
    ax.legend()

    title = ", ".join(str(value) for value in map_item_id[map_item_id["item_id"] == 0].values)
    ax.set_title(title)

    name = "forecast.png"
    if validation:
        name = "evaluation.png"
    mlflow.log_figure(fig, name)
