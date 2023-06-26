import matplotlib.pyplot as plt
import mlflow
import pandas as pd
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


def log_plots(item_id, ts_it, forecast_it, map_item_id, nb_past_pts, validation=False):
    fig, ax = plt.subplots()
    ts_it = ts_it[item_id].tail(nb_past_pts)
    x0 = ts_it.index.to_timestamp()
    y0 = ts_it.values
    last_value = ts_it[0].iloc[-1]
    last_index = x0[-1]

    forecast_it = forecast_it[item_id].mean(axis=1)
    if validation is False:
        last_value = pd.Series([last_value], index=[last_index])
        # Concatenate the new series with the second series
        forecast_it = pd.concat([last_value, forecast_it])

    x1, y1 = forecast_it.index, forecast_it.values
    ax.plot(x0, y0, color="blue", label="Target")
    ax.plot(x1, y1, color="red", label="Forecast")
    ax.legend()

    title = ", ".join(str(value) for value in map_item_id[map_item_id["item_id"] == 0].values)
    ax.set_title(title)

    name = "forecast.png"
    if validation:
        name = "evaluation.png"
    mlflow.log_figure(fig, name)
