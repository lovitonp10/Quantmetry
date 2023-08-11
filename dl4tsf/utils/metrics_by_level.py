import logging
from typing import Dict, List, Tuple

import pandas as pd
import utils.metrics_sklearn as utils_metrics

log = logging.getLogger(__name__)


def calculate_metric_bilevel(
    df: pd.DataFrame,
    target: str,
    metrics: List[str] = ["mae"],
    group1: str = "flight_type",
    group2: str = "code_airport",
) -> pd.DataFrame:
    err_group = {}
    df_err = pd.DataFrame(columns=["metric", group1, group2, "value"])
    for metric in metrics:
        metric_func = getattr(utils_metrics, metric)
        for val1 in df.index.get_level_values(group1).unique():
            for val2 in df.index.get_level_values(group2).unique():
                df_filter = df.query(f"({group1}=='{val1}') & ({group2}=='{val2}')")
                err_group = metric_func(df_filter[target], df_filter["y_pred"])
                df_err.loc[len(df_err.index)] = [metric.name, val1, val2, err_group]
    return df_err.set_index(["metric", group1, group2])


def calculate_metric_unilevel(
    df: pd.DataFrame, target: str, metrics: List[str] = ["mae"], group: str = "flight_type"
) -> pd.DataFrame:
    err_group = {}
    df_err = pd.DataFrame(columns=["metric", group, "value"])
    for metric in metrics:
        metric_func = getattr(utils_metrics, metric)
        for val in df.index.get_level_values(group).unique():
            df_filter = df[df.index.get_level_values(group) == val]
            err_group = metric_func(df_filter[target], df_filter["y_pred"])
            df_err.loc[len(df_err.index)] = [metric.name, val, err_group]
    return df_err.set_index(["metric", group])


def calculate_metric_level(
    df: pd.DataFrame,
    target: str,
    metrics: List[str],
    group1: str = "flight_type",
    group2: str = None,
):

    if (not group1) and (not group2):
        return None
    if not group2:
        return calculate_metric_unilevel(df, target, metrics, group=group1)
    elif not group1:
        return calculate_metric_unilevel(df, target, metrics, group=group2)
    else:
        return calculate_metric_bilevel(df, target, metrics, group1=group1, group2=group2)


def calculate_metrics_byflowvalue(
    df: pd.DataFrame,
    target: str,
    metrics: List[str],
    group1: str = "flight_type",
    group2: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    value = 0
    df_filtered_0 = df[df[target] == value]
    df_filtered_diff0 = df[df[target] != value]

    df_err_0 = calculate_metric_level(
        df_filtered_0, metrics=metrics, target=target, group1=group1, group2=group2
    )

    df_err_diff0 = calculate_metric_level(
        df_filtered_diff0, metrics=metrics, target=target, group1=group1, group2=group2
    )

    return df_err_0, df_err_diff0


def calculate_metrics(
    df: pd.DataFrame, target: str, metrics: List[str], prefix=""
) -> Dict[str, float]:

    error_funcs = [getattr(utils_metrics, error_metric) for error_metric in metrics]
    errs = {
        f"{prefix}{error_metric.name}": error_func(df[target], df["y_pred"])
        for error_metric, error_func in zip(metrics, error_funcs)
    }

    return errs


def calculate_log_errors(
    df: pd.DataFrame, target: str, metrics: List[str], output_dir: str = "", prefix=""
):

    errs = calculate_metrics(df=df, target=target, metrics=metrics, prefix=prefix)
    log.info(f"Errors on all Test {prefix}")
    log.info(errs)

    df_err = calculate_metric_level(df, target, metrics, "STATION", None)

    return errs, df_err
    # df_err_0, df_err_diff0 = calculate_metrics_byflowvalue(
    #     df, target, metrics, group1="flight_type", group2="code_airport"
    # )
    # df_err.to_csv(f"{output_dir}/{prefix}errors.csv", index=True)
    # df_err_0.to_csv(f"{output_dir}/{prefix}errors_PMR=0.csv", index=True)
    # df_err_diff0.to_csv(f"{output_dir}/{prefix}errors_PMR!=0.csv", index=True)

    # mlflow.log_metrics(errs)
    # mlflow.log_metrics({"PMR_0_" + k: v for k, v in errs_PMR_0.items()})
    # mlflow.log_metrics({"PMR_diff0_" + k: v for k, v in errs_PMR_diff_0.items()})
