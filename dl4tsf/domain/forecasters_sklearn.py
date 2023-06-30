import logging

import numpy as np
import pandas as pd
import skopt
import utils.metrics as utils_metrics
import utils.training as training_utils
from configs import TrainConfig

# from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin

# from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error  # mean_absolute_error,
from utils import getters
from utils.training_spaces import models_spaces

logger = logging.getLogger(__name__)


class SklearnEstimator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        train_config: TrainConfig,
        scoring_search=mean_squared_error,
    ):

        self.target = train_config.target
        self.model_name = train_config.model_name
        self.model_config = train_config.model_config
        self.date_col = train_config.date_col
        self.error_metrics = train_config.error_metrics
        self.features = train_config.features
        self.cv_config = train_config.cv_config
        self.model = getattr(getters, self.model_name)(**self.model_config)
        self.error_funcs = [
            getattr(utils_metrics, error_metric) for error_metric in self.error_metrics
        ]
        self.scoring = scoring_search
        self.dict_space = {space.name: space for space in models_spaces[self.model_name]}

        if self.model_config == {}:
            logger.warning("Consider searching for params using searchcv method")

    def fit(self, X: pd.DataFrame, y=None):
        if len(self.features) == 0:
            self.features = X.columns.difference([self.target])
            self.model.fit(X[self.features], X[self.target])

    def predict_0_flights(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out_0 = X.copy()
        X_out_0["y_pred"] = 0
        return X_out_0[["y_pred"]]

    def predict(self, X: pd.DataFrame):

        X_out = X.copy()
        X_out["y_pred"] = np.clip(self.model.predict(X[self.features]), 0, np.inf)

        X_out["y_pred"] = X_out["y_pred"].round()
        return X_out[["y_pred"]]

    def evaluate_CV_timeseries(self, X):
        if len(self.features) == 0:
            self.features = X.columns.difference([self.target])

        X_out = X.copy()

        df_pred, feature_importance = training_utils.cv_timeseries(
            X=X_out[self.features],
            y=X_out[self.target],
            model=self.model,
            date_col=self.date_col,
            training_minimum_window=self.cv_config.training_minimum_window,
            test_window=self.cv_config.test_window,
            unit=self.cv_config.unit,
        )

        df_pred["y_pred"] = np.clip(df_pred["y_pred"], 0, np.inf)
        df_pred["y_pred"] = df_pred["y_pred"].round()
        df_pred[self.target] = X_out[self.target]
        return df_pred, feature_importance

    def score(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_out = X.copy()
        y_pred = self.predict(X_out)

        X_score = X_out[[self.target]]
        X_score = pd.concat([X_score, y_pred], axis=1)

        errs = {
            error_metric.name: round(
                error_func(X_score[self.target].values, X_score["y_pred"].values), 4
            )
            for error_metric, error_func in zip(self.error_metrics, self.error_funcs)
        }
        return errs

    def get_cvfolds(self, df: pd.DataFrame, n_splits: int = 5):
        groups = df.index.get_level_values(self.date_col).date
        gtss = training_utils.GroupTimeSeriesSplit(n_splits=n_splits)
        cv_folds = gtss.split(df, groups=groups)
        return cv_folds

    def objective(self, args) -> float:
        for param_name, param_value in zip(self.dict_space.keys(), args):
            setattr(self.model, param_name, param_value)

        errors = []
        for X_train, y_train, X_test, y_test in self.data_folds:
            self.model.fit(X_train, y_train)
            y_pred = np.clip(self.model.predict(X_test), 0, np.inf)
            y_pred = np.round(y_pred)
            errors.append(self.scoring(y_test[y_test > 0], y_pred[y_test > 0]))
        return np.mean(errors)

    def searchcv(
        self,
        df: pd.DataFrame,
        verbose: bool = True,
        n_splits: int = 5,
        n_calls: int = 10,
        random_state: int = 42,
    ) -> dict:
        if len(self.features) == 0:
            self.features = df.columns.difference([self.target])

        cv_folds = self.get_cvfolds(df, n_splits=n_splits)

        if self.dict_space == {}:
            return {}
        list_data_folds = list()
        for index_train, index_test in cv_folds:
            print(len(index_train), len(index_test))
            df_train = df.iloc[index_train]
            X_train = df_train[self.features]
            y_train = df_train[self.target]

            df_test = df.iloc[index_test]
            X_test = df_test[self.features]
            y_test = df_test[self.target]
            list_data_folds.append((X_train, y_train, X_test, y_test))
        self.data_folds = list_data_folds

        res = skopt.gp_minimize(
            self.objective,
            self.dict_space.values(),
            n_calls=n_calls,
            random_state=random_state,
            n_jobs=-1,
            verbose=verbose,
        )
        best_params = dict(zip(self.dict_space.keys(), res.x))

        if not self.model_config:
            self.model_config = best_params
            self.model = getattr(getters, self.model_name)(**self.model_config)
        return best_params
