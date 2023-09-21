import logging
from typing import Union

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection._split import _BaseKFold, _num_samples, indexable
from sklearn.utils.validation import _deprecate_positional_args

# pd.options.mode.chained_assignment = None  # default "warn"
logger = logging.getLogger(__name__)


# https://stackoverflow.com/questions/51963713/cross-validation-for-grouped-time-series-panel-data
# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class GroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum size for a single training set.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import GroupTimeSeriesSplit
    >>> groups = np.array(['a', 'a', 'a', 'a', 'a', 'a',\
                           'b', 'b', 'b', 'b', 'b',\
                           'c', 'c', 'c', 'c',\
                           'd', 'd', 'd'])
    >>> gtss = GroupTimeSeriesSplit(n_splits=3)
    >>> for train_idx, test_idx in gtss.split(groups, groups=groups):
    ...     print("TRAIN:", train_idx, "TEST:", test_idx)
    ...     print("TRAIN GROUP:", groups[train_idx],\
                  "TEST GROUP:", groups[test_idx])
    TRAIN: [0, 1, 2, 3, 4, 5] TEST: [6, 7, 8, 9, 10]
    TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a']\
    TEST GROUP: ['b' 'b' 'b' 'b' 'b']
    TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] TEST: [11, 12, 13, 14]
    TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b']\
    TEST GROUP: ['c' 'c' 'c' 'c']
    TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]\
    TEST: [15, 16, 17]
    TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b' 'c' 'c' 'c' 'c']\
    TEST GROUP: ['d' 'd' 'd']
    """

    @_deprecate_positional_args
    def __init__(self, n_splits=5, *, max_train_size=None):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if groups[idx] in group_dict:
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                (
                    "Cannot have number of folds={0} greater than" " the number of groups={1}"
                ).format(n_folds, n_groups)
            )
        group_test_size = n_groups // n_folds
        group_test_starts = range(n_groups - n_splits * group_test_size, n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []
            for train_group_idx in unique_groups[:group_test_start]:
                train_array_tmp = group_dict[train_group_idx]
                train_array = np.sort(
                    np.unique(np.concatenate((train_array, train_array_tmp)), axis=None),
                    axis=None,
                )
            train_end = train_array.size
            if self.max_train_size and self.max_train_size < train_end:
                train_array = train_array[train_end - self.max_train_size : train_end]
            for test_group_idx in unique_groups[
                group_test_start : group_test_start + group_test_size
            ]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(
                    np.unique(np.concatenate((test_array, test_array_tmp)), axis=None),
                    axis=None,
                )
            yield [int(i) for i in train_array], [int(i) for i in test_array]


def cv_timeseries(
    X: pd.DataFrame,
    y: pd.DataFrame,
    model: Union[LGBMRegressor, RandomForestRegressor, LinearRegression],
    training_minimum_window: int = 1,
    unit: str = "M",
    test_window: int = 1,
    date_col: str = "date",
) -> pd.DataFrame:
    """Crossvalidation on time series dataset based on date

    Args:
        X (pd.DataFrame): training data
        y (pd.DataFrame): target data
        model_name (sklearn model, optional): model name. Defaults to "Randomforest".
        training_minimum_window (str, optional): minimum training window to start with.
            Defaults to "1 month".
        test_window (str, optional): test window (equals to incremental window).
            Defaults to "1 month".

    Returns:
        pd.DataFrame: prediction of the folds of the training sets
    """
    if unit == "M":
        training_minimum_window = training_minimum_window * 30
        test_window = test_window * 30
    elif unit == "W":
        training_minimum_window = training_minimum_window * 7
        test_window = test_window * 7

    incremental_window = test_window
    training_start_date = X.index.get_level_values(date_col).min()
    training_end_date = training_start_date + pd.DateOffset(days=training_minimum_window)
    test_end_date = training_end_date + pd.DateOffset(days=test_window)

    test_df = pd.DataFrame()
    feature_importance = pd.DataFrame()
    while test_end_date <= X.index.get_level_values(date_col).max() + pd.DateOffset(days=1):
        logger.info(f"{training_start_date}, {training_end_date}, {test_end_date}")
        train_x = X[
            (X.index.get_level_values(date_col) >= training_start_date)
            & (X.index.get_level_values(date_col) < training_end_date)
        ]
        test_x = X[
            (X.index.get_level_values(date_col) >= training_end_date)
            & (X.index.get_level_values(date_col) < test_end_date)
        ]
        if test_x.size == 0:
            training_end_date = training_end_date + pd.DateOffset(days=incremental_window)
            test_end_date = training_end_date + pd.DateOffset(days=test_window)
            continue

        train_y = y[
            (y.index.get_level_values(date_col) >= training_start_date)
            & (y.index.get_level_values(date_col) < training_end_date)
        ]
        # logger.info("training")
        model.fit(train_x, train_y)
        feat_tmp = pd.DataFrame(
            {"feature": train_x.columns, "importance": model.feature_importances_}
        )
        # logger.info("feature importance")
        feature_importance = pd.concat([feat_tmp, feature_importance])
        test_x["y_pred"] = model.predict(test_x)
        test_df = pd.concat([test_df, test_x[["y_pred"]]])
        training_end_date = training_end_date + pd.DateOffset(days=incremental_window)
        test_end_date = training_end_date + pd.DateOffset(days=test_window)

    print(feature_importance)
    feature_importance = (
        feature_importance.groupby(["feature"])["importance"].sum()
        / feature_importance["importance"].sum()
    ).sort_values(ascending=False)
    return test_df, feature_importance
