import numpy as np
from skopt import space

SPACE_ET = [
    space.Integer(low=200, high=300, name="n_estimators"),
    space.Integer(low=2, high=20, name="min_samples_split"),
    space.Categorical([True, False], name="bootstrap"),
]

SPACE_RF = [
    space.Integer(low=200, high=900, name="n_estimators"),
    space.Integer(low=2, high=50, name="min_samples_split"),
    space.Real(0.01, 1.0, name="min_samples_leaf"),
    space.Real(0.01, 1.0, name="max_samples"),
    space.Integer(2, np.inf, name="max_leaf_nodes"),
    space.Real(0.0, 1.0, name="ccp_alpha"),
    space.Integer(2, np.inf, name="max_depth"),
    space.Categorical(["auto", "sqrt", "log2"], name="max_features"),
]


SPACE_GB = [
    space.Integer(low=200, high=300, name="n_estimators"),
    space.Integer(low=2, high=20, name="min_samples_split"),
]

SPACE_LGBM = [
    space.Categorical(["gbdt"], name="boosting_type"),
    space.Integer(low=100, high=900, name="n_estimators"),
    space.Integer(low=50, high=500, name="num_leaves"),
    space.Integer(low=5, high=100, name="min_child_samples"),
    space.Real(low=0.2, high=1, name="subsample"),
    space.Real(low=0.2, high=1, name="colsample_bytree"),
    space.Real(low=1, high=10, prior="log-uniform", name="min_child_weight"),
    space.Real(low=1e-8, high=10, prior="log-uniform", name="reg_alpha"),
    space.Real(low=1e-8, high=10, prior="log-uniform", name="reg_lambda"),
]

SPACE_XGB = [
    space.Integer(low=100, high=200, name="n_estimators"),
    space.Integer(low=2, high=20, name="min_child_samples"),
]


models_spaces = {
    "ExtraTreesRegressor": SPACE_ET,
    "lightgbm.LGBMRegressor": SPACE_LGBM,
    "XGBRegressor": SPACE_XGB,
    "RandomForestRegressor": SPACE_RF,
}
