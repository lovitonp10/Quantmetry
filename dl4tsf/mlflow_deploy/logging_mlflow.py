import mlflow
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
