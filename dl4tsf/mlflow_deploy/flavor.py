from mlflow.models import Model

# from mlflow_flavor_example.utils import FakeModel
from domain.forecasters import Forecaster
from mlflow.models.model import MLMODEL_FILE_NAME
from pathlib import Path
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
import mlflow_deploy

FLAVOR_NAME = "my_flavor"


def save_model(
    model: Forecaster,
    path,
    mlflow_model=None,
):

    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)

    mlflow_mlmodel_file_path = path / MLMODEL_FILE_NAME
    model_subpath = path / "model.pkl"
    if mlflow_model is None:
        mlflow_model = Model()
    mlflow_model.add_flavor(FLAVOR_NAME, foo=123, bar="abc")  # , offset=my_model.offset)
    mlflow_model.save(mlflow_mlmodel_file_path)
    model.save(model_subpath)


def log_model(
    model: Forecaster,
    artifact_path,
    registered_model_name: None,
    **kwargs,
):
    return Model.log(
        artifact_path=str(artifact_path),  # must be string, numbers etc
        flavor=mlflow_deploy.flavor,  # points to this module itself
        model=model,
        registered_model_name=registered_model_name,
        **kwargs,
    )


def load_model(model_uri, dst_path=None):
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    model_subpath = Path(local_model_path) / "model.pkl"
    return Forecaster.load(model_subpath)
