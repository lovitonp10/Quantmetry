import mlflow
from mlflow.tracking._tracking_service.utils import get_tracking_uri


logged_model = "runs:/4f158ab63e764c91a730a880edceac44/model"

"""from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from domain.forecasters import Forecaster
from pathlib import Path


def load_model(model_uri, dst_path=None):
    local_model_path = _download_artifact_from_uri(
        artifact_uri=model_uri, output_path=dst_path
    )
    model_subpath = Path(local_model_path) / "model.pkl"
    return Forecaster.load(model_subpath)"""


mlflow.set_tracking_uri("http://127.0.0.1:5000/")

print(get_tracking_uri())

# Load model
# loaded_model = load_model(logged_model)
experiment_id = "test_1"
# experiment_id = mlflow.get_experiment_by_name(experiment_id)
