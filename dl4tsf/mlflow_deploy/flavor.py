from pathlib import Path

import mlflow_deploy
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME

FLAVOR_NAME = "my_forecaster"


def save_model(
    model,
    path,
    mlflow_model=None,
):

    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)

    mlflow_mlmodel_file_path = path / MLMODEL_FILE_NAME
    model_subpath = path / "model.pkl"
    if mlflow_model is None:
        mlflow_model = Model()
    mlflow_model.add_flavor(FLAVOR_NAME)
    mlflow_model.save(mlflow_mlmodel_file_path)
    model.save(model_subpath)


def register_model_mlflow(
    model,
    artifact_path: str,
    model_name: None,
    **kwargs,
):
    return Model.log(
        artifact_path=str(artifact_path),  # must be string, numbers etc
        flavor=mlflow_deploy.flavor,  # points to this module itself
        model=model,
        registered_model_name=model_name,
        **kwargs,
    )


"""import mlflow

# Set the tracking URI to the appropriate MLflow server
mlflow.set_tracking_uri("http://your_mlflow_server")

# Get the latest run for the specified experiment
experiment_id = "your_experiment_id"
runs = mlflow.search_runs(experiment_ids=experiment_id, order_by=["-start_time"],
max_results=1)

# Extract the run ID of the latest run
latest_run_id = runs.loc[0]["run_id"]

print("Latest run ID:", latest_run_id)"""
