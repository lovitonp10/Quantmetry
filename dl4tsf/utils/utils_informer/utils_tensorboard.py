import os
from torch.utils.tensorboard import SummaryWriter


def get_summary_writer(
    log_dir: str = "tensorboard_logs",
    dataset_name: str = "enedis",
    model_name: str = "Informer",
) -> SummaryWriter:
    # Define the base folder and file name
    directory = log_dir + "/" + dataset_name
    # file_name = 'result.txt'

    # Determine the next version number
    version_number = 1
    subfolder_name = f"version_{version_number}"
    subfolder_path = os.path.join(directory, subfolder_name)

    while os.path.exists(subfolder_path):
        version_number += 1
        subfolder_name = f"version_{version_number}"
        subfolder_path = os.path.join(directory, subfolder_name)

    # Create the subfolder
    os.makedirs(subfolder_path)

    # Set the new file path
    new_file_path = os.path.join(subfolder_path, model_name)

    writer = SummaryWriter(log_dir=new_file_path)

    return writer
