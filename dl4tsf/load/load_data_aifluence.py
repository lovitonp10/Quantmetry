import glob
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_validations(
    path: str = "data/idf_mobilites/",
) -> pd.DataFrame:
    """Read and concatenate validation files
    from folders containing files of validation history

    Args:
        history (dict): Dict of urls where to download
    Returns:
        df (pd.DataFrame): Concatenation of validations files
    """
    list_df = []  # list of datas download
    list_path = glob.glob(path + "/*")

    for path_folder in list_path:
        year = int(path_folder[-4:])
        path_files = glob.glob(path_folder + "/*")

        for file in path_files:
            if year == 2015:
                df_temp = pd.read_csv(file, sep=";", low_memory=False)
            else:
                df_temp = pd.read_csv(file, sep="\t", low_memory=False)
            list_df.append(df_temp)

    df = pd.concat(list_df)
    return df
