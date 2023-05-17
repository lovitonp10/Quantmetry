import os
import tempfile
import logging
import zipfile
import wget
import pandas as pd
from collections.abc import Mapping

logger = logging.getLogger(__name__)


def download_validations(path) -> pd.DataFrame:
    """Download and concatenate validation files
    from dictionary containing url of validation
    history

    Args:
        history (dict): Dict of urls where to download
    Returns:
        df (pd.DataFrame): Concatenation of validations files
    """

    # PART 1 : Download data on the website "data.iledefrance-mobilites.fr"
    prefix1 = "https://data.iledefrance-mobilites.fr/"
    prefix2 = "explore/dataset/histo-validations-reseau-ferre/files/"
    prefix = prefix1 + prefix2
    list_df = []  # list of datas download
    files = "histo-validations-reseau-ferre.json"  # Json file with url for download data
    history = pd.read_json(path + files, typ="series")  # Read  the json file

    temp_dir = tempfile.TemporaryDirectory(prefix="AIFL_")  # Create a temporal link
    logger.info(f"Creation of temporary directory: {temp_dir.name}")

    for dict_year in history:  # Read each link by years
        fields = dict_year["fields"]  # Save the information of url
        year = int(fields["annee"].split(" ")[-1])  # Save the year of the url
        for name_sem, semester in fields.items():  # Read each information in url
            if isinstance(semester, Mapping):
                if (year < 2019) or (
                    year == 2019 and name_sem == "semestre_1"
                ):  # Takes into account the particularity of the years after 2019

                    url = prefix + semester["id"] + "/download/"  # Create the complete name of url
                    logger.info(f"Importing file {semester['filename']} from: \n {url}")
                    path_file = os.path.join(temp_dir.name, semester["filename"])

                    try:
                        wget.download(url=url, out=path_file, bar=None)  # Download the file
                    except Exception as e:
                        logger.warning(f"Could not download file {url}")
                        logger.error(e)

                    # PART 2 : Unzip the file for take the data
                    zip_ref = zipfile.ZipFile(path_file, "r")
                    file_nb_fer = [
                        file for file in zip_ref.namelist() if "nb_fer" in file.lower()
                    ][0]
                    zip_ref.extract(file_nb_fer, path=temp_dir.name)

                    # All files are ".txt" format with "\t" separation
                    # Except 2015 files are ".csv" format with "";" separation
                    if year == 2015:
                        sep = ";"
                    else:
                        sep = "\t"

                    df_temp = pd.read_csv(os.path.join(temp_dir.name, file_nb_fer), sep=sep)
                    list_df.append(df_temp)
    df = pd.concat(list_df)
    temp_dir.cleanup()
    return df
