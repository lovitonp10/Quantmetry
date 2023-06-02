import glob
import pandas as pd


def load_validations(
    path: str = "data/idf_mobilites/",
) -> pd.DataFrame:
    """Read and concatenate validation files
    from folders containing files of validation history

    Parameters
    ----------
    path : str, optional
        links of folder where to load data, by default "data/idf_mobilites/"

    Returns
    -------
    pd.DataFrame
        concatenation of validations files
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
            del df_temp
    df = pd.concat(list_df)
    return df


def change_column_validations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """First preprocess : change the columns name and values below 5

    Parameters
    ----------
    df : pd.DataFrame
        dataFrame of validation load on the aifluence folder

    Returns
    -------
    pd.DataFrame
        validation dataframe with a modification for the columns and values
    """

    df_copy = df.copy()
    df_copy.replace(to_replace="Moins de 5", value=3, inplace=True)
    df_copy["NB_VALD"] = df_copy["NB_VALD"].astype(int)

    df_copy.rename(
        columns={
            "LIBELLE_ARRET": "STATION",
            "JOUR": "DATE",
        },
        inplace=True,
    )
    df_copy["CATEGORIE_TITRE"] = df_copy["CATEGORIE_TITRE"].replace("?", "INCONNU")
    df_copy["DATE"] = pd.to_datetime(df_copy["DATE"], dayfirst=True)

    return df_copy


def process_validation_titre(df: pd.DataFrame) -> pd.DataFrame:
    """Unstack and fusion the validations categories of aifluence dataframe

    Parameters
    ----------
    df : pd.DataFrame
        the validation dataset preprocess of aifluence

    Returns
    -------
    pd.DataFrame
        a sample validation dataframe with 7 validations categories and the total validation
    """

    indexes = ["DATE", "STATION", "CATEGORIE_TITRE"]
    df_group = df.groupby(indexes).sum()
    df_unstack = df_group.unstack(["CATEGORIE_TITRE"])
    new_columns = df_unstack.columns.map("_".join)
    df_unstack.columns = new_columns
    df_unstack = df_unstack.fillna(0)

    df_unstack_index = df_unstack.reset_index(level=["STATION", "DATE"])
    df_unstack_index.index = df_unstack_index["DATE"]

    df_unstack_index["NB_VALD_AUTRE"] = (
        df_unstack_index["NB_VALD_AUTRE TITRE"]
        + df_unstack_index["NB_VALD_INCONNU"]
        + df_unstack_index["NB_VALD_NON DEFINI"]
    )
    df_unstack_drop = df_unstack_index.drop(
        columns=["DATE", "NB_VALD_INCONNU", "NB_VALD_AUTRE TITRE", "NB_VALD_NON DEFINI"]
    )
    df_unstack_drop.rename(
        columns={
            "NB_VALD_IMAGINE R": "NB_VALD_IMAGINE_R",
            "NB_VALD_NAVIGO JOUR": "NB_VALD_NAVIGO_JOUR",
        },
        inplace=True,
    )
    df_unstack_drop["VALD_TOTAL"] = df_unstack_drop.sum(numeric_only=True, axis=1)

    return df_unstack_drop


def preprocess_station(df: pd.DataFrame, p_data_station: float) -> pd.DataFrame:
    """Preprocess the time series by station

    Parameters
    ----------
    df : pd.DataFrame
        validation dataframe for each station
    p_data_station : float
        proportion of data for each station

    Returns
    -------
    pd.DataFrame
        validation dataframe preprocess for each station
    """

    df_aifluence = df.copy()
    df_aifluence["STATION"] = df_aifluence["STATION"].str.strip(" ")
    group_station = df_aifluence.groupby(["STATION"]).size()

    time = df_aifluence.index
    size_time = max(time) - min(time)
    size_time_int = size_time.days + 1
    n_data_station = int(p_data_station * size_time_int)

    select_station = group_station[group_station < n_data_station].index
    df_aifluence = df_aifluence[~df_aifluence["STATION"].isin(select_station)]
    df_resampled = df_aifluence.groupby(["STATION"]).resample("D").sum(numeric_only=True)
    df_aifluence = df_resampled.reset_index(level="STATION")

    return df_aifluence


def cut_start_end_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Cuts the start and end of the time series

    Parameters
    ----------
    df : pd.DataFrame
        validation dataframe preprocess for each station

    Returns
    -------
    pd.DataFrame
        validation dataframe preprocess for each station
        with the same start and end time for each station
    """

    df_tmp = df.copy()
    df_tmp["DATE"] = df_tmp.index
    start_date = max(df_tmp.groupby(["STATION"]).min(numeric_only=False).reset_index()["DATE"])
    end_date = min(df_tmp.groupby(["STATION"]).max(numeric_only=False).reset_index()["DATE"])
    df_aifluence = df_tmp.loc[(df_tmp.index >= start_date) & (df_tmp.index <= end_date)]
    return df_aifluence
