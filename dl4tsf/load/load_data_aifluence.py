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


def unstack_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Unstack the validations categories of aifluence dataframe

    Parameters
    ----------
    df_unstack : pd.DataFrame
        the validation dataset preprocess of aifluence

    Returns
    -------
    pd.DataFrame
        the validation dataset unstack by validations categories
    """

    indexes = ["DATE", "STATION", "CATEGORIE_TITRE"]
    df_group = df.groupby(indexes).sum()
    df_unstack = df_group.unstack(["CATEGORIE_TITRE"])
    new_columns = ["".join(map(str, col)) for col in df_unstack.columns.get_level_values(1)]
    df_unstack.columns = new_columns
    df_unstack = df_unstack.fillna(0)

    return df_unstack


def fusion_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Fusion the validations categories and compute the total validation

    Parameters
    ----------
    df : pd.DataFrame
        validation dataframe with all validations categories

    Returns
    -------
    pd.DataFrame
        a sample validation dataframe with 7 validations categories and the total validation
    """
    df_aifluence = df.copy()
    df_aifluence["AUTRE"] = (
        df_aifluence["AUTRE TITRE"] + df_aifluence["INCONNU"] + df_aifluence["NON DEFINI"]
    )
    df_aifluence = df_aifluence.drop(columns=["DATE", "INCONNU", "AUTRE TITRE", "NON DEFINI"])
    df_aifluence.rename(
        columns={
            "IMAGINE R": "IMAGINE_R",
            "NAVIGO JOUR": "NAVIGO_JOUR",
        },
        inplace=True,
    )
    df_aifluence["VALD_TOTAL"] = df_aifluence.sum(numeric_only=True, axis=1)

    return df_aifluence


def preprocess_station(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the time series by station

    Parameters
    ----------
    df : pd.DataFrame
        validation dataframe for each station

    Returns
    -------
    pd.DataFrame
        validation dataframe preprocess for each station
    """

    df_aifluence = df.copy()
    df_aifluence["STATION"] = df_aifluence["STATION"].str.strip(" ")
    group_station = df_aifluence.groupby(["STATION"]).size()
    select_station = group_station[group_station < 2300].index
    df_aifluence = df_aifluence[~df_aifluence["STATION"].isin(select_station)]
    df_resampled = df_aifluence.groupby(["STATION"]).resample("D").sum(numeric_only=True)
    df_aifluence = df_resampled.reset_index(level="STATION")

    return df_aifluence
