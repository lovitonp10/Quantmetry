import glob
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class Aifluence:
    def __init__(
        self,
        path: str = "data/idf_mobilites/",
    ) -> None:
        self.path = path

    def load_validations(self) -> pd.DataFrame:
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
        list_path = glob.glob(self.path + "/*")

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
        self.df = pd.concat(list_df)

    def get_preprocessed_data(self, start_date: str, end_date: str, p_data_station: float = 0.9):
        if not hasattr(self, "df"):
            self.load_validations()
        df_out = self.change_column_validations(self.df)
        df_out = df_out.drop(
            columns=["CODE_STIF_TRNS", "CODE_STIF_RES", "CODE_STIF_ARRET", "ID_REFA_LDA"]
        )
        # Add Amenities
        df_out = self.preprocess_validation_titre(df_out)
        df_out = self.preprocess_station(df_out, p_data_station)

        df_out = df_out.rename_axis("date")
        df_out = df_out.sort_values(by=["STATION", "date"])
        df_out = self.cut_start_end_ts(df_out, start=start_date, end=end_date)

        return df_out

    def change_column_validations(
        self,
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

    def preprocess_validation_titre(self, df: pd.DataFrame) -> pd.DataFrame:
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
        df_in = df.copy()
        indexes = ["DATE", "STATION", "CATEGORIE_TITRE"]
        df_in["CATEGORIE_TITRE"] = df_in["CATEGORIE_TITRE"].replace(
            ("AUTRE TITRE", "INCONNU", "NON DEFINI"), "AUTRE"
        )
        df_in["CATEGORIE_TITRE"] = df_in["CATEGORIE_TITRE"].str.replace(" ", "_")

        df_group = df_in.groupby(indexes).sum(numeric_only=False)
        df_unstack = df_group.unstack(["CATEGORIE_TITRE"])
        new_columns = df_unstack.columns.map("_".join)
        df_unstack.columns = new_columns
        df_unstack = df_unstack.fillna(0)

        df_unstack_index = df_unstack.reset_index(level=["STATION", "DATE"])
        df_unstack_index.index = df_unstack_index["DATE"]

        df_unstack_drop = df_unstack_index.drop(columns=["DATE"])
        df_unstack_drop["VALD_TOTAL"] = df_unstack_drop.sum(numeric_only=True, axis=1)

        return df_unstack_drop

    def preprocess_station(self, df: pd.DataFrame, p_data_station: float) -> pd.DataFrame:
        """Preprocess the time series by station
            Keep stations with at least `p_data_station` ratio of total length of station data

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
        df_resampled = df_aifluence.groupby(["STATION"]).resample("1D").sum()
        df_aifluence = df_resampled.reset_index(level="STATION")

        return df_aifluence

    def cut_start_end_ts(
        self, df: pd.DataFrame, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        """Cuts the start and end of the time series

        Parameters
        ----------
        df : pd.DataFrame
            validation dataframe preprocess for each station
        start : Optional[str], optional
            starting date of time series, by default None
        end : Optional[str], optional
            ending date of time series, by default None
        Returns
        -------
        pd.DataFrame
            validation dataframe preprocess for each station
            with the same start and end time for each station
        """

        df_tmp = df.copy()
        df_tmp["date"] = df_tmp.index
        if start is None:
            start_date = max(
                df_tmp.groupby(["STATION"]).min(numeric_only=False).reset_index()["date"]
            )
        else:
            start_date = pd.to_datetime(start, dayfirst=True)
        if end is None:
            end_date = min(
                df_tmp.groupby(["STATION"]).max(numeric_only=False).reset_index()["date"]
            )
        else:
            end_date = pd.to_datetime(end, dayfirst=True)
        df_aifluence = df_tmp.loc[(df_tmp.index >= start_date) & (df_tmp.index <= end_date)]
        df_aifluence = df_aifluence.drop(columns=["date"])
        return df_aifluence
