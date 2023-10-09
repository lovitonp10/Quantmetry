import glob
import logging

import pandas as pd
from qolmat.imputations import imputers

logger = logging.getLogger(__name__)


class Enedis:
    def __init__(
        self,
        path: str = "data/enedis/",
        target: str = "total_energy",
    ) -> None:
        self.path = path
        self.target = target

    def load_data(self) -> pd.DataFrame:

        list_csv = glob.glob(self.path + "*.csv")
        df = pd.DataFrame()
        for file in list_csv:
            df_tmp = pd.read_csv(file)
            self.df = pd.concat([df, df_tmp], axis=0)

    def get_preprocessed_data(self):
        if not hasattr(self, "df"):
            self.load_data()

        df = self.change_columns_name(self.df)
        df = self.filter_data(df)
        df = self.add_power_values(df)
        return df

    def change_columns_name(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        df_out.rename(
            columns={
                "horodate": "date",
                "nb_points_soutirage": "soutirage",
                "total_energie_soutiree_wh": self.target,
                "plage_de_puissance_souscrite": "power",
            },
            inplace=True,
        )
        return df_out

    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.sort_values(by=["region", "profil", "power", "date"]).copy()
        df_out["date"] = pd.to_datetime(df_out["date"])
        df_out["hour_test"] = df_out.date.dt.hour
        df_out = df_out.set_index(["date","hour_test","profil","region","power"])
        imputer = imputers.ImputerLOCF(groups=["profil","region","power","hour_test"])
        df_out["total_energy"] = imputer.fit_transform(df_out[["total_energy"]])
        # df = df[["region", "profil", "power", self.target, "soutirage"]]
        #df_na = df_out[df_out.total_energy.isna()]
        # groups_with_nan = list(
        #     df_na[["region", "profil", "power"]]
        #     .drop_duplicates()
        #     .itertuples(index=False, name=None)
        # )
        # df_out = df_out[
        #     ~df_out.set_index(["region", "profil", "power"]).index.isin(groups_with_nan)
        # ]
        return df_out.reset_index().drop(columns= ["hour_test"]).set_index("date")

    def add_power_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        df_out["power_min"] = df_out["power"].str.extract(r"](\d+)-").fillna(0).astype(int)
        df_out["power_max"] = (
            df_out["power"]
            .str.extract(r"\-(\d+)]")
            .fillna(df_out["power"].str.extract(r"<= (\d+)"))
            .astype(int)
        )

        return df_out
