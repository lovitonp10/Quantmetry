import glob
import pandas as pd
import logging

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
        df.rename(
            columns={
                "horodate": "date",
                "nb_points_soutirage": "soutirage",
                "total_energie_soutiree_wh": self.target,
                "plage_de_puissance_souscrite": "power",
            },
            inplace=True,
        )
        return df

    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(by=["region", "profil", "power", "date"])
        df.index = pd.to_datetime(df.date)
        df = df[["region", "profil", "power", self.target, "soutirage"]]
        df_na = df[df.total_energy.isna()]
        groups_with_nan = (
            df_na.groupby(["region", "profil", "power"]).apply(lambda x: x.any()).index.tolist()
        )
        df = df[~df.set_index(["region", "profil", "power"]).index.isin(groups_with_nan)]
        return df

    def add_power_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df["power_min"] = df["power"].str.extract(r"](\d+)-").fillna(0).astype(int)
        df["power_max"] = (
            df["power"]
            .str.extract(r"\-(\d+)]")
            .fillna(df["power"].str.extract(r"<= (\d+)"))
            .astype(int)
        )

        return df
