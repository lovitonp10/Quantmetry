import json
import math
import os
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import osmnx as ox
import pandas as pd
from dateutil import relativedelta
from geopy.distance import distance as geodist
from scipy.spatial.distance import cdist
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import BallTree


class Weather:
    def __init__(
        self,
        path_weather: str = "data/all_weather/",
    ) -> None:
        self.path_weather = path_weather

    def get_station_id(self, name: str = "ORLY") -> int:
        df_stations = pd.read_csv(self.path_weather + "stations.txt", sep=";")
        dict_stations = df_stations[["ID", "Nom"]].set_index("Nom")["ID"].to_dict()
        id_station = dict_stations[name]
        # id_station = df_stations[df_stations["Nom"] == name]["ID"].iloc[0]

        return id_station

    def load_weather(
        self,
        start: str = "30-01-2022",
        end: str = "1-02-2022",
        dynamic_features: list = ["t", "rr3", "pmer"],
        cat_features: list = ["cod_tend"],
        station_name: str = "ORLY",
        freq: str = "30T",
    ) -> pd.DataFrame:
        station = self.get_station_id(name=station_name)

        start_start_month = start.replace(day=1)
        end_start_month = end.replace(day=1)

        columns = ["numer_sta", "date"] + dynamic_features + cat_features

        diff = relativedelta.relativedelta(end_start_month, start_start_month)

        month_difference = diff.years * 12 + diff.months

        df = pd.DataFrame(columns=columns)

        for i in range(month_difference + 1):
            date = start + relativedelta.relativedelta(months=i)

            YM_str = date.strftime("%Y%m")
            path_weather = self.path_weather + "synop." + YM_str + ".csv.gz"

            df2 = pd.read_csv(path_weather, sep=";", compression="gzip")
            df2.rename(
                columns={
                    "t": "temperature",
                    "pmer": "pressure",
                    "cod_tend": "barometric_trend",
                    "rr3": "rainfall",
                },
                inplace=True,
            )
            df2["date"] = df2["date"].apply(lambda x: datetime.strptime(str(x), "%Y%m%d%H%M%S"))
            df2 = df2[columns]
            df2 = df2[df2["numer_sta"] == station]

            if i == 0:
                df2 = df2[df2["date"] >= start]

            if i == month_difference:
                df2 = df2[df2["date"] <= end]

            df = pd.concat([df, df2], ignore_index=True)

        df.drop(["numer_sta"], axis=1, inplace=True)

        for feat in dynamic_features:
            df[feat] = (
                pd.to_numeric(df[feat], errors="coerce").fillna(method="ffill").astype(float)
            )
        df.replace("mq", np.nan, inplace=True)
        df[cat_features] = df[cat_features].fillna(method="ffill").astype(int)

        df.set_index("date", inplace=True)

        duplicates = df.index.duplicated()
        df = df[~duplicates].copy()
        df = df.resample(freq)
        df = df.ffill()

        return df

    def generate_itemid_weather(
        self, df: pd.DataFrame, item_ids: List, item_col: str = "item_id"
    ) -> pd.DataFrame:
        new_df = pd.DataFrame()
        for item_id in item_ids:
            tmp = df.copy()
            tmp[item_col] = item_id
            new_df = pd.concat([new_df, tmp])
            del tmp
        return new_df

    def add_weather(
        self,
        df: pd.DataFrame,
        weather: Dict[str, any] = {
            "path_weather": "data/all_weather/",
            "dynamic_features": ["t", "rr3", "pmer"],
            "cat_features": ["cod_tend"],
            "station_name": "ORLY",
        },
        prediction_length: int = 7,
        frequency: str = "30T",
    ) -> pd.DataFrame:
        dynamic_features = weather["dynamic_features"]
        cat_features = weather["cat_features"]
        station_name = weather["station_name"]

        unique_dates = df.index.unique()
        sorted_dates = unique_dates.sort_values(ascending=True)
        first_date = sorted_dates.min()
        last_date = sorted_dates.max()

        first_date_str = first_date.strftime("%d-%m-%Y")
        last_date_str = last_date.strftime("%d-%m-%Y")

        # frequency = pd.infer_freq(sorted_dates)

        forecast_days = self.count_days_for_pred(freq=frequency, pred_length=prediction_length)

        start = datetime.strptime(first_date_str, "%d-%m-%Y")
        end = datetime.strptime(last_date_str, "%d-%m-%Y") + timedelta(days=1 + forecast_days)

        weather = self.load_weather(
            start=start,
            end=end,
            dynamic_features=dynamic_features,
            cat_features=cat_features,
            station_name=station_name,
            freq=frequency,
        )

        df.index = df.index.tz_localize(None)
        weather.index = weather.index.tz_localize(None)
        weather = self.generate_itemid_weather(weather, item_ids=df["item_id"].unique())

        forecast_date_range = pd.date_range(
            start=last_date, periods=prediction_length + 1, freq=frequency
        )[1:].tz_localize(None)
        weather_forecast = weather.loc[forecast_date_range][
            dynamic_features + cat_features + ["item_id"]
        ]

        index_names = df.index.names
        weather.index.names = index_names
        df = df.reset_index()
        weather = weather.reset_index()

        df_merge = pd.merge(df, weather, on=["item_id"] + index_names, how="left")
        df_merge = df_merge.set_index(index_names)
        return df_merge, weather_forecast

    def count_days_for_pred(self, freq, pred_length):
        freq = pd.Timedelta(freq)
        # Calculer la durée totale en minutes
        total_minutes = freq.total_seconds() / 60 * pred_length
        # Calculer le nombre de jours
        days = math.ceil(total_minutes / (24 * 60))
        return days


class Amenities:
    def __init__(
        self,
        path: str = "data/exo_idf_mobilites/",
    ) -> None:
        self.path = path

    def download_amenities(self, json_amenities: json) -> pd.DataFrame:
        """Download amenities with json file

        Parameters
        ----------
        json_amenities : json
            json file

        Returns
        -------
        pd.DataFrame
            dataframe amenities download
        """
        list_amenities = [amenity for group in json_amenities.values() for amenity in group]
        lat_stLazare, lon_stLazare = 48.8763, 2.3254
        distance_from_stLazare = 100_000
        size_batch = 10
        start_batch = 0
        list_amenities_full = []
        cols = ["element_type", "osmid", "amenity", "geometry"]

        while start_batch < len(list_amenities):
            end_batch = min(len(list_amenities), start_batch + size_batch)
            amenities_batch = list_amenities[start_batch:end_batch]
            tags = {"amenity": amenities_batch}
            print(f"Querying OSMNX for {amenities_batch}...", end="", flush=True)
            amenities_full = ox.geometries.geometries_from_point(
                center_point=(lat_stLazare, lon_stLazare),
                dist=distance_from_stLazare,
                tags=tags,
            )
            amenities_full = amenities_full.reset_index()[cols].set_index("osmid")
            list_amenities_full.append(amenities_full)
            start_batch += size_batch
        amenities_full = pd.concat(list_amenities_full)
        is_poly = amenities_full["element_type"] != "Point"
        amenities_full["geometry"][is_poly] = amenities_full["geometry"][is_poly].apply(
            lambda x: x.centroid
        )
        amenities_full.drop(columns=["element_type"], inplace=True)

        return amenities_full

    def save_amenities(self) -> None:
        """Save amenities data download"""
        path_json_amenities = self.path + "amenities.json"
        with open(path_json_amenities, "r") as f:
            json_amenities = json.load(f)
        df_amenities = self.download_amenities(json_amenities)

        df_amenities.to_csv(self.path + "amenities_full", index=True)
        directory = "cache/"
        extension = ".json"

        for filename in os.listdir(directory):
            if filename.endswith(extension):
                file_path = os.path.join(directory, filename)
                os.remove(file_path)

    def pairwise_distances(
        self,
        locations1: pd.DataFrame,
        locations2: pd.DataFrame,
        fast: bool,
        max_radius: int = None,
    ) -> pd.DataFrame:
        """Matches stations in two datasets with respect to distances between GeoPoints

        Parameters
        ----------
        locations1 : pd.DataFrame
            first dataframe
        locations2 : pd.DataFrame
            second dataframe
        fast : bool
            boolean for the method to compute distance
        max_radius : int, optional
            max radius use for compute distance, by default None

        Returns
        -------
        pd.DataFrame
            dataframe with unique station
        """
        if fast:
            earth_radius = 6378140
            if max_radius is None:
                haversine_dist = DistanceMetric.get_metric("haversine")
                pdist = earth_radius * haversine_dist.pairwise(
                    np.radians(locations1[["latitude", "longitude"]].values),
                    np.radians(locations2[["latitude", "longitude"]].values),
                )
            else:
                ball = BallTree(
                    np.radians(locations2[["latitude", "longitude"]].values),
                    metric="haversine",
                    leaf_size=40,
                )
                (indices, dist) = ball.query_radius(
                    count_only=False,
                    return_distance=True,
                    X=np.radians(locations1[["latitude", "longitude"]].values),
                    r=max_radius / earth_radius,
                )
                pdist = np.full(shape=(len(locations1), len(locations2)), fill_value=np.nan)
                for ind_row, inds_col in enumerate(indices):
                    pdist[ind_row, inds_col] = dist[ind_row] * earth_radius

        else:
            pdist = cdist(
                locations1[["latitude", "longitude"]],
                locations2[["latitude", "longitude"]],
                lambda u, v: geodist(u, v).meters,
            )
        df = pd.DataFrame(pdist, index=locations1.index, columns=locations2.index)
        return df

    def pivot_amenity(self, df: pd.DataFrame, station_col: str, amenity_col: str) -> pd.DataFrame:
        """Pivot the amenities data

        Parameters
        ----------
        df : pd.DataFrame
            amenities data
        station_col : str
            column for station identification
        amenity_col : str
            name of json file for pivot on amenities

        Returns
        -------
        pd.DataFrame
            amenities data with pivot
        """
        df = df.groupby([station_col, amenity_col]).agg({"in_neighbour": "sum"}).reset_index()
        df = df.pivot(index=station_col, columns=amenity_col, values="in_neighbour").reset_index()
        df = df.set_index(station_col)
        df.columns = [f"{amenity_col}={col}" for col in df.columns]
        return df

    def load_stations_idfm(self, filename: str = "stations_idfm.csv"):
        df_stations_idfm = pd.read_csv(self.path + filename, sep=",")
        df_stations_idfm = df_stations_idfm.rename(columns={"nom_long": "station"})
        df_stations_idfm = df_stations_idfm.drop_duplicates(subset=["station"])
        df_stations_idfm[["latitude", "longitude"]] = df_stations_idfm["Geo Point"].str.split(
            ",", expand=True
        )
        df_stations_idfm["latitude"] = df_stations_idfm["latitude"].astype(float)
        df_stations_idfm["longitude"] = df_stations_idfm["longitude"].astype(float)
        df_stations_idfm = df_stations_idfm.loc[
            ~np.any(df_stations_idfm[["longitude", "latitude"]].isnull(), axis=1), :
        ]
        df_stations_idfm = df_stations_idfm.set_index(["id_ref_lda"])
        df_stations_idfm = df_stations_idfm[["latitude", "longitude"]]
        return df_stations_idfm

    def create_amenities(self, df_amenities: pd.DataFrame, amenities_type: str) -> pd.DataFrame:
        """Preprocess the amenities data and prepare for merge in dataset

        Parameters
        ----------
        df_amenities : pd.DataFrame
            amenities data

        Returns
        -------
        pd.DataFrame
            amenities data preprocess and ready for merge
        """
        radius_influence = 1_000

        df_stations_idfm = self.load_stations_idfm(filename="stations_idfm.csv")

        df_amenities = df_amenities[["amenity", "geometry"]]
        pattern = r"POINT \((-?\d+\.\d+) (-?\d+\.\d+)\)"
        df_tmp = df_amenities.geometry.str.extract(pattern)
        df_amenities[["longitude", "latitude"]] = df_tmp.values.astype(float)

        distances_array = self.pairwise_distances(
            df_stations_idfm, df_amenities.set_index("amenity"), fast=True
        )
        distances_array = distances_array.stack().rename("distance").reset_index()
        distances_array["in_neighbour"] = (distances_array["distance"] < radius_influence).astype(
            int
        )
        json_amenities = self.load_amenities_json(amenities_type=amenities_type)
        distances_array["amenity_type"] = distances_array["amenity"].map(json_amenities)

        df_json = self.pivot_amenity(
            distances_array, station_col="id_ref_lda", amenity_col="amenity_type"
        ).reset_index()
        return df_json

    def load_amenities_json(self, amenities_type: str):
        if amenities_type == "amenities_manual":
            path_json_amenities_manual = self.path + "amenities_manual.json"
            with open(path_json_amenities_manual, "r") as f:
                json_amenities_manual = json.load(f)

            inverse_json_amenities_manual = {
                item: k for k, v in json_amenities_manual.items() for item in v
            }
            return inverse_json_amenities_manual
        else:
            path_json_amenities = self.path + "amenities.json"
            with open(path_json_amenities, "r") as f:
                json_amenities = json.load(f)

            inverse_json_amenities = {item: k for k, v in json_amenities.items() for item in v}
            return inverse_json_amenities

    def get_amenities(self) -> pd.DataFrame:
        """Read the amenities data from csv, if does not exist, create it in _bronze file

        Returns
        -------
        pd.DataFrame
            amenities data
        """
        processed_amenities = "data/_bronze/amenities_idf.csv"
        if not os.path.isfile(processed_amenities):
            df_amenities = pd.read_csv(self.path + "amenities_full", sep=",")
            df_amenities_final = self.create_amenities(df_amenities, amenities_type="amenityJSON")
            df_amenities_final.to_csv(processed_amenities, index=False)
            return df_amenities_final
        return pd.read_csv(processed_amenities)

    def get_calendar(self) -> pd.DataFrame:
        """Read the calendar data

        Returns
        -------
        pd.DataFrame
            calendar data
        """
        df_load = pd.read_csv(self.path + "Data_Outliers_with_calendar.csv", sep=",")
        df_load.set_index("JOUR", inplace=True, drop=True)
        df_load.index.name = None
        df_load = df_load.drop(columns=["DESCRIPTION_PH", "LIBELLE_ARRET", "NB_VALD"])
        return df_load

    def add_amenities(self) -> pd.DataFrame:
        """Add amenities data

        Returns
        -------
        pd.DataFrame
            amenities data
        """
        df_amenities = self.get_amenities()
        df_amenities["id_ref_lda"] = df_amenities["id_ref_lda"].astype(int)
        df_amenities.rename(
            columns={
                "id_ref_lda": "ID_REFA_LDA",
            },
            inplace=True,
        )
        return df_amenities
