import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import relativedelta
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import BallTree
from scipy.spatial.distance import cdist
from geopy.distance import distance as geodist
from typing import Dict, List
import math
import json
import os
import osmnx as ox
from unidecode import unidecode
from geopy.geocoders import Nominatim


class Weather:
    def __init__(
        self,
        path_weather: str = "data/all_weather/",
    ) -> None:
        self.path_weather = path_weather

    def map_region(self, latitude, longitude) -> str:
        geolocator = Nominatim(user_agent="my_reverse_geocoder")
        location = geolocator.reverse([latitude, longitude], exactly_one=True)
        address = location.raw["address"]
        region = address.get("state")
        return region

    def get_station_id(
        self, names: List[str] = ["ORLY"], with_regions: bool = False
    ) -> pd.DataFrame:
        df_stations = pd.read_csv(self.path_weather + "stations.txt", sep=";")
        # dict_stations = df_stations[["ID", "Nom"]].set_index("Nom")["ID"].to_dict()
        df_stations = df_stations[df_stations["Nom"].isin(names)]

        if with_regions:
            df_stations["region"] = str()
            for i in range(len(df_stations)):
                latitude = df_stations["Latitude"].iloc[i]
                longitude = df_stations["Longitude"].iloc[i]
                df_stations["region"].iloc[i] = self.map_region(latitude, longitude)

        # for _, name in zip(dict_stations, names):
        #    id_stations.append(dict_stations[name])
        # id_station = df_stations[df_stations["Nom"] == name]["ID"].iloc[0]

        return df_stations

    def load_weather(
        self,
        start: str = "30-01-2022",
        end: str = "1-02-2022",
        station_names: List[str] = ["ORLY"],
        freq: str = "30T",
        with_regions: bool = False,
    ) -> pd.DataFrame:
        df_stations = self.get_station_id(names=station_names, with_regions=with_regions)

        start_start_month = start.replace(day=1)
        end_start_month = end.replace(day=1)

        columns = ["numer_sta", "date"] + self.dynamic_features + self.cat_features

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
            df2 = df2[df2["numer_sta"].isin(df_stations["ID"])]
            df2 = df2.rename(columns={"numer_sta": "ID"})

            if with_regions:
                df2 = pd.merge(df2, df_stations[["ID", "region"]], on="ID", how="left")

            if i == 0:
                df2 = df2[df2["date"] >= start]

            if i == month_difference:
                df2 = df2[df2["date"] <= end]

            df = pd.concat([df, df2], ignore_index=True)

        df.drop(["ID", "numer_sta"], axis=1, inplace=True)

        # for feat in dynamic_features:
        #    df[feat] = (
        #        pd.to_numeric(df[feat], errors="coerce").fillna(method="ffill").astype(float)
        #    )
        df.replace("mq", np.nan, inplace=True)
        # df[cat_features] = df[cat_features].fillna(method="ffill").astype(int)

        df.set_index("date", inplace=True)

        # duplicates = df.index.duplicated()
        # df = df[~duplicates].copy()
        # df = df.resample(freq)
        # df = df.ffill()

        return df

    def generate_itemid_weather(
        self,
        df: pd.DataFrame,
        df_item_ids: List,
        item_col: str = "item_id",
        freq: str = "30T",
        with_regions: bool = False,
    ) -> pd.DataFrame:
        new_df = pd.DataFrame()
        if with_regions:
            df.region = df.region.replace("Nouvelle-Aquitaine", "Nouvelle Aquitaine")
            df.region = df.region.replace("Grand Est", "Grand-Est")

        for item_id in df_item_ids["item_id"]:
            tmp = df.copy()
            if with_regions:
                region = df_item_ids[df_item_ids["item_id"] == item_id]["region"][0]
                tmp = tmp[tmp["region"] == region]

            tmp[item_col] = item_id
            for feat in self.dynamic_features:
                tmp[feat] = (
                    pd.to_numeric(tmp[feat], errors="coerce").fillna(method="ffill").astype(float)
                )
            tmp[self.cat_features] = tmp[self.cat_features].fillna(method="ffill").astype(int)
            duplicates = tmp.index.duplicated()
            tmp = tmp[~duplicates].copy()
            tmp = tmp.resample(freq)
            tmp = tmp.ffill()

            new_df = pd.concat([new_df, tmp])
            del tmp
        return new_df[[item_col] + self.cat_features + self.dynamic_features]

    def add_weather(
        self,
        df: pd.DataFrame,
        weather: Dict[str, any] = {
            "path_weather": "data/all_weather/",
            "dynamic_features": ["t", "rr3", "pmer"],
            "cat_features": ["cod_tend"],
            "station_names": ["ORLY"],
        },
        prediction_length: int = 7,
        frequency: str = "30T",
        with_regions: bool = False,
    ) -> pd.DataFrame:
        self.dynamic_features = weather["dynamic_features"]
        self.cat_features = weather["cat_features"]
        station_names = weather["station_names"]

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
            freq=frequency,
            with_regions=with_regions,
            station_names=station_names,
        )

        df.index = df.index.tz_localize(None)
        weather.index = weather.index.tz_localize(None)

        df_item_ids = pd.DataFrame(df.drop_duplicates(subset=["item_id"])["item_id"])
        if with_regions:
            df_item_ids = df.drop_duplicates(subset=["item_id"])[["item_id", "region"]]
        weather = self.generate_itemid_weather(
            df=weather, df_item_ids=df_item_ids, freq=frequency, with_regions=with_regions
        )

        forecast_date_range = pd.date_range(
            start=last_date, periods=prediction_length + 1, freq=frequency
        )[1:].tz_localize(None)
        weather_forecast = weather.loc[forecast_date_range][
            self.dynamic_features + self.cat_features + ["item_id"]
        ]

        index_names = df.index.names
        weather.index.names = index_names
        df = df.reset_index()
        weather = weather.reset_index()

        merge_on_col = ["item_id"] + index_names

        df_merge = pd.merge(df, weather, on=merge_on_col, how="left")
        df_merge = df_merge.set_index(index_names)
        return df_merge, weather_forecast

    def count_days_for_pred(self, freq, pred_length):
        freq = pd.Timedelta(freq)
        # Calculer la durée totale en minutes
        total_minutes = freq.total_seconds() / 60 * pred_length
        # Calculer le nombre de jours
        days = math.ceil(total_minutes / (24 * 60))
        return days


class Aifluence:
    def __init__(
        self,
        path: str = "data/exo_idf_mobilites/",
    ) -> None:
        self.path = path

    def download_amenities(self, json_amenities):
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

    def save_amenities(self):
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

    def pairwise_distances(self, locations1, locations2, fast, max_radius=None):
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
        return pdist

    def create_amenities(self, df_amenities: pd.DataFrame):

        df_stations_idfm = pd.read_csv(self.path + "stations_idfm.csv", sep=",")
        infostation_idfm = df_stations_idfm.rename(columns={"nom_long": "station"})
        infostation_idfm = infostation_idfm.drop_duplicates(subset=["station"])
        infostation_idfm[["latitude", "longitude"]] = infostation_idfm["Geo Point"].str.split(
            ",", expand=True
        )
        infostation_idfm["latitude"] = infostation_idfm["latitude"].astype(float)
        infostation_idfm["longitude"] = infostation_idfm["longitude"].astype(float)
        infostation_idfm = infostation_idfm.loc[
            ~np.any(infostation_idfm[["longitude", "latitude"]].isnull(), axis=1), :
        ]
        infostation_idfm = infostation_idfm.set_index(["id"])
        df_stations = infostation_idfm[["latitude", "longitude"]]

        df_amenities = df_amenities[["amenity", "geometry"]]
        pattern = r"POINT \((-?\d+\.\d+) (-?\d+\.\d+)\)"
        df_tmp = df_amenities.geometry.str.extract(pattern)
        df_amenities[["longitude", "latitude"]] = df_tmp.values.astype(float)

        distances_array = self.pairwise_distances(df_stations, df_amenities, fast=True)
        distances_array = pd.DataFrame(
            distances_array, index=df_stations.index, columns=df_amenities.amenity
        )
        distances_array = distances_array.stack().rename("distance").reset_index()
        radius_influence = 1_000
        distances_array["in_neighbour"] = (distances_array["distance"] < radius_influence).astype(
            int
        )
        df = (
            distances_array.groupby(["station", "amenity"])
            .agg({"in_neighbour": "sum"})
            .reset_index()
        )
        df = df.pivot(index="station", columns="amenity", values="in_neighbour").reset_index()
        return df

    def get_amenities(self):
        df_amenities = pd.read_csv(self.path + "amenities_full", sep=",")
        df_amenities_final = self.create_amenities(df_amenities)

        df_amenities_final.rename(columns={"station": "STATION"}, inplace=True)
        df_amenities_final["STATION"] = df_amenities_final["STATION"].str.upper().apply(unidecode)
        df_amenities_final["STATION"] = df_amenities_final["STATION"].str.strip(" ")
        df_amenities_final["STATION"] = df_amenities_final["STATION"].str.replace(" - ", "-")
        return df_amenities_final

    def get_calendar(self):
        df_load = pd.read_csv(self.path + "Data_Outliers_with_calendar.csv", sep=",")
        df_load.set_index("JOUR", inplace=True, drop=True)
        df_load.index.name = None
        df_load = df_load.drop(columns=["DESCRIPTION_PH", "LIBELLE_ARRET", "NB_VALD"])
        return df_load

    def add_amenities(self):
        df_amenities = self.get_amenities()
        return df_amenities
