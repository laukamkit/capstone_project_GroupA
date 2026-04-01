import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class NSWDataLoader:
    def __init__(self, train_size: float = 0.6, val_size: float = 0.2):
        self.nsw_path = self._get_repo_path()
        self.train_size = train_size
        self.val_size = val_size

    def _get_repo_path(self) -> str:
        # try different approaches to get the repo path.
        # based on all our individual modelling notebooks.
        possible_paths = [
            os.path.join("capstone_project_GroupA", "data", "NSW"),
            str(Path.cwd().resolve().parents[0] / 'data' / 'NSW'),
            os.path.join(str(os.getcwd()), "data", "NSW"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                print(f'Found NSW data path: {path}')
                return path
        raise FileNotFoundError("NSW data path not found.")

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads the NSW dataset from CSV files. If the processed CSV files do not exist, it extracts and processes the raw data from zip files.
        Returns:
            A tuple containing six pandas DataFrames: train, validation, test, train_scaled, val_scaled, test_scaled.
        """
        if not os.path.exists(os.path.join(self.nsw_path, "train.csv")):
            self.extract_data_from_zip()
        train = pd.read_csv(os.path.join(self.nsw_path, "train.csv"), index_col=0, parse_dates=True)
        validation = pd.read_csv(os.path.join(self.nsw_path, "validation.csv"), index_col=0, parse_dates=True)
        test = pd.read_csv(os.path.join(self.nsw_path, "test.csv"), index_col=0, parse_dates=True)
        train_scaled = pd.read_csv(os.path.join(self.nsw_path, "train_scaled.csv"), index_col=0, parse_dates=True)
        val_scaled = pd.read_csv(os.path.join(self.nsw_path, "val_scaled.csv"), index_col=0, parse_dates=True)
        test_scaled = pd.read_csv(os.path.join(self.nsw_path, "test_scaled.csv"), index_col=0, parse_dates=True)
        return train, validation, test, train_scaled, val_scaled, test_scaled

    def extract_data_from_zip(self) -> None:
        """This function performs the following steps:
        1. Reads the raw CSV files (demand, temperature, forecast) from the provided zip files.
        2. Converts date columns to datetime format and rounds to nearest 30 minutes.
        3. Processes the forecast data to pivot it into a wide format (P01, P02, ..., P48).
        4. Merges the demand, temperature, and forecast datasets on the DATETIME column.
        5. Cleans the merged dataset by reindexing to a complete 30-minute timeline and interpolating missing values.
        6. Computes error metrics (RMSE and TSE) for the forecasts.
        7. Adds lag features for demand (1 day ago, 1 week ago, 1 year ago).
        8. Splits the final dataset into train, validation, and test sets based on the specified proportions.
        9. Saves the processed datasets to CSV files for later use.
        """
        def _compute_merged_se(df_merged: pd.DataFrame) -> pd.DataFrame:
            p_cols = [f"P{i:02d}" for i in range(1, 49)]
            # Compute squared errors
            sq_errors = (df_merged[p_cols].sub(df_merged["TOTALDEMAND"], axis=0)) ** 2
            # (1) RMSE per datetime
            df_merged["RMSE_48"] = np.sqrt(sq_errors.mean(axis=1))
            # (2) Total squared error (sum of squared errors)
            df_merged["TSE_48"] = sq_errors.sum(axis=1)
            return df_merged

        def _clean_merged(df_merged: pd.DataFrame) -> pd.DataFrame:
            if "REGIONID" in df_merged.columns:
                df_merged = df_merged.drop(columns=["REGIONID"])

            # Create full 30-min time index
            full_index = pd.date_range(
                start=df_merged.index.min(),
                end=df_merged.index.max(),
                freq="30min"
            )

            # Reindex to full timeline
            df_merged = df_merged.reindex(full_index)
            df_merged.index.name = "DATETIME"

            # =========================================================
            # Handle missing values
            # =========================================================

            # Interpolate numeric columns using time method
            numeric_cols = ["TOTALDEMAND", "TEMPERATURE",
            "P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10",
            "P11", "P12", "P13", "P14", "P15", "P16", "P17", "P18", "P19", "P20",
            "P21", "P22", "P23", "P24", "P25", "P26", "P27", "P28", "P29", "P30",
            "P31", "P32", "P33", "P34", "P35", "P36", "P37", "P38", "P39", "P40",
            "P41", "P42", "P43", "P44", "P45", "P46", "P47", "P48"]

            df_merged[numeric_cols] = (
                df_merged[numeric_cols]
                .interpolate(method="time")
                .ffill()
                .bfill()
            )
            assert not df_merged.isnull().sum().any(), "There are still missing values after interpolation!"
            return df_merged

        def _merge_datasets(df_demand: pd.DataFrame, df_temp: pd.DataFrame, df_forecast: pd.DataFrame) -> pd.DataFrame:
            df_merged = df_demand.merge(df_temp, on="DATETIME", how="inner")
            df_merged = df_merged.merge(df_forecast, on="DATETIME", how="left")
            df_merged = df_merged.sort_values("DATETIME") # Sort by time
            df_merged = df_merged.set_index("DATETIME")   # Set as time index
            df_merged = df_merged.asfreq("30min")         # Ensure frequency is set
            return df_merged

        def _process_forecast_data(df_forecast: pd.DataFrame, demand_start_date: pd.Timestamp, demand_end_date: pd.Timestamp) -> pd.DataFrame:
            # =========================================================
            # 1. Handle duplicates (same DATETIME + PERIODID)
            # =========================================================
            # Keep only PERIODID 1–48
            df_forecast_48 = df_forecast[df_forecast["PERIODID"].between(1, 48)].copy()
            # Keep only relevant columns
            df_forecast_48 = df_forecast_48[["TARGET_DATETIME", "PERIODID", "FORECASTDEMAND"]]
            # Rename TARGET_DATETIME → DATETIME
            df_forecast_48 = df_forecast_48.rename(columns={"TARGET_DATETIME": "DATETIME"})
            df_forecast_48 = (
                df_forecast_48
                .groupby(["DATETIME", "PERIODID"], as_index=False)["FORECASTDEMAND"]
                .mean()
            )

            # =========================================================
            # 2. Pivot to wide format → P01 ... P48
            # =========================================================
            df_forecast_wide = df_forecast_48.pivot(
                index="DATETIME",
                columns="PERIODID",
                values="FORECASTDEMAND"
            )
            df_forecast_wide.columns = [f"P{int(c):02d}" for c in df_forecast_wide.columns]
                    
            # =========================================================
            # 3. Build full datetime index (same as demand)
            # =========================================================
            forecast_index = pd.date_range(
                start=demand_start_date,
                end=demand_end_date,
                freq="30min"
            )
            df_forecast_wide = (
                df_forecast_wide
                .reindex(forecast_index)
            )
            df_forecast_wide.index.name = "DATETIME"

            # =========================================================
            # 4. Interpolate missing values (IMPORTANT)
            # =========================================================
            # Interpolate each horizon separately (column-wise)
            df_forecast_wide = df_forecast_wide.interpolate(method="time")
            # Fill any edge missing values
            df_forecast_wide = df_forecast_wide.ffill().bfill()

            # =========================================================
            # 5. Reset index
            # =========================================================
            df_forecast_wide = df_forecast_wide.reset_index()
            return df_forecast_wide

        def _convert_to_date_time(df_demand: pd.DataFrame, df_temp: pd.DataFrame, df_forecast: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            df_demand["DATETIME"] = pd.to_datetime(df_demand["DATETIME"], dayfirst=True)
            df_temp["DATETIME"] = pd.to_datetime(df_temp["DATETIME"], dayfirst=True)
            df_forecast["LASTCHANGED"] = pd.to_datetime(df_forecast["LASTCHANGED"])
            df_forecast["TARGET_DATETIME"] = (
                df_forecast["LASTCHANGED"] + pd.to_timedelta(df_forecast["PERIODID"] * 30, unit="m")
            )
            return df_demand, df_temp, df_forecast

        def _round_df_to_30min(df_demand: pd.DataFrame, df_temp: pd.DataFrame, df_forecast: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            df_demand["DATETIME"] = df_demand["DATETIME"].dt.floor("30min")
            df_temp["DATETIME"] = df_temp["DATETIME"].dt.floor("30min")
            df_forecast["TARGET_DATETIME"] = df_forecast["TARGET_DATETIME"].dt.floor("30min")
            return df_demand, df_temp, df_forecast

        def _add_lag_features(df_merged: pd.DataFrame) -> pd.DataFrame:
            df_merged["demand_1_day_ago"] = df_merged["TOTALDEMAND"].shift(24)
            df_merged["demand_1_week_ago"] = df_merged["TOTALDEMAND"].shift(168)
            df_merged["demand_1_year_ago"] = df_merged["TOTALDEMAND"].shift(8760)
            df_model = df_merged.dropna(subset=[
                "demand_1_day_ago",
                "demand_1_week_ago",
                "demand_1_year_ago"
            ])
            return df_model
        
        def _train_val_test_split(df_model: pd.DataFrame, base_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            n = len(df_model)
            train_size = int(n * self.train_size)
            val_size = int(n * self.val_size)
            train       = df_model[base_features].iloc[:train_size]
            validation  = df_model[base_features].iloc[train_size:train_size + val_size]
            test        = df_model[base_features].iloc[train_size + val_size:]
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train[base_features])
            val_scaled = scaler.transform(validation[base_features])
            test_scaled = scaler.transform(test[base_features])
            train_scaled = pd.DataFrame(train_scaled, columns=base_features, index=train.index)
            val_scaled = pd.DataFrame(val_scaled, columns=base_features, index=validation.index)
            test_scaled = pd.DataFrame(test_scaled, columns=base_features, index=test.index)
            return train, validation, test, train_scaled, val_scaled, test_scaled

        part_a = os.path.join(self.nsw_path, "forecastdemand_nsw.csv.zip.partaa")
        part_b = os.path.join(self.nsw_path, "forecastdemand_nsw.csv.zip.partab")
        forecast_zip = os.path.join(self.nsw_path, "forecastdemand_nsw.csv.zip")

        with open(forecast_zip, "wb") as outfile:
            for p in [part_a, part_b]:
                with open(p, "rb") as infile:
                    outfile.write(infile.read())

        df_demand = pd.read_csv(os.path.join(self.nsw_path, "totaldemand_nsw.csv.zip"))
        df_temp = pd.read_csv(os.path.join(self.nsw_path, "temperature_nsw.csv.zip"))
        df_forecast = pd.read_csv(os.path.join(self.nsw_path, "forecastdemand_nsw.csv.zip"))
        df_demand, df_temp, df_forecast = _convert_to_date_time(df_demand, df_temp, df_forecast)
        df_demand, df_temp, df_forecast = _round_df_to_30min(df_demand, df_temp, df_forecast)
        df_forecast = _process_forecast_data(df_forecast, df_demand["DATETIME"].min(), df_demand["DATETIME"].max())
        df_temp = df_temp.groupby("DATETIME", as_index=False)[["TEMPERATURE"]].mean()
        df_merged = _merge_datasets(df_demand, df_temp, df_forecast)
        df_merged = _clean_merged(df_merged)
        df_merged = _compute_merged_se(df_merged)
        df_model = _add_lag_features(df_merged)

        n = len(df_model)
        base_features = ["TOTALDEMAND",
                    "demand_1_day_ago",
                    "demand_1_week_ago",
                    "demand_1_year_ago",
                    "TEMPERATURE",
                    "RMSE_48",
                    "TSE_48"]
        train, validation, test, train_scaled, val_scaled, test_scaled = _train_val_test_split(df_model, base_features)
        # Create folder if it does not exist
        os.makedirs(self.nsw_path, exist_ok=True)

        # Save main modelling dataset
        df_model[base_features].to_csv(os.path.join(self.nsw_path, "df_model.csv"))

        # Save train / validation / test splits
        train.to_csv(os.path.join(self.nsw_path, "train.csv"))
        validation.to_csv(os.path.join(self.nsw_path, "validation.csv"))
        test.to_csv(os.path.join(self.nsw_path, "test.csv"))

        # Save scaled datasets (used for LSTM and PatchTST models)
        train_scaled.to_csv(os.path.join(self.nsw_path, "train_scaled.csv"))
        val_scaled.to_csv(os.path.join(self.nsw_path, "val_scaled.csv"))
        test_scaled.to_csv(os.path.join(self.nsw_path, "test_scaled.csv"))

        # Display confirmation
        print("Saved files to:", self.nsw_path)
        print(os.listdir(self.nsw_path))

        




if __name__ == "__main__":
    data_loader = NSWDataLoader()
    train, validation, test, train_scaled, val_scaled, test_scaled = data_loader.load_data()