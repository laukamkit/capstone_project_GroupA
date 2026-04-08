
import pandas as pd

def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "TOTALDEMAND",
    feature_cols: list[str] = ["TEMPERATURE"],
    demand_lags: list[int] | None = None,
    feature_lags: list[int] | None = None,
):
    df = df.copy()
    demand_lags = demand_lags or []
    feature_lags = feature_lags or []

    for lag in demand_lags:
        df[f"demand_lag_{lag}"] = df[target_col].shift(lag)

    for feature_col in feature_cols:
        if feature_col in df.columns:
            for lag in feature_lags:
                df[f"{feature_col}_lag_{lag}"] = df[feature_col].shift(lag)

    df = df.dropna().reset_index()
    return df