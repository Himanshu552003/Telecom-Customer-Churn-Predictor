"""
Preprocessing and feature engineering for the telecom churn dataset.

Exposes `load_and_engineer(path)` which returns a cleaned DataFrame with
engineered features, and `build_preprocessor(df)` which returns a
scikit-learn ColumnTransformer ready to be used inside a Pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ADDON_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def tenure_bucket(tenure):
    if tenure <= 12:
        return "0-1yr"
    elif tenure <= 24:
        return "1-2yr"
    elif tenure <= 48:
        return "2-4yr"
    else:
        return "4yr+"


def engineer_features(df):
    """Apply cleaning + feature engineering to a raw customer DataFrame.

    Expects the original telecom schema columns (minus customerID/Churn are
    optional). Safe to call on a single-row DataFrame built from a JSON
    payload, e.g. for real-time inference in the Flask app.
    """
    df = df.copy()

    # --- Clean TotalCharges: blank strings -> NaN -> impute ---
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # --- Feature engineering ---
    # 1. Tenure buckets (captures non-linear lifecycle risk)
    df["TenureBucket"] = df["tenure"].apply(tenure_bucket)

    # 2. Average monthly spend over the customer's lifetime
    df["AvgMonthlySpend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    # 3. Number of add-on services subscribed
    df["NumAddonServices"] = (df[ADDON_COLUMNS] == "Yes").sum(axis=1)

    # 4. Has any streaming service (engagement proxy)
    df["HasStreaming"] = (
        (df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")
    ).astype(int)

    return df


def load_and_engineer(path="data/telecom_churn.csv"):
    df = pd.read_csv(path)

    df = engineer_features(df)

    # --- Target ---
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    df = df.drop(columns=["customerID"])

    return df


def get_feature_lists(df):
    target = "Churn"
    categorical_cols = [
        c for c in df.select_dtypes(include=["object"]).columns if c != target
    ]
    numeric_cols = [
        c for c in df.select_dtypes(include=["int64", "float64"]).columns
        if c not in (target, "SeniorCitizen")
    ]
    # SeniorCitizen is binary 0/1, treat as numeric passthrough too
    numeric_cols = list(dict.fromkeys(numeric_cols + ["SeniorCitizen"]))
    return categorical_cols, numeric_cols


def build_preprocessor(df):
    categorical_cols, numeric_cols = get_feature_lists(df)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_cols),
        ]
    )
    return preprocessor, categorical_cols, numeric_cols


if __name__ == "__main__":
    df = load_and_engineer()
    print(df.shape)
    print(df.dtypes)
    print(df.isna().sum().sum(), "missing values remaining")
    cat_cols, num_cols = get_feature_lists(df)
    print("Categorical:", cat_cols)
    print("Numeric:", num_cols)
