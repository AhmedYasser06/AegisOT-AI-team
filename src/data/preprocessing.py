import numpy as np
import pandas as pd


def clean_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean sensor data types and invalid numeric values."""
    result = df.copy()

    # Convert infinite values to missing values
    result = result.replace([np.inf, -np.inf], np.nan)

    # Convert numeric-looking object columns where possible
    for column in result.columns:
        if result[column].dtype == "object":
            converted = pd.to_numeric(result[column], errors="coerce")

            if converted.notna().sum() > 0:
                result[column] = converted

    return result


def split_features_labels_metadata(df: pd.DataFrame):
    """Separate features, labels, and available metadata."""
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    metadata_cols = [
        col for col in ["File", "Date", "data_type"]
        if col in df.columns
    ]

    feature_cols = [
        col for col in df.columns
        if col not in metadata_cols + ["label"]
    ]

    X = df[feature_cols].copy()
    y = df["label"].copy()
    metadata = df[metadata_cols].copy()

    # Drop columns that contain no usable values
    X = X.dropna(axis=1, how="all")

    return X, y, metadata
