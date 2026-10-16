import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    clean_sensor_data,
    split_features_labels_metadata,
)
from src.data.features import create_features


def test_clean_sensor_data_replaces_infinity():
    df = pd.DataFrame({"temperature": [20.0, np.inf, -np.inf, 30.0]})

    result = clean_sensor_data(df)

    assert len(result) == 4
    assert result["temperature"].isna().sum() == 2


def test_split_features_labels_metadata():
    df = pd.DataFrame({
        "File": ["a.csv", "b.csv"],
        "temperature": [20.0, 25.0],
        "label": ["normal", "anomaly"],
    })

    X, y, metadata = split_features_labels_metadata(df)

    assert list(X.columns) == ["temperature"]
    assert y.tolist() == ["normal", "anomaly"]
    assert list(metadata.columns) == ["File"]


def test_split_requires_label():
    df = pd.DataFrame({"temperature": [20.0, 25.0]})

    with pytest.raises(ValueError, match="label"):
        split_features_labels_metadata(df)


def test_create_features_calculates_temperature_rise():
    df = pd.DataFrame({
        "temp_mot": [40.0, 50.0],
        "temp_amb": [20.0, 25.0],
    })

    result = create_features(df)

    assert result["temp_rise"].tolist() == [20.0, 25.0]
    assert "temp_mot" in result.columns
