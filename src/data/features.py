import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered sensor features while preserving existing columns."""
    result = df.copy()

    if "temp_mot" in result.columns and "temp_amb" in result.columns:
        result["temp_rise"] = result["temp_mot"] - result["temp_amb"]

    return result
