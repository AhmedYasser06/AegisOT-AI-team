import pandas as pd
import numpy as np

def load_excel_dataset(file_path):
    df = pd.read_excel(file_path)
    return df

def load_csv_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def check_missing_values(df):
    return pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percentage":
            df.isnull().mean() * 100
    })

def check_duplicates(df):
    return df.duplicated().sum()

def check_required_columns( df, required_columns):
    return {
        column:
        column in df.columns
        for column in required_columns
    }

def generate_quality_report(df):
    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicates": df.duplicated().sum(),
        "missing_values":df.isnull().sum().sum()
    }

    return report

def initialize_labels(df):
    data = df.copy()
    data["label"] = "unknown"
    # data["fault_type"] = "none"
    return data
