import pandas as pd
from pathlib import Path

from data_quality_initLabeling import load_excel_dataset
from data_quality_initLabeling import generate_quality_report, initialize_labels
from synthetic_generator import generate_synthetic_data

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA = BASE_DIR / "data" / "raw" / "open_DC_motor-processing data.xlsx"
REAL_OUTPUT = BASE_DIR / "data" / "processed" / "real_sensor_data.csv"
SYNTHETIC_OUTPUT = BASE_DIR / "data" / "processed" / "synthetic_sensor_data.csv"
COMBINED_OUTPUT = BASE_DIR / "data" / "processed" / "combined_sensor_data.csv"

def merge_real_and_synthetic(real_data, synthetic_data):
    real_data["label"] = "unknown"
    real_data["data_type"] = "real"

    synthetic_data["label"] = "unknown"
    synthetic_data["data_type"] = "synthetic"

    label_columns = ["label", "data_type"]

    measurement_columns = [
        c for c in real_data.columns
        if c not in label_columns
    ]

    synthetic_data = synthetic_data.reindex(
        columns=measurement_columns + label_columns
    )

    real_data = real_data[measurement_columns + label_columns]

    combined = pd.concat(
        [real_data, synthetic_data],
        ignore_index=True
    )

    return combined

def main():
    print("Loading real F.A.I.R data...")

    real_data = load_excel_dataset(RAW_DATA)
    print("Real dataset shape:", real_data.shape)

    report = generate_quality_report(real_data)

    print("\nData Quality Report")
    for key, value in report.items():
        print(f"{key}: {value}")

    real_data = initialize_labels(real_data)
    real_data["data_type"] = "real"

    print("\nGenerating synthetic data...")

    synthetic_data = generate_synthetic_data(
        real_data,
        n_samples=3000
    )

    print("Synthetic dataset shape:", synthetic_data.shape)

    synthetic_data = initialize_labels(synthetic_data)
    synthetic_data["data_type"] = "synthetic"

    REAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    real_data.to_csv(REAL_OUTPUT, index=False)

    print("\nReal data saved:", REAL_OUTPUT)

    SYNTHETIC_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    synthetic_data.to_csv(SYNTHETIC_OUTPUT, index=False)

    print("Synthetic data saved:", SYNTHETIC_OUTPUT)

    print("\nMerging real and synthetic data...")

    combined_data = merge_real_and_synthetic(
        real_data,
        synthetic_data
    )

    COMBINED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined_data.to_csv(COMBINED_OUTPUT, index=False)

    print("Combined data saved:", COMBINED_OUTPUT)

    print("\nPipeline completed successfully.")
    print("\nCombined dataset shape:")
    print(combined_data.shape)

    print("\nData type distribution:")
    print(combined_data["data_type"].value_counts())

    print("\nLabel distribution:")
    print(combined_data["label"].value_counts())

if __name__ == "__main__":
    main()