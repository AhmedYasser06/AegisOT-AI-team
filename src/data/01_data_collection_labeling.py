from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "data" / "raw" / "raw_dataFF.csv"
# FINAL_FILE = ROOT / "data" / "raw" / "final.csv"   # optional: labeled with the exact same rules

OUTPUT_DIR = ROOT / "data" / "processed"

REQUIRED_COLUMNS = [
    "host_received_timestamp_utc", "condition", "motor_running",
    "speed_percent", "current_ma", "temperature_c",
    "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2", "rpm"
]
VALID_CONDITIONS = {"normal", "overheated", "vibration_issue"}

FEATURES = [
    "motor_running", "speed_percent", "current_ma", "temperature_c",
    "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",'rpm'
]

# the overheated runs start cold (or end cooling down), those rows do not show the fault.
# the rule is a fixed temperature: normal and vibration_issue never go above ~28.8 C in the
# raw data, so an overheated row is only labeled as a fault when it is at or above this value.
# it does not depend on the other rows, so the same rule is safe for raw and final.
REFINE_LABELS = True
HOT_TEMP_C = 30.0


def label_data(data: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data.copy()
    data["host_received_timestamp_utc"] = pd.to_datetime(
        data["host_received_timestamp_utc"], errors="coerce", utc=True
    )
    data["condition"] = data["condition"].astype("string").str.strip().str.lower()
    unknown = sorted(set(data["condition"].dropna().unique()) - VALID_CONDITIONS)
    if unknown:
        raise ValueError(f"Unexpected condition values: {unknown}")

    data["label"] = data["condition"].map(
        {"normal": 0, "overheated": 1, "vibration_issue": 1}
    ).astype("Int64")
    data["fault_type"] = data["condition"].map(
        {"normal": "none", "overheated": "overheating", "vibration_issue": "vibration"}
    )

    data = data.sort_values("host_received_timestamp_utc", kind="stable").reset_index(drop=True)

    data["label_note"] = ""
    if REFINE_LABELS:
        temp = pd.to_numeric(data["temperature_c"], errors="coerce")
        # missing temperature on an overheated row: cannot tell, so it is left out too
        not_hot = (data["condition"] == "overheated") & ~(temp >= HOT_TEMP_C)
        idle_vibration = (data["condition"] == "vibration_issue") & (data["motor_running"] == 0)
        data.loc[not_hot, "label_note"] = "not_hot_yet"
        data.loc[idle_vibration, "label_note"] = "idle_motor"
        data.loc[not_hot | idle_vibration, "label"] = pd.NA
    return data

DROP_COLUMNS = ["rpm_valid", "encoder_counts_per_revolution"]
def make_clean(data: pd.DataFrame) -> pd.DataFrame:
    # rows used for training / evaluation: has a label and every feature present
    clean = data.dropna(subset=["label"] + FEATURES).copy()
    clean["label"] = clean["label"].astype(int)
    clean = clean.drop(columns=DROP_COLUMNS, errors="ignore")
    return clean


def report(name, data, clean):
    print(f"\n===== {name} =====")
    print(f"Rows: {len(data)}")
    print("Recorded condition counts:")
    print(data["condition"].value_counts(dropna=False).to_string())
    print("Label counts (NA = left out):")
    print(data["label"].value_counts(dropna=False).to_string())
    print("Left-out reasons:")
    print(data.loc[data["label"].isna(), "label_note"].value_counts().to_string())
    print("Missing values (all columns):")
    print(data.isna().sum().loc[lambda v: v > 0].to_string())
    print(f"Clean rows (label + features present): {len(clean)}")
    print(clean.groupby(["condition", "label"]).size().to_string())


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Place the original dataset at: {RAW_FILE}")

    for name, path, stem in [("RAW", RAW_FILE, "labeled_sensor_data"),]:
        if not path.exists():
            continue
        data = label_data(pd.read_csv(path))
        clean = make_clean(data)
        data.to_csv(OUTPUT_DIR / f"{stem}.csv", index=False)
        clean.to_csv(OUTPUT_DIR / f"{stem}_clean.csv", index=False)
        report(name, data, clean)
    print(f"\nSaved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()