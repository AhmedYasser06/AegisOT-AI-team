from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

import sklearn

from pipelines import MODEL_KEYS, PREPROCESSING_TEXT, add_features, build_preprocessor

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "labeled_sensor_data_clean.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

# rows are cut into blocks of 50 (10 s) in time order, inside each recorded condition, and whole
# blocks go to a set. a plain random split would put neighbours of the same second in train and test
BLOCK_SIZE = 50
SET_PATTERN = ["train", "train", "val", "train", "test"]   # 60 / 20 / 20
RANDOM_STATE = 42
PROBE_ROWS = 20

DROP_CURRENT_FEATURES = False     # set to True if current features are unreliable or unsafe

EXCLUDED_COLUMNS = {
    "host_received_timestamp_utc", "condition", "label", "fault_type",
    "last_motor_stop_reason", "motor_direction", "data_source",
    "device_time_ms", "stage_elapsed_ms", "label_note"
}

BASE_FEATURES = [
    "motor_running", "speed_percent","bus_voltage_v", "shunt_voltage_mv", 
    "current_ma", "power_mw","temperature_c", "accel_x_m_s2", "accel_y_m_s2",
    "accel_z_m_s2","gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
    "rpm",
]
HEALTH_FEATURES = [
    "ina219_ok", "power_data_valid", "ds18b20_ok",
    "temperature_valid", "mpu6050_ok", "mpu_data_valid",
]

def warn_if_not_chronological(frame):
    """The block split assumes rows are in time order inside each condition. We only check and warn,
    we never re-sort, so the split stays identical to the original experiments."""
    if "host_received_timestamp_utc" not in frame.columns:
        print("WARNING: no timestamp column, cannot check the row order.")
        return
    ts = pd.to_datetime(frame["host_received_timestamp_utc"], utc=True, errors="coerce")
    for condition, group in ts.groupby(frame["condition"]):
        if not group.dropna().is_monotonic_increasing:
            print(f"WARNING: rows of condition '{condition}' are not in time order. "
                "Sort the csv by timestamp first, otherwise the block split is not time based.")


def assign_sets(frame):
    data = frame.copy()
    # dropna=False: rows without a condition get a block too (they are dropped later) instead of crashing
    position = data.groupby("condition", dropna=False).cumcount()
    data["block"] = position // BLOCK_SIZE
    data["split"] = data["block"].map(lambda b: SET_PATTERN[b % len(SET_PATTERN)])
    return data


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Run 01_data_collection_labeling.py before this script."
        )

    frame = pd.read_csv(INPUT_FILE)
    warn_if_not_chronological(frame)
    frame = add_features(frame)

    feature_columns = [
        column for column in BASE_FEATURES + HEALTH_FEATURES
        + ["acceleration_magnitude", "gyro_magnitude", "speed_when_running"]
        if column in frame.columns and column not in EXCLUDED_COLUMNS
    ]

    if DROP_CURRENT_FEATURES:
        feature_columns = [
            column for column in feature_columns
            if column not in {"current_ma", "shunt_voltage_mv", "power_mw"}
        ]

    if not feature_columns:
        raise ValueError("No usable feature columns were found.")

    # blocks are counted before unlabeled rows are dropped so the time order stays intact
    frame = assign_sets(frame)
    frame = frame.dropna(subset=["label", "condition"]).reset_index(drop=True)
    frame["label"] = frame["label"].astype(int)

    features = frame[feature_columns].apply(pd.to_numeric, errors="coerce")

    # the models only learn from normal rows of the train set
    fit_mask = ((frame["split"] == "train") & (frame["label"] == 0)).to_numpy()

    # a column that never changes in the normal training rows carries nothing (all the *_ok flags),
    # and in the robust baseline it would turn any later change into a huge score
    varying = features[fit_mask].nunique(dropna=True) > 1
    dropped = [c for c in feature_columns if not varying[c]]
    feature_columns = [c for c in feature_columns if varying[c]]
    features = features[feature_columns]

    sets = {name: (frame["split"] == name).to_numpy() for name in ["train", "val", "test"]}
    x_fit = features[fit_mask].to_numpy()

    pre_notes = {}
    for key in MODEL_KEYS:
        # each model gets its own preprocessing, fitted on the normal training rows only
        preprocessor = build_preprocessor(key).fit(x_fit)
        arrays = {}
        for name, mask in sets.items():
            arrays[f"X_{name}"] = preprocessor.transform(features[mask].to_numpy())
            arrays[f"y_{name}"] = frame.loc[mask, "label"].to_numpy()
            arrays[f"cond_{name}"] = frame.loc[mask, "condition"].astype(str).to_numpy()
        # first raw train rows: 03 re-transforms them with the saved preprocessor and compares with
        # X_train[:N], which proves the .joblib is the same object that produced the arrays
        arrays["raw_probe"] = features[sets["train"]].to_numpy()[:PROBE_ROWS]
        np.savez_compressed(
            OUTPUT_DIR / f"model_data_{key}.npz",
            feature_names=np.array(feature_columns),
            **arrays
        )
        joblib.dump(preprocessor, OUTPUT_DIR / f"preprocessor_{key}.joblib")
        pre_notes[key] = PREPROCESSING_TEXT[key]

    summary = (
        frame.groupby(["split", "condition"]).size().unstack(fill_value=0)
        .reindex(["train", "val", "test"])
    )
    frame[["host_received_timestamp_utc", "condition", "label", "block", "split"]].to_csv(
        OUTPUT_DIR / "split_assignment.csv", index=False
    )

    metadata = {
        "feature_columns": feature_columns,
        "dropped_constant_features": dropped,
        "fit_rows_normal_train": int(fit_mask.sum()),
        "rows_per_set": {k: int(v.sum()) for k, v in sets.items()},
        "block_size": BLOCK_SIZE,
        "set_pattern": SET_PATTERN,
        "preprocessing_per_model": pre_notes,
        "drop_current_features": DROP_CURRENT_FEATURES,
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
        "probe_rows": PROBE_ROWS,
    }
    with open(OUTPUT_DIR / "preprocessing_metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)

    print(f"Features kept ({len(feature_columns)}): {feature_columns}")
    print(f"Dropped because they never change in normal training rows: {dropped}")
    print(f"\nNormal rows used to fit models and preprocessors: {int(fit_mask.sum())}")
    print("\nRows per set and condition (fault rows are never used for fitting):")
    print(summary.to_string())
    print("\nPreprocessing per model:")
    for key, text in pre_notes.items():
        print(f"  {key:18s} {text}")
    print(f"\nSaved arrays and fitted preprocessors in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
