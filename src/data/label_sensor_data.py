import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
COMBINED_PATH = BASE_DIR / "data" / "processed" / "combined_sensor_data.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "combined_sensor_data_labeled.csv"


# Load dataset
df = pd.read_csv(COMBINED_PATH)
required_columns = [
    "Voltage",
    "Current",
    "speed_rpm",
    "temp_mot",
    "0-4k_Hz[Vibration]",
    "4k-8k_Hz[Vibration]",
    "8k-16kHz[Vibration]",
    "16k-26kHz[Vibration]",
    "label",
    "data_type",
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")


# Separate real data
real_data = df[df["data_type"].astype(str).str.lower() == "real"].copy()
if real_data.empty: raise ValueError("No real data found.")


# Operating levels
# The real dataset contains two main operating levels.
# The very-low-voltage samples are treated as signal-loss cases.
LOW_VOLTAGE_THRESHOLD = 4.0
SIGNAL_LOSS_VOLTAGE = 1.0
SIGNAL_LOSS_CURRENT = 0.2


def get_operating_level(voltage):
    if pd.isna(voltage):
        return "unknown"

    if voltage < SIGNAL_LOSS_VOLTAGE:
        return "signal_loss"

    if voltage < LOW_VOLTAGE_THRESHOLD:
        return "low"

    return "high"


# Temporary operating level for real data
real_data["operating_level"] = real_data["Voltage"].apply( get_operating_level)

# Keep only valid operating levels for baseline calculation
valid_real = real_data[
    real_data["operating_level"].isin(["low", "high"])
].copy()


# Calculate robust center and spread
def robust_stats(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    median = series.median()
    mad = np.median(np.abs(series - median))   # Median Absolute Deviation
    robust_std = 1.4826 * mad
    return median, robust_std                  # expected value + natural variation

# Expected values for each operating level
operating_levels = {}
for level in ["low", "high"]:
    level_data = valid_real[
        valid_real["operating_level"] == level
    ]

    current_median, current_spread = robust_stats(
        level_data["Current"]
    )

    speed_median, speed_spread = robust_stats(
        level_data["speed_rpm"]
    )

    operating_levels[level] = {
        "current_expected": current_median,
        "current_spread": current_spread,
        "speed_expected": speed_median,
        "speed_spread": speed_spread,
    }


# Print detected operating levels
print("\nOperating levels:")
for level, values in operating_levels.items():
    print(f"\n{level.upper()}")

    print(
        f"Expected Current: "
        f"{values['current_expected']:.3f} A"
    )

    print(
        f"Current robust spread: "
        f"{values['current_spread']:.3f}"
    )

    print(
        f"Expected Speed: "
        f"{values['speed_expected']:.1f} RPM"
    )

    print(
        f"Speed robust spread: "
        f"{values['speed_spread']:.1f}"
    )


# Temperature threshold
# Use the highest observed real temperature as the baseline.
# A margin is added to avoid labeling small natural variations as abnormal.
real_temperature = pd.to_numeric(
    valid_real["temp_mot"],
    errors="coerce"
).dropna()

max_real_temperature = real_temperature.max()
TEMPERATURE_MARGIN = 5.0
TEMPERATURE_ABNORMAL = (max_real_temperature + TEMPERATURE_MARGIN)
TEMPERATURE_EXTREME = (TEMPERATURE_ABNORMAL + 5.0)    # we don't have datasheet info, so we add a margin for extreme cases


print("\nTemperature thresholds:")
print(
    f"Abnormal: > {TEMPERATURE_ABNORMAL:.2f} °C"
)
print(
    f"Extreme: > {TEMPERATURE_EXTREME:.2f} °C"
)


# Vibration features
vibration_columns = [
    "0-4k_Hz[Vibration]",
    "4k-8k_Hz[Vibration]",
    "8k-16kHz[Vibration]",
    "16k-26kHz[Vibration]",
]


# Build vibration baselines per operating level
vibration_baselines = {}

for level in ["low", "high"]:
    level_data = valid_real[
        valid_real["operating_level"] == level
    ]
    vibration_baselines[level] = {}
    for column in vibration_columns:

        median, robust_spread = robust_stats(
            np.log10(level_data[column].clip(lower=1e-6))
        )

        # Abnormal threshold
        abnormal_threshold = (
            median + 4.0 * robust_spread
        )

        # Extreme threshold
        extreme_threshold = (
            median + 6.0 * robust_spread
        )

        vibration_baselines[level][column] = {
            "median": median,
            "spread": robust_spread,
            "abnormal": abnormal_threshold,
            "extreme": extreme_threshold,
        }


# Domain classification
def classify_row(row):
    voltage = pd.to_numeric(
        row["Voltage"],
        errors="coerce"
    )

    current = pd.to_numeric(
        row["Current"],
        errors="coerce"
    )

    speed = pd.to_numeric(
        row["speed_rpm"],
        errors="coerce"
    )

    temperature = pd.to_numeric(
        row["temp_mot"],
        errors="coerce"
    )


    abnormal_domains = 0           # Electrical(current, voltage), Motion(speed), Temperature, Vibration
    extreme_domain = False


    # Electrical domain
    electrical_abnormal = False
    electrical_extreme = False

    # Missing electrical signal
    if (
        pd.isna(voltage)
        or pd.isna(current)
    ):
        electrical_abnormal = True

    # Signal loss
    elif (
        voltage < SIGNAL_LOSS_VOLTAGE
        and current < SIGNAL_LOSS_CURRENT
    ):
        electrical_abnormal = True
        electrical_extreme = True

    else:

        level = get_operating_level(voltage)

        if level in operating_levels:

            expected_current = operating_levels[level]["current_expected"]
            current_spread = operating_levels[level]["current_spread"]

            # Minimum tolerance prevents extremely small
            # MAD values from creating an unrealistically
            # strict threshold.
            tolerance = max(
                3.0 * current_spread,
                0.20 * expected_current
            )

            extreme_tolerance = max(
                5.0 * current_spread,
                0.35 * expected_current
            )

            current_deviation = abs(
                current - expected_current
            )

            if current_deviation > tolerance:
                electrical_abnormal = True

            if current_deviation > extreme_tolerance:
                electrical_extreme = True


    if electrical_abnormal:
        abnormal_domains += 1

    if electrical_extreme:
        extreme_domain = True


    # Motion domain
    motion_abnormal = False
    motion_extreme = False

    level = get_operating_level(voltage)

    if (
        pd.isna(speed)
        or level == "signal_loss"
        or level == "unknown"
    ):
        motion_abnormal = True

        if level == "signal_loss":
            motion_extreme = True

    else:

        expected_speed = operating_levels[level]["speed_expected"]
        speed_spread = operating_levels[level]["speed_spread"]

        tolerance = max(
            3.0 * speed_spread,
            0.10 * expected_speed
        )

        extreme_tolerance = max(
            5.0 * speed_spread,
            0.20 * expected_speed
        )

        speed_deviation = abs(
            speed - expected_speed
        )

        if speed_deviation > tolerance:
            motion_abnormal = True

        if speed_deviation > extreme_tolerance:
            motion_extreme = True


    if motion_abnormal:
        abnormal_domains += 1

    if motion_extreme:
        extreme_domain = True


    # Temperature domain
    temperature_abnormal = False
    temperature_extreme = False

    if pd.isna(temperature):
        temperature_abnormal = True

    elif temperature > TEMPERATURE_ABNORMAL:
        temperature_abnormal = True

    if (
        not pd.isna(temperature)
        and temperature > TEMPERATURE_EXTREME
    ):
        temperature_extreme = True


    if temperature_abnormal:
        abnormal_domains += 1

    if temperature_extreme:
        extreme_domain = True


    # Vibration domain
    vibration_abnormal_count = 0
    vibration_extreme_count = 0

    if level in vibration_baselines:

        for column in vibration_columns:

            value = pd.to_numeric(
                row[column],
                errors="coerce"
            )

            if pd.isna(value):
                vibration_abnormal_count += 1
                continue

            value = np.log10(max(value, 1e-6))
            baseline = vibration_baselines[level ][column]
            
            if value > baseline["abnormal"]:
                vibration_abnormal_count += 1
            if value > baseline["extreme"]:
                vibration_extreme_count += 1


    # Require at least two vibration bands
    # to classify the vibration domain as abnormal.
    vibration_abnormal = (
        vibration_abnormal_count >= 2
    )

    vibration_extreme = (
        vibration_extreme_count >= 2
    )

    if vibration_abnormal:
        abnormal_domains += 1

    if vibration_extreme:
        extreme_domain = True


    # Final decision
    if extreme_domain or abnormal_domains >= 2:
        return "anomaly"

    return "normal"


# Apply labeling
df["label"] = df.apply(classify_row,axis=1)

# Save output
OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)
df.to_csv( OUTPUT_PATH, index=False)

# Summary
print("\nLabeling completed.")

print("\nLabel distribution:")
print(
    df["label"].value_counts()
)

print("\nData type + label:")
print(
    pd.crosstab(df["data_type"],df["label"]))

print(f"\nSaved to:\n{OUTPUT_PATH}")

'''
Data type + label:
label      anomaly  normal
data_type                 
real             8     369
synthetic       59    2941

'''