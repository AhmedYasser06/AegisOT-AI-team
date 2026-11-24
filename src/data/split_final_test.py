import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA = r"F:\college pdfs\aegisot\data\raw\raw_dataF.csv"
FINAL_DATA = r"F:\college pdfs\aegisot\data\raw\final.csv"
OUTPUT_DIR_Raw = Path("data/raw")
OUTPUT_DIR_Final = Path("api")

# Speed ranges that the model must NEVER see during development.
# These ranges will be reserved for final evaluation.

HOLDOUT_SPEED_RANGES = [
    (18, 22),   # Around 20% speed
    (48, 52),   # Around 50% speed
    (78, 80)    # Around 80% speed
]
BUFFER_ROWS = 5

raw_df = pd.read_csv(RAW_DATA)
final_df = pd.read_csv(FINAL_DATA)

original_columns = list(raw_df.columns)

if set(raw_df.columns) != set(final_df.columns):
    raise ValueError("Raw and Final datasets have different columns.")

raw_df["_source"] = "raw"
final_df["_source"] = "final"

df = pd.concat([raw_df, final_df], ignore_index=True)

df["_ts"] = pd.to_datetime(
    df["host_received_timestamp_utc"],
    errors="coerce",
    utc=True
)

if df["_ts"].isna().any():
    raise ValueError("Some timestamps could not be parsed.")

df = df.sort_values(
    ["_source", "_ts"],
    kind="stable"
).reset_index(drop=True)

in_final = pd.Series(False, index=df.index)

for low, high in HOLDOUT_SPEED_RANGES:

    in_final |= df["speed_percent"].between(low, high)

near_final = np.zeros(len(df), dtype=bool)

for source, group in df.groupby("_source", sort=False):

    positions = np.flatnonzero(
        in_final.loc[group.index].to_numpy()
    )

    if len(positions) == 0:
        continue

    # Find continuous held-out segments.
    segments = np.split(
        positions,
        np.where(np.diff(positions) > 1)[0] + 1
    )

    for segment in segments:

        start = segment[0]
        end = segment[-1]

        # Expand each held-out segment by BUFFER_ROWS.
        lo = max(0, start - BUFFER_ROWS)
        hi = min(len(group), end + BUFFER_ROWS + 1)

        near_final[group.index[lo:hi]] = True

final_df = df[in_final].copy()

raw_df = df[~near_final].copy()

buffer_dropped = int(
    ((~in_final) & near_final).sum()
)

# Validation checks
# No shared records.
assert set(raw_df.index).isdisjoint(final_df.index)

# No hidden speed values in Raw.
for low, high in HOLDOUT_SPEED_RANGES:

    assert not raw_df["speed_percent"].between(low, high).any(), (
        f"Raw contains hidden speed range {low}-{high}%"
    )

# Check expected conditions.
expected_conditions = {
    "normal",
    "overheated",
    "vibration_issue"
}

if not expected_conditions.issubset(set(df["condition"].unique())):
    raise ValueError("Some expected conditions are missing.")

OUTPUT_DIR_Raw.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_Final.mkdir(parents=True, exist_ok=True)

raw_df[original_columns].to_csv(
    OUTPUT_DIR_Raw / "raw_dataFF.csv",
    index=False
)

final_df[original_columns].to_csv(
    OUTPUT_DIR_Final / "final.csv",
    index=False
)


print("Dataset separation completed.")

print(f"\nTotal rows: {len(df)}")
print(f"Raw rows: {len(raw_df)}")
print(f"Final rows: {len(final_df)}")
print(f"Buffer rows removed: {buffer_dropped}")

print("\nHidden speed ranges:")

for low, high in HOLDOUT_SPEED_RANGES:
    print(f"{low}% - {high}%")

print("\nFinal speed distribution:")
print(final_df["speed_percent"].describe())

print("\nFinal condition distribution:")
print(final_df["condition"].value_counts())

print("\nRaw condition distribution:")
print(raw_df["condition"].value_counts())

print("\nFinal speed values:")
print(sorted(final_df["speed_percent"].dropna().unique()))

print("\nRaw speed values:")
print(sorted(raw_df["speed_percent"].dropna().unique()))

print("\nOriginal columns preserved:",
      original_columns == list(raw_df[original_columns].columns)
      == list(final_df[original_columns].columns))