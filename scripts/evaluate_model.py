"""STEP 1: test the ML model alone on a labelled CSV (default: data/processed/final.csv).

    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --csv path/to/other.csv --save report.json

It loads the saved model (models/best_model.joblib), scores every row, and compares with the `condition`
column:  normal = 0 (no alarm expected),  overheated / vibration_issue = 1 (alarm expected).

Read the output top to bottom:
  1. overall   : precision / recall / false-alarm rate / ROC-AUC at the model's own threshold
  2. by condition and by recording day  : where the alarms come from
  3. overheated by temperature rise     : the "overheated" label also covers rows that are barely warmer than normal
  4. single-feature check               : how much ONE column alone already separates normal from faults
                                          (if a column like bus_voltage_v scores high, the model may be cheating)
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data"))          # the saved model needs `pipelines.py` to be importable
from pipelines import predict_raw                        # noqa: E402


def pct(x) -> str:
    return f"{100 * x:5.1f}%"

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the saved anomaly model on a labelled CSV")
    ap.add_argument("--csv", default=str(ROOT / "data" / "processed" / "final.csv"))
    ap.add_argument("--bundle", default=str(ROOT / "models" / "best_model.joblib"))
    ap.add_argument("--save", help="write the numbers to this .json file")
    a = ap.parse_args()

    bundle = joblib.load(a.bundle)
    df = pd.read_csv(a.csv)
    scores, flags = predict_raw(bundle, df)
    df["score"], df["flag"] = scores, flags
    df["fault"] = (df["condition"] != "normal").astype(int)          # ground truth from the label
    df["day"] = pd.to_datetime(df["host_received_timestamp_utc"], utc=True, format="ISO8601").dt.date.astype(str)

    print(f"model      : {bundle['display_name']}   threshold = {bundle['threshold']:.3f}   (alarm when score > threshold)")
    print(f"data       : {a.csv}   {len(df)} rows   {df['condition'].value_counts().to_dict()}")

    # 1. overall
    tp = int(((df.fault == 1) & (df.flag == 1)).sum()); fn = int(((df.fault == 1) & (df.flag == 0)).sum())
    fp = int(((df.fault == 0) & (df.flag == 1)).sum()); tn = int(((df.fault == 0) & (df.flag == 0)).sum())
    overall = {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
               "false_alarm_rate": fp / max(1, fp + tn), "roc_auc": float(roc_auc_score(df.fault, df.score))}
    print("\n1. OVERALL")
    print(f"   caught faults TP={tp}  missed FN={fn}  false alarms FP={fp}  correct normal TN={tn}")
    print(f"   precision {pct(overall['precision'])}   recall {pct(overall['recall'])}   "
          f"false-alarm rate {pct(overall['false_alarm_rate'])}   ROC-AUC {overall['roc_auc']:.4f}")

    # 2. by condition / day
    print("\n2. ALARM RATE")
    print("   by condition:", {c: pct(v).strip() for c, v in df.groupby("condition").flag.mean().items()})
    by_day = df.groupby(["day", "condition"]).agg(rows=("flag", "size"), alarm_rate=("flag", "mean"),
                                                  bus_voltage=("bus_voltage_v", "mean"), temp=("temperature_c", "mean"))
    print("   by recording day (a good model has ~0% on normal and ~100% on faults, on EVERY day):")
    print("   " + by_day.round(2).to_string().replace("\n", "\n   "))

    # 3. overheated by temperature rise over the normal median
    ok_temp = df[(df.condition == "normal") & (df.get("temperature_valid", 1) == 1)]
    base = float(ok_temp.temperature_c.median())
    hot = df[df.condition == "overheated"].copy()
    hot["rise_C"] = hot.temperature_c - base
    buckets = hot.groupby(pd.cut(hot.rise_C, [-10, 1, 2, 5, 10, 100])).flag.agg(rows="size", alarm_rate="mean")
    print(f"\n3. 'OVERHEATED' ROWS BY TEMPERATURE RISE over the normal median ({base:.2f} C)")
    print("   " + buckets.round(2).to_string().replace("\n", "\n   "))
    print("   rows with a rise of 2 C or less are barely different from normal: their label is the recording stage, not the heat.")

    # 4. single-feature separation
    feats = [f for f in bundle["features"] if f in df.columns and df[f].nunique() > 1]
    sep = {f: max(roc_auc_score(df.fault, df[f].fillna(df[f].median())), 1 - roc_auc_score(df.fault, df[f].fillna(df[f].median())))
           for f in feats}
    top = sorted(sep.items(), key=lambda kv: -kv[1])[:5]
    print("\n4. SINGLE-FEATURE CHECK (AUC of one column alone; 0.5 = useless, 1.0 = perfect)")
    print("   " + ", ".join(f"{k} {v:.2f}" for k, v in top))

    print("\nLimits: this file comes from the same recordings the model was trained on (a speed hold-out), so it shows how "
          "the model handles unseen speeds, not unseen days. Record new runs for the final test.")

    if a.save:
        report = {"overall": overall, "alarm_rate_by_condition": df.groupby("condition").flag.mean().to_dict(),
                  "overheated_by_rise": {str(k): float(v) for k, v in buckets["alarm_rate"].items()},
                  "single_feature_auc": sep}
        Path(a.save).write_text(json.dumps(report, indent=2, default=str))
        print(f"saved: {a.save}")


if __name__ == "__main__":
    main()
