from pathlib import Path
import json
import joblib
import mlflow
import numpy as np

from mlflow import MlflowClient
from sklearn.metrics import classification_report
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "src" / "data"

sys.path.insert(0, str(DATA_DIR))

from pipelines import (
    MODEL_KEYS,
    DISPLAY_NAMES,
    score_rows,
    score_fit_rows,
)


# =========================================================
# 1. PATHS AND CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "data" / "results"
REPORTS_DIR = RESULTS_DIR / "classification_reports"

TRACKING_DB = ROOT / "mlflow_tracking_clean.db"
TRACKING_URI = f"sqlite:///{TRACKING_DB.as_posix()}"

SPLITS = ("train", "val", "test")

CLASS_LABELS = [0, 1]
CLASS_NAMES = ["Normal", "Anomaly"]


# =========================================================
# 2. MLFLOW SETUP
# =========================================================

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)


# =========================================================
# 3. FIND LATEST COMMON EXECUTION
# =========================================================

def find_latest_execution():

    execution_ids = []

    for key in MODEL_KEYS:

        experiment = client.get_experiment_by_name(
            f"AegisOT_{key}"
        )

        if experiment is None:
            raise RuntimeError(
                f"Experiment not found: AegisOT_{key}"
            )

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.stage = 'sweep'",
            order_by=["attributes.start_time DESC"],
            max_results=100,
        )

        ids = {
            run.data.tags.get("execution_id")
            for run in runs
            if run.data.tags.get("execution_id")
        }

        execution_ids.append(ids)

    common_ids = set.intersection(*execution_ids)

    if not common_ids:
        raise RuntimeError(
            "No common execution_id found across model families."
        )

    # execution IDs use YYYYMMDD_HHMMSS format
    latest = max(common_ids)

    print(f"\nSelected execution: {latest}")

    return latest


# =========================================================
# 4. GET BEST RUN FOR EACH MODEL
# =========================================================

def get_best_run(key, execution_id):

    experiment = client.get_experiment_by_name(
        f"AegisOT_{key}"
    )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "tags.stage = 'sweep' "
            f"and tags.execution_id = '{execution_id}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=10,
    )

    if not runs:
        raise RuntimeError(
            f"No sweep run found for {key}, execution {execution_id}"
        )

    best_run_id = runs[0].data.tags.get("best_run_id")

    if not best_run_id:
        raise RuntimeError(
            f"best_run_id missing in sweep run for {key}"
        )

    return best_run_id


# =========================================================
# 5. LOAD MODEL BUNDLE
# =========================================================

def load_best_bundle(key, execution_id):

    best_run_id = get_best_run(key, execution_id)

    print(f"\nLoading {key}")
    print(f"Best run ID: {best_run_id}")

    bundle_dir = client.download_artifacts(
        best_run_id,
        "model_bundle"
    )

    bundle_files = list(Path(bundle_dir).glob("*.joblib"))

    if len(bundle_files) != 1:
        raise RuntimeError(
            f"Expected one joblib bundle for {key}, "
            f"found {len(bundle_files)}"
        )

    bundle = joblib.load(bundle_files[0])

    required = {"key", "model", "threshold"}

    if not required.issubset(bundle):
        raise RuntimeError(
            f"Invalid bundle for {key}. Missing: "
            f"{required - set(bundle)}"
        )

    if bundle["key"] != key:
        raise RuntimeError(
            f"Bundle mismatch: expected {key}, "
            f"got {bundle['key']}"
        )

    print(f"Threshold: {bundle['threshold']}")

    return bundle, best_run_id


# =========================================================
# 6. LOAD DATA
# =========================================================

def load_data(key):

    path = PROCESSED_DIR / f"model_data_{key}.npz"

    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)

    result = {}

    for split in SPLITS:

        result[split] = {
            "X": data[f"X_{split}"],
            "y": data[f"y_{split}"].astype(int),
        }

    return result


# =========================================================
# 7. GET SCORES
# =========================================================

def get_split_scores(key, model, data):

    scores = {}

    # Training scores: preserve the original fit-row behavior.
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    fit_mask = y_train == 0

    train_scores = np.empty(len(y_train), dtype=float)

    train_scores[fit_mask] = score_fit_rows(
        key,
        model,
        X_train[fit_mask],
    )

    train_scores[~fit_mask] = score_rows(
        key,
        model,
        X_train[~fit_mask],
    )

    scores["train"] = train_scores

    # Validation and test scores
    for split in ("val", "test"):

        scores[split] = score_rows(
            key,
            model,
            data[split]["X"],
        )

    return scores


# =========================================================
# 8. GENERATE AND SAVE REPORTS
# =========================================================

def generate_report(key, split, y_true, scores, threshold,
                    run_id):

    y_pred = (scores > threshold).astype(int)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    model_dir = REPORTS_DIR / key
    model_dir.mkdir(parents=True, exist_ok=True)

    # TXT
    txt_path = model_dir / f"{split}_report.txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {key}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"MLflow Run ID: {run_id}\n\n")
        f.write(report_text)

    # JSON
    json_path = model_dir / f"{split}_report.json"

    output = {
        "model_key": key,
        "model_name": DISPLAY_NAMES.get(key, key),
        "split": split,
        "threshold": float(threshold),
        "run_id": run_id,
        "prediction_rule": "score > threshold",
        "classification_report": report_dict,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"\n{key} | {split.upper()}")
    print(report_text)


# =========================================================
# 9. MAIN
# =========================================================

def main():

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    execution_id = find_latest_execution()

    for key in MODEL_KEYS:

        print("\n" + "=" * 65)
        print(f"MODEL: {key}")
        print("=" * 65)

        bundle, run_id = load_best_bundle(
            key,
            execution_id,
        )

        model = bundle["model"]
        threshold = float(bundle["threshold"])

        data = load_data(key)

        scores = get_split_scores(
            key,
            model,
            data,
        )

        for split in SPLITS:

            generate_report(
                key=key,
                split=split,
                y_true=data[split]["y"],
                scores=scores[split],
                threshold=threshold,
                run_id=run_id,
            )

    print("\n" + "=" * 65)
    print("ALL CLASSIFICATION REPORTS GENERATED SUCCESSFULLY")
    print(f"Saved in: {REPORTS_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
