from pathlib import Path
import itertools
import json
import tempfile
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from pathlib import Path
import sys



from pipelines import (
    MODEL_KEYS,
    DISPLAY_NAMES,
    PREPROCESSING_TEXT,
    PARAM_GRIDS,
    build_model,
    fit_model,
    score_rows,
    score_fit_rows,
    make_bundle,
    bundle_metadata,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================
ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "data" / "results"
PLOTS_DIR = RESULTS_DIR / "final_plots"
MODELS_DIR = ROOT / "models"

# One tracking database and one artifact folder, both anchored to the project root, so the result does
# not depend on the directory the script is started from.
TRACKING_DB = ROOT / "mlflow_tracking_clean.db"
TRACKING_URI = f"sqlite:///{TRACKING_DB.as_posix()}"
ARTIFACT_DIR = ROOT / "mlartifacts"

EXPERIMENT_NAMES = {key: f"AegisOT_{key}" for key in MODEL_KEYS}
COMPARISON_EXPERIMENT = "AegisOT_Models_Comparison"

RANDOM_STATE = 42
SELECTION_SETS = ["train", "val"]          # the only sets the sweep may touch
SELECTION_METRIC = "val_balanced_accuracy"
TIE_BREAK_METRIC = "val_roc_auc"
METRIC_NAMES = ["precision", "recall", "f1", "roc_auc", "fpr", "balanced_accuracy"]

OVERFIT_GAP = 0.10     # train->val drop that counts as one overfitting signal
UNDERFIT_AUC = 0.80    # ROC-AUC below this on BOTH train and val counts as underfitting

# plot colours
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

# ============================================================
# 2. DATA LOADING AND CONSISTENCY CHECKS
# ============================================================
def load_data(key):
    raw = np.load(PROCESSED_DIR / f"model_data_{key}.npz", allow_pickle=True)
    return {name: raw[name] for name in raw.files}


def load_preprocessor(key):
    return joblib.load(PROCESSED_DIR / f"preprocessor_{key}.joblib")


def check_data(key, data, meta, preprocessor):
    """Make sure the arrays, the metadata and the saved preprocessor all come from the same 02 run."""
    names = [str(n) for n in data["feature_names"]]
    if names != meta["feature_columns"]:
        raise ValueError(f"[{key}] feature names in the .npz differ from preprocessing_metadata.json. Re-run 02.")
    n_features = len(names)

    for s in ["train", "val", "test"]:
        x, y = data[f"X_{s}"], data[f"y_{s}"]
        if x.ndim != 2 or x.shape[1] != n_features:
            raise ValueError(f"[{key}] X_{s} has shape {x.shape}, expected {n_features} columns.")
        if len(x) != len(y):
            raise ValueError(f"[{key}] X_{s} and y_{s} have different lengths.")
        if not np.isfinite(x).all():
            raise ValueError(f"[{key}] X_{s} contains NaN/inf after preprocessing.")
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError(f"[{key}] y_{s} must contain only 0 (normal) and 1 (fault).")
    if (data["y_train"] == 0).sum() == 0:
        raise ValueError(f"[{key}] no normal rows in the train set.")

    if preprocessor.n_features_in_ != n_features:
        raise ValueError(f"[{key}] saved preprocessor expects {preprocessor.n_features_in_} features, data has {n_features}.")
    if "raw_probe" not in data:
        raise ValueError(f"[{key}] .npz has no raw_probe. Re-run 02_preprocessing_feature_engineering.py.")
    probe = data["raw_probe"]
    if not np.allclose(preprocessor.transform(probe), data["X_train"][:len(probe)], equal_nan=True):
        raise ValueError(f"[{key}] the saved preprocessor does not reproduce X_train. "
                        "The .joblib and the .npz are from different runs. Re-run 02.")


# ============================================================
# 3. METRICS
# ============================================================
def set_metrics(y_true, pred, score):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    try:
        auc = roc_auc_score(y_true, score)
    except ValueError:        # only one class present
        auc = np.nan
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall,
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": auc,
        "fpr": fpr,
        "balanced_accuracy": (recall + (1 - fpr)) / 2,
    }


def prefixed_metrics(split, y_true, scores, threshold):
    pred = (scores > threshold).astype(int)
    return {f"{split}_{k}": float(v) for k, v in set_metrics(y_true, pred, scores).items()}


# ============================================================
# 4. MODEL DIAGNOSIS (train + validation only, never test)
# ============================================================
def diagnose(m):
    """
    Simple, explainable diagnosis. Returns (label, reason).

    Labels: possible_overfit, possible_underfit, no_major_gap, inconclusive.

    Reasoning:
      * The threshold is the train-normal quantile, so train FPR is ~(1 - quantile) by construction.
        If validation FPR is clearly higher, normal behaviour in unseen blocks looks more anomalous than
        the fitted normal rows: that is the main overfitting symptom for a model trained on normal data only.
      * Faults are never used for fitting, so train recall is not "seen" either. A big train->val drop in
        balanced accuracy or ROC-AUC can therefore also come from different faults / conditions in the
        blocks, not only from overfitting. That is why ONE signal is never enough: overfit needs >= 2.
      * Underfit means the model separates faults poorly everywhere (ROC-AUC low on train AND validation).
        With unsupervised detectors this can also mean "these features do not show this fault".
    """
    need = ["train_roc_auc", "val_roc_auc", "train_fpr", "val_fpr",
            "train_balanced_accuracy", "val_balanced_accuracy"]
    missing = [k for k in need if not np.isfinite(m.get(k, np.nan))]
    if missing:
        return "inconclusive", f"metrics not available: {', '.join(missing)}"

    gap_ba = m["train_balanced_accuracy"] - m["val_balanced_accuracy"]
    gap_auc = m["train_roc_auc"] - m["val_roc_auc"]
    gap_fpr = m["val_fpr"] - m["train_fpr"]

    signals = []
    if gap_ba > OVERFIT_GAP:
        signals.append(f"balanced accuracy -{gap_ba:.2f} train->val")
    if gap_auc > OVERFIT_GAP:
        signals.append(f"roc_auc -{gap_auc:.2f} train->val")
    if gap_fpr > OVERFIT_GAP:
        signals.append(f"fpr +{gap_fpr:.2f} on unseen normal rows")

    if len(signals) >= 2:
        return "possible_overfit", "; ".join(signals)
    if m["train_roc_auc"] < UNDERFIT_AUC and m["val_roc_auc"] < UNDERFIT_AUC:
        return "possible_underfit", (f"roc_auc below {UNDERFIT_AUC} on train ({m['train_roc_auc']:.2f}) "
                                    f"and validation ({m['val_roc_auc']:.2f})")
    if len(signals) == 1:
        return "inconclusive", f"only one overfit signal ({signals[0]})"
    if min(m["train_roc_auc"], m["val_roc_auc"]) < UNDERFIT_AUC:
        return "inconclusive", (f"roc_auc is low on one set only (train {m['train_roc_auc']:.2f}, "
                                f"val {m['val_roc_auc']:.2f})")
    return "no_major_gap", "train and validation are close"


# ============================================================
# 5. TRAIN AND EVALUATE ONE CONFIGURATION (train + validation only)
# ============================================================
def run_config(key, params, data):
    z = {s: data[f"X_{s}"] for s in SELECTION_SETS}
    y = {s: data[f"y_{s}"].astype(int) for s in SELECTION_SETS}
    fit_mask = y["train"] == 0                      # models learn from normal train rows only

    model = build_model(key, params, RANDOM_STATE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = fit_model(key, model, z["train"][fit_mask])
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

    # train: fitted normal rows use the fit-row score (leave-one-out for LOF), fault rows are unseen
    train_scores = np.empty(len(z["train"]), dtype=float)
    train_scores[fit_mask] = score_fit_rows(key, model, z["train"][fit_mask])
    if (~fit_mask).any():
        train_scores[~fit_mask] = score_rows(key, model, z["train"][~fit_mask])
    scores = {"train": train_scores, "val": score_rows(key, model, z["val"])}

    # threshold: from NORMAL TRAIN scores only. Validation / test labels and scores never enter here.
    if key == "robust_baseline":
        threshold = float(params["z_threshold"])
    else:
        threshold = float(np.quantile(scores["train"][fit_mask], params["threshold_quantile"]))

    metrics = {}
    for s in SELECTION_SETS:
        metrics.update(prefixed_metrics(s, y[s], scores[s], threshold))
    metrics["gap_balanced_accuracy_train_minus_val"] = metrics["train_balanced_accuracy"] - metrics["val_balanced_accuracy"]
    metrics["gap_roc_auc_train_minus_val"] = metrics["train_roc_auc"] - metrics["val_roc_auc"]
    metrics["gap_fpr_val_minus_train"] = metrics["val_fpr"] - metrics["train_fpr"]

    label, reason = diagnose(metrics)

    if key == "autoencoder":
        # Note: validation_scores_ / best_validation_score_ are R2 on sklearn's INTERNAL random 15% split of
        # the normal train rows. They are not the external validation set and not a reconstruction error.
        metrics["ae_n_iter"] = float(model.n_iter_)
        metrics["ae_converged"] = float(converged)
        metrics["ae_best_internal_val_r2"] = float(getattr(model, "best_validation_score_", np.nan))
        if not converged:
            # reaching max_iter is a training-convergence warning, not proof of underfitting
            reason += " | autoencoder reached max_iter before the stopping criterion (not converged)"

    return {
        "model": model, "scores": scores, "y": y, "fit_mask": fit_mask,
        "threshold": threshold, "metrics": metrics,
        "diagnosis": label, "reason": reason,
    }


def selection_key(rec):
    """validation metric first, validation ROC-AUC second. NaN never wins."""
    def safe(v):
        return v if np.isfinite(v) else -np.inf
    return safe(rec["metrics"][SELECTION_METRIC]), safe(rec["metrics"][TIE_BREAK_METRIC])


def pick_best(records):
    return sorted(records, key=selection_key, reverse=True)[0]


# ============================================================
# 6. FINAL TEST EVALUATION (only for already selected configurations)
# ============================================================
def evaluate_on_test(rec):
    data = load_data(rec["key"])
    z, y = data["X_test"], data["y_test"].astype(int)
    scores = score_rows(rec["key"], rec["model"], z)
    return scores, y, prefixed_metrics("test", y, scores, rec["threshold"])


# ============================================================
# 7. PLOTS
# ============================================================
def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_train_vs_val(best, path):
    """selection plot: train vs validation of the best configuration of each model."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    labels = [f"{DISPLAY_NAMES[r['key']]}\n({r['diagnosis']})" for r in best]
    x = np.arange(len(best))
    w = 0.36
    for ax, metric, title in zip(axes, ["balanced_accuracy", "roc_auc"], ["Balanced accuracy", "ROC AUC"]):
        style_axes(ax)
        tr = [r["metrics"][f"train_{metric}"] for r in best]
        va = [r["metrics"][f"val_{metric}"] for r in best]
        b1 = ax.bar(x - w / 2 - 0.01, np.nan_to_num(tr), w, color=C_BLUE, label="Train")
        b2 = ax.bar(x + w / 2 + 0.01, np.nan_to_num(va), w, color=C_ORANGE, label="Validation")
        for bars, vals in ((b1, tr), (b2, va)):
            for rect, v in zip(bars, vals):
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01,
                        "n/a" if not np.isfinite(v) else f"{v:.2f}",
                        ha="center", va="bottom", color=INK, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", color=INK, fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Train vs validation, best config of each model (selection data)",
                color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_score_distribution(rec, test_scores, y_test, path):
    s, fit = rec["scores"], rec["fit_mask"]
    groups = [
        ("Normal, train (fitted)", s["train"][fit], C_BLUE),
        ("Normal, test (unseen)", test_scores[y_test == 0], C_AQUA),
        ("Faults, test", test_scores[y_test == 1], C_ORANGE),
    ]
    groups = [g for g in groups if len(g[1]) > 0]
    c = max(abs(rec["threshold"]), 1e-6)
    squash = lambda v: np.arcsinh(np.asarray(v) / c)

    allv = squash(np.concatenate([g[1] for g in groups]))
    bins = np.linspace(allv.min(), allv.max() + 1e-9, 45)

    fig, ax = plt.subplots(figsize=(8.5, 4), facecolor=SURFACE)
    style_axes(ax)
    for name, values, color in groups:
        ax.hist(squash(values), bins=bins, density=True, histtype="stepfilled",
                alpha=0.35, color=color, edgecolor=color, linewidth=1.4, label=name)
    thr = float(squash(rec["threshold"]))
    ax.axvline(thr, color=INK2, ls="--", lw=1.3)
    ax.text(thr, ax.get_ylim()[1] * 0.95, " threshold", color=INK2, fontsize=9, va="top")

    raw_ticks = np.array([-1000, -100, -10, -1, 0, 1, 10, 100, 1000, 10000, 100000]) * c
    pos = squash(raw_ticks)
    keep = (pos >= bins[0]) & (pos <= bins[-1])
    ax.set_xticks(pos[keep])
    ax.set_xticklabels([f"{v:.3g}" for v in raw_ticks[keep]])
    ax.set_xlabel("anomaly score, higher = more anomalous (arcsinh scale)", color=INK2)
    ax.set_ylabel("density", color=INK2)
    ax.set_title(f"{DISPLAY_NAMES[rec['key']]}: score distributions", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion(rec, test_scores, y_test, path):
    pred = (test_scores > rec["threshold"]).astype(int)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, pred, labels=[0, 1], display_labels=["Normal", "Fault"], cmap="Blues")
    disp.ax_.set_title(f"Test confusion matrix - {DISPLAY_NAMES[rec['key']]}")
    disp.figure_.tight_layout()
    disp.figure_.savefig(path, dpi=160)
    plt.close(disp.figure_)


def plot_ae_curve(model, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), facecolor=SURFACE)
    for ax in axes:
        style_axes(ax)
    axes[0].plot(model.loss_curve_, color=C_BLUE, lw=2)
    axes[0].set_title("Training loss (0.5*MSE + L2 penalty)", color=INK, fontsize=10, loc="left")
    axes[0].set_xlabel("epoch", color=INK2)
    scores = getattr(model, "validation_scores_", None)
    if scores:
        axes[1].plot(scores, color=C_ORANGE, lw=2)
    axes[1].set_title("Internal val R2 (random 15% of normal train rows)", color=INK, fontsize=10, loc="left")
    axes[1].set_xlabel("epoch", color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ============================================================
# 8. MLFLOW HELPERS
# ============================================================
def get_or_create_experiment(name):
    """Create the experiment once, with an artifact folder inside the project. Existing experiments are kept."""
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return mlflow.create_experiment(name, artifact_location=(ARTIFACT_DIR / name).as_uri())
    if exp.lifecycle_stage == "deleted":
        raise RuntimeError(f"Experiment '{name}' is in the MLflow trash. Restore it in the UI or with "
                        f"`mlflow experiments restore`; this script will not delete or recreate it.")
    return exp.experiment_id


def setup_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    ids = {name: get_or_create_experiment(name) for name in [*EXPERIMENT_NAMES.values(), COMPARISON_EXPERIMENT]}
    print(f"MLflow tracking DB : {TRACKING_DB}")
    print(f"MLflow artifacts   : {ARTIFACT_DIR} (for experiments created by this script)")
    return ids


def threshold_strategy(key):
    if key == "robust_baseline":
        return "fixed z_threshold (hyperparameter)"
    return "quantile of normal train scores (hyperparameter threshold_quantile)"


def config_params(key, params):
    p = {k: str(v) for k, v in params.items()}
    p.update({
        "model": DISPLAY_NAMES[key],
        "model_key": key,
        "preprocessing": PREPROCESSING_TEXT[key],
        "random_state": RANDOM_STATE,
        "threshold_strategy": threshold_strategy(key),
    })
    return p


def log_finite_metrics(metrics):
    """MLflow gets finite values only; NaN metrics are listed in a tag (and stay NaN in the csv files)."""
    ok = {k: v for k, v in metrics.items() if np.isfinite(v)}
    mlflow.log_metrics(ok)
    skipped = sorted(set(metrics) - set(ok))
    if skipped:
        mlflow.set_tag("nan_metrics", ", ".join(skipped))


def log_ae_curves(run_id, model):
    now = int(time.time() * 1000)
    batch = [Metric("ae_train_loss", float(v), now, step) for step, v in enumerate(model.loss_curve_)]
    batch += [Metric("ae_internal_val_r2", float(v), now, step)
            for step, v in enumerate(getattr(model, "validation_scores_", None) or [])]
    client = MlflowClient(tracking_uri=TRACKING_URI)
    for i in range(0, len(batch), 900):          # MLflow accepts at most 1000 metrics per batch
        client.log_batch(run_id, metrics=batch[i:i + 900])


def log_config_run(key, params, rec, index, meta, preprocessor, execution_id):
    """one child run = one hyperparameter configuration (train + validation metrics only)."""
    with mlflow.start_run(run_name=f"{key}_{index:02d}", nested=True) as run:
        rec["run_id"] = run.info.run_id
        mlflow.set_tags({
            "stage": "config",
            "model_key": key,
            "execution_id": execution_id,
            "diagnosis": rec["diagnosis"],
            "diagnosis_reason": rec["reason"],
        })
        mlflow.log_params(config_params(key, params))
        log_finite_metrics(rec["metrics"])
        mlflow.log_metric("threshold", rec["threshold"])

        if key == "autoencoder":
            log_ae_curves(run.info.run_id, rec["model"])

        bundle = make_bundle(key, preprocessor, rec["model"], params, rec["threshold"],
                            meta["feature_columns"], RANDOM_STATE)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{key}_{index:02d}.joblib"
            joblib.dump(bundle, path)
            mlflow.log_artifact(str(path), artifact_path="model_bundle")
        mlflow.log_dict(bundle_metadata(bundle), "model_bundle/metadata.json")


# ============================================================
# 9. SWEEP: ONE EXPERIMENT PER MODEL FAMILY
# ============================================================
def run_family(key, meta, experiment_ids, execution_id):
    """Run every configuration of one model family. Returns (all records, best record)."""
    print("\n" + "=" * 65)
    print(f"Experiment: {EXPERIMENT_NAMES[key]}")
    print("=" * 65)

    data = load_data(key)
    preprocessor = load_preprocessor(key)
    check_data(key, data, meta, preprocessor)

    grid = PARAM_GRIDS[key]
    combos = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    if not combos:
        raise ValueError(f"empty parameter grid for {key}")

    mlflow.set_experiment(experiment_id=experiment_ids[EXPERIMENT_NAMES[key]])
    records = []
    with mlflow.start_run(run_name=f"{key}_sweep") as parent:
        mlflow.set_tags({
            "stage": "sweep",
            "model_key": key,
            "execution_id": execution_id,
            "selection_metric": SELECTION_METRIC,
        })
        mlflow.log_params({
            "model": DISPLAY_NAMES[key],
            "preprocessing": PREPROCESSING_TEXT[key],
            "n_configurations": len(combos),
            "n_features": len(meta["feature_columns"]),
            "rows_per_set": json.dumps(meta["rows_per_set"]),
            "block_size": meta["block_size"],
            "random_state": RANDOM_STATE,
            "overfit_gap": OVERFIT_GAP,
            "underfit_auc": UNDERFIT_AUC,
        })

        for i, params in enumerate(combos):
            print(f"[{key}] configuration {i + 1}/{len(combos)}: {params}")
            rec = run_config(key, params, data)
            rec.update({"key": key, "params": params, "index": i})
            log_config_run(key, params, rec, i, meta, preprocessor, execution_id)
            records.append(rec)

        best = pick_best(records)
        # sweep parent: validation-based summary only. No test numbers exist yet at this point.
        mlflow.set_tags({
            "best_run_id": best["run_id"],
            "best_params": json.dumps(best["params"]),
            "best_diagnosis": best["diagnosis"],
        })
        log_finite_metrics({
            f"best_{SELECTION_METRIC}": best["metrics"][SELECTION_METRIC],
            f"best_{TIE_BREAK_METRIC}": best["metrics"][TIE_BREAK_METRIC],
        })
        best["sweep_run_id"] = parent.info.run_id

    # keep memory small: only the selected model is needed later (for test evaluation and saving)
    for rec in records:
        if rec is not best:
            rec["model"] = None
    print(f"[{key}] best by validation: {SELECTION_METRIC} = {best['metrics'][SELECTION_METRIC]:.4f}")
    return records, best


# ============================================================
# 10. TABLES
# ============================================================
def build_selection_tables(records, best_by_key):
    """Validation-based tables. No test numbers in here."""
    all_runs = pd.DataFrame([
        {"run_id": r["run_id"], "key": r["key"], "model": DISPLAY_NAMES[r["key"]],
        "params": json.dumps(r["params"]), "threshold": r["threshold"],
         "diagnosis": r["diagnosis"], "reason": r["reason"], **r["metrics"]}
        for r in records
    ])
    all_runs.to_csv(RESULTS_DIR / "all_runs.csv", index=False)

    ranked = sorted(best_by_key.values(), key=selection_key, reverse=True)
    rows = []
    for rank, r in enumerate(ranked, start=1):
        m = r["metrics"]
        row = {"rank_by_validation": rank, "selected": rank == 1, "model": DISPLAY_NAMES[r["key"]],
            "key": r["key"], "preprocessing": PREPROCESSING_TEXT[r["key"]],
            "params": json.dumps(r["params"]), "threshold": r["threshold"], "run_id": r["run_id"]}
        for s in SELECTION_SETS:
            for c in METRIC_NAMES:
                row[f"{s}_{c}"] = m[f"{s}_{c}"]
        row["gap_balanced_accuracy_train_minus_val"] = m["gap_balanced_accuracy_train_minus_val"]
        row["gap_roc_auc_train_minus_val"] = m["gap_roc_auc_train_minus_val"]
        row["gap_fpr_val_minus_train"] = m["gap_fpr_val_minus_train"]
        row["diagnosis"], row["reason"] = r["diagnosis"], r["reason"]
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    return ranked, comparison


def build_final_test_table(ranked):
    """Final report. Rows keep the validation ranking, they are not re-ranked by test results."""
    rows = []
    for rank, r in enumerate(ranked, start=1):
        row = {"rank_by_validation": rank, "selected": rank == 1, "model": DISPLAY_NAMES[r["key"]],
            "key": r["key"], "params": json.dumps(r["params"]), "threshold": r["threshold"],
            "source_run_id": r["run_id"], SELECTION_METRIC: r["metrics"][SELECTION_METRIC]}
        row.update(r["test_metrics"])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "final_test_results.csv", index=False)
    return table


# ============================================================
# 11. SAVE AND VERIFY THE WINNER BUNDLE
# ============================================================
def save_best_model(winner, meta):
    key = winner["key"]
    bundle = make_bundle(key, load_preprocessor(key), winner["model"], winner["params"],
                        winner["threshold"], meta["feature_columns"], RANDOM_STATE)
    path = MODELS_DIR / "best_model.joblib"
    joblib.dump(bundle, path)
    with open(MODELS_DIR / "best_model_metadata.json", "w") as f:
        json.dump(bundle_metadata(bundle), f, indent=4)
    return bundle


def verify_bundle(winner):
    """Reload the saved bundle and check it scores exactly like the training pipeline."""
    bundle = joblib.load(MODELS_DIR / "best_model.joblib")
    data = load_data(winner["key"])
    probe = data["raw_probe"]
    z_bundle = bundle["preprocessor"].transform(probe)
    z_train = data["X_train"][:len(probe)]
    if not np.allclose(z_bundle, z_train, equal_nan=True):
        raise RuntimeError("bundle preprocessor does not reproduce X_train")
    s_bundle = score_rows(bundle["key"], bundle["model"], z_bundle)
    s_train = score_rows(winner["key"], winner["model"], z_train)
    if not np.allclose(s_bundle, s_train):
        raise RuntimeError("bundle scores differ from training-time scores")
    if bundle["threshold"] != winner["threshold"] or bundle["features"] != list(data["feature_names"]):
        raise RuntimeError("bundle threshold / feature list differ from the training run")
    print("Bundle check passed: preprocessing, scores, threshold and feature order match the training run.")


# ============================================================
# 12. FINAL COMPARISON EXPERIMENT (selection summary + final test report)
# ============================================================
def log_final_comparison(ranked, winner, meta, experiment_ids, execution_id):
    mlflow.set_experiment(experiment_id=experiment_ids[COMPARISON_EXPERIMENT])
    with mlflow.start_run(run_name="final_comparison") as parent:
        mlflow.set_tags({
            "stage": "final_comparison",
            "execution_id": execution_id,
            "selected_model": DISPLAY_NAMES[winner["key"]],
            "selected_by": f"validation ({SELECTION_METRIC}, tie-break {TIE_BREAK_METRIC})",
            "selected_config_run_id": winner["run_id"],
            "test_note": "test metrics are reported after selection and were not used to choose anything",
        })
        mlflow.log_params({
            "selected_model_key": winner["key"],
            "selected_params": json.dumps(winner["params"]),
            "selection_metric": SELECTION_METRIC,
            "random_state": RANDOM_STATE,
        })
        log_finite_metrics({
            "selected_val_balanced_accuracy": winner["metrics"]["val_balanced_accuracy"],
            "selected_val_roc_auc": winner["metrics"]["val_roc_auc"],
            "selected_test_balanced_accuracy": winner["test_metrics"]["test_balanced_accuracy"],
            "selected_test_roc_auc": winner["test_metrics"]["test_roc_auc"],
        })

        # one child per model family: its validation-selected configuration, with validation and test metrics
        for r in ranked:
            key = r["key"]
            with mlflow.start_run(run_name=f"final_{key}", nested=True):
                mlflow.set_tags({
                    "stage": "final_family_best",
                    "model_key": key,
                    "execution_id": execution_id,
                    "source_run_id": r["run_id"],
                    "is_selected_model": str(r is winner),
                    "diagnosis": r["diagnosis"],
                })
                mlflow.log_params(config_params(key, r["params"]))
                val_and_train = {k: v for k, v in r["metrics"].items()
                                if k.startswith(("train_", "val_", "gap_"))}
                log_finite_metrics({**val_and_train, **r["test_metrics"], "threshold": r["threshold"]})
                mlflow.log_artifact(str(PLOTS_DIR / f"{key}_confusion_matrix.png"), artifact_path="plots")
                mlflow.log_artifact(str(PLOTS_DIR / f"{key}_score_distribution.png"), artifact_path="plots")
                if key == "autoencoder":
                    mlflow.log_artifact(str(PLOTS_DIR / "autoencoder_learning_curve.png"), artifact_path="plots")

        report = {
            "selected_model_by_validation": DISPLAY_NAMES[winner["key"]],
            "selection_metric": SELECTION_METRIC,
            "tie_break_metric": TIE_BREAK_METRIC,
            "params": {k: str(v) for k, v in winner["params"].items()},
            "threshold": winner["threshold"],
            "diagnosis": winner["diagnosis"],
            "diagnosis_reason": winner["reason"],
            "feature_names": meta["feature_columns"],
            "validation_metrics": {k: v for k, v in winner["metrics"].items() if k.startswith("val_")},
            "final_test_metrics (not used for selection)": winner["test_metrics"],
            "mlflow_config_run_id": winner["run_id"],
            "mlflow_sweep_run_id": winner["sweep_run_id"],
            "mlflow_final_comparison_run_id": parent.info.run_id,
            "mlflow_execution_id": execution_id,
            "warning": ("Selected on a single recording per condition. Check recall and false positive "
                        "rate on new runs before deploying."),
        }
        with open(MODELS_DIR / "best_model.json", "w") as f:
            json.dump(report, f, indent=4, default=str)

        for name in ["model_comparison.csv", "final_test_results.csv", "all_runs.csv"]:
            mlflow.log_artifact(str(RESULTS_DIR / name), artifact_path="tables")
        mlflow.log_artifact(str(RESULTS_DIR / "train_vs_val.png"), artifact_path="plots")
        mlflow.log_artifact(str(MODELS_DIR / "best_model.joblib"), artifact_path="best_model")
        mlflow.log_artifact(str(MODELS_DIR / "best_model_metadata.json"), artifact_path="best_model")
        mlflow.log_artifact(str(MODELS_DIR / "best_model.json"), artifact_path="best_model")
        return parent.info.run_id


# ============================================================
# 13. MAIN
# ============================================================
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    missing = [k for k in MODEL_KEYS if not (PROCESSED_DIR / f"model_data_{k}.npz").exists()
            or not (PROCESSED_DIR / f"preprocessor_{k}.joblib").exists()]
    if missing or not (PROCESSED_DIR / "preprocessing_metadata.json").exists():
        raise FileNotFoundError(f"Run 02_preprocessing_feature_engineering.py first (missing files for: {missing}).")

    meta = json.loads((PROCESSED_DIR / "preprocessing_metadata.json").read_text())
    experiment_ids = setup_mlflow()
    execution_id = time.strftime("%Y%m%d_%H%M%S")
    print(f"Execution id: {execution_id}")

    # ---- 1. sweep (train + validation only) ----
    records, best_by_key = [], {}
    for key in MODEL_KEYS:
        family_records, best = run_family(key, meta, experiment_ids, execution_id)
        records += family_records
        best_by_key[key] = best

    # ---- 2. selection (validation only). From here on the selection is frozen. ----
    ranked, comparison = build_selection_tables(records, best_by_key)
    winner = ranked[0]
    plot_train_vs_val(ranked, RESULTS_DIR / "train_vs_val.png")

    # ---- 3. final test evaluation of the selected configuration of each family ----
    for r in ranked:
        test_scores, y_test, r["test_metrics"] = evaluate_on_test(r)
        plot_confusion(r, test_scores, y_test, PLOTS_DIR / f"{r['key']}_confusion_matrix.png")
        plot_score_distribution(r, test_scores, y_test, PLOTS_DIR / f"{r['key']}_score_distribution.png")
        if r["key"] == "autoencoder":
            plot_ae_curve(r["model"], PLOTS_DIR / "autoencoder_learning_curve.png")
    test_table = build_final_test_table(ranked)

    # ---- 4. save the validation winner and verify the bundle ----
    save_best_model(winner, meta)
    verify_bundle(winner)

    # ---- 5. MLflow comparison experiment ----
    final_run_id = log_final_comparison(ranked, winner, meta, experiment_ids, execution_id)

    # ---- console summary ----
    show = ["model", "train_balanced_accuracy", "val_balanced_accuracy", "val_roc_auc", "val_fpr", "diagnosis"]
    fmt = lambda v: f"{v:.3f}"
    print("\n" + "=" * 75)
    print("SELECTION (validation): best configuration of each model, ranked by validation")
    print("=" * 75)
    print(comparison[show].to_string(index=False, float_format=fmt))

    print("\n" + "=" * 75)
    print("FINAL TEST REPORT (selected configurations, not used for selection)")
    print("=" * 75)
    print(test_table[["model", "test_balanced_accuracy", "test_roc_auc", "test_recall", "test_fpr"]]
        .to_string(index=False, float_format=fmt))

    print("\n" + "=" * 75)
    print(f"Selected by validation: {DISPLAY_NAMES[winner['key']]}")
    print(f"Parameters            : {winner['params']}")
    print(f"Validation BA         : {winner['metrics']['val_balanced_accuracy']:.4f}")
    print(f"Test BA (report only) : {winner['test_metrics']['test_balanced_accuracy']:.4f}")
    print(f"Diagnosis             : {winner['diagnosis']} ({winner['reason']})")
    print(f"MLflow config run id  : {winner['run_id']}")
    print(f"MLflow comparison run : {final_run_id}")
    print(f"Results saved in      : {RESULTS_DIR}")
    print("\nOpen the MLflow UI with:")
    print(f"  mlflow ui --backend-store-uri {TRACKING_URI}")

if __name__ == "__main__":
    main()
