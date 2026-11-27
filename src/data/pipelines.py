import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import OneClassSVM

MODEL_KEYS = ["robust_baseline", "isolation_forest", "lof", "ocsvm", "autoencoder"]
ENCODER_CPR = 836 
DISPLAY_NAMES = {
    "robust_baseline": "Robust Statistical Baseline",
    "isolation_forest": "Isolation Forest",
    "lof": "LOF",
    "ocsvm": "One-Class SVM",
    "autoencoder": "Neural Autoencoder",
}

PREPROCESSING_TEXT = {
    "robust_baseline": "median imputer -> median/MAD robust z",
    "isolation_forest": "median imputer",
    "lof": "median imputer -> RobustScaler",
    "ocsvm": "median imputer -> RobustScaler",
    "autoencoder": "median imputer -> StandardScaler",
}

PARAM_GRIDS = {
    "robust_baseline": {"z_threshold": [3.5, 5.0, 8.0]},
    "isolation_forest": {
        "n_estimators": [100, 300],
        "max_samples": ["auto", 128],
        "threshold_quantile": [0.95, 0.99],
    },
   "lof": {"n_neighbors": [10, 20, 30, 40, 60],"threshold_quantile": [0.95, 0.975, 0.99, 0.995],},
    "ocsvm": {
        "nu": [0.02, 0.05, 0.15],
        "gamma": ["scale", "auto"],
        "threshold_quantile": [0.95, 0.99],
    },
    "autoencoder": {
        "hidden_layer_sizes": [
            (32, 8, 32),
            (32, 16, 32),
            (64, 16, 64)
        ],
        "alpha": [0.0001, 0.001],
        "learning_rate_init": [0.0005, 0.001],
        "threshold_quantile": [0.95, 0.99],
    },
}

def add_features(frame):
    """Feature engineering shared by training (02 script) and inference (predict_raw).

    Moved here unchanged from 02_preprocessing_feature_engineering.py so a saved bundle can be applied
    to raw sensor rows without re-implementing these steps by hand.
    """
    data = frame.copy()

    if {"accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2"}.issubset(data.columns):
        data["acceleration_magnitude"] = np.sqrt(
            data["accel_x_m_s2"] ** 2
            + data["accel_y_m_s2"] ** 2
            + data["accel_z_m_s2"] ** 2
        )

    if {"gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s"}.issubset(data.columns):
        data["gyro_magnitude"] = np.sqrt(
            data["gyro_x_rad_s"] ** 2
            + data["gyro_y_rad_s"] ** 2
            + data["gyro_z_rad_s"] ** 2
        )

    # if {"speed_percent", "motor_running"}.issubset(data.columns):
    #     data["speed_when_running"] = data["speed_percent"] * data["motor_running"]

    if {"temperature_c", "temperature_valid"}.issubset(data.columns):
        data.loc[data["temperature_valid"] != 1, "temperature_c"] = np.nan

    if {"encoder_counts_per_second", "encoder_counts_per_revolution"}.issubset(data.columns):
        if "rpm" not in data.columns:
            data["rpm"] = np.nan
        else:
            data["rpm"] = pd.to_numeric(
                data["rpm"], errors="coerce"
            ).astype(float)

        wrong_cpr = data["encoder_counts_per_revolution"] != ENCODER_CPR

        data.loc[wrong_cpr, "rpm"] = (
            data.loc[wrong_cpr, "encoder_counts_per_second"] / ENCODER_CPR * 60
        )
    return data


class RobustZ(BaseEstimator, TransformerMixin):
    """median / MAD z-score, same formula as the original baseline."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.median_ = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.median_), axis=0)
        # a column that is nearly constant gives mad ~ 0 and any tiny change blows up the score
        self.mad_ = np.maximum(mad, 0.25 * X.std(axis=0) + 1e-9)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return 0.6745 * (X - self.median_) / self.mad_


def build_preprocessor(key):
    impute = ("impute", SimpleImputer(strategy="median"))

    if key == "robust_baseline":
        return Pipeline([impute, ("robust_z", RobustZ())])
    if key == "isolation_forest":
        # trees split on order only, scaling would change nothing
        return Pipeline([impute])
    if key in ("lof", "ocsvm"):
        # both are distance / kernel based so the scale of each column matters
        return Pipeline([impute, ("scale", RobustScaler())])
    if key == "autoencoder":
        return Pipeline([impute, ("scale", StandardScaler())])
    raise ValueError(f"unknown model key: {key}")


def build_model(key, params, seed):
    if key == "robust_baseline":
        return None
    if key == "isolation_forest":
        return IsolationForest(
            n_estimators=params["n_estimators"],
            max_samples=params["max_samples"],
            random_state=seed,
        )
    if key == "lof":
        return LocalOutlierFactor(n_neighbors=params["n_neighbors"], novelty=True)
    if key == "ocsvm":
        return OneClassSVM(kernel="rbf", gamma=params["gamma"], nu=params["nu"])
    if key == "autoencoder":
        return MLPRegressor(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            activation="relu",
            solver="adam",
            alpha=params["alpha"],
            learning_rate_init=params["learning_rate_init"],
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            tol=1e-5,
            random_state=seed,
        )
    raise ValueError(f"unknown model key: {key}")



def fit_model(key, model, Z):
    """
    Fit an anomaly detection model using training data.

    Robust Baseline learns through its preprocessor.
    Autoencoder uses input data as its reconstruction target.
    Other models learn from input data only.
    """
    if key not in MODEL_KEYS:
        raise ValueError(f"Unknown model key: {key}")

    if Z is None or len(Z) == 0:
        raise ValueError("Training data is empty.")

    if key == "robust_baseline":
        return None

    if model is None:
        raise ValueError(f"Model is not initialized: {key}")

    if key == "autoencoder":
        return model.fit(Z, Z)

    return model.fit(Z)



def score_rows(key, model, Z):
    """higher = more anomalous, for every model. Use this for rows the model has NOT been fitted on.

    LOF: we return the plain Local Outlier Factor (-score_samples). We do NOT use -decision_function:
    decision_function = score_samples - offset_, and offset_ is -1.5 for contamination="auto", so
    -decision_function = LOF - 1.5. negative_outlier_factor_ (used for the fitted rows) has no such offset,
    so mixing the two put the threshold 1.5 units away from the validation/test scores.
    """
    if key == "robust_baseline":
        return np.max(np.abs(Z), axis=1)
    if key == "autoencoder":
        return np.mean((model.predict(Z) - Z) ** 2, axis=1)
    if key == "lof":
        return -model.score_samples(Z)
    return -model.decision_function(Z).ravel()

def explain_lof_features(model, Z, feature_names, top_n=5):
    """
    Estimate local feature contributions for LOF
    by replacing each scaled feature with its reference median.
    """
    explanations = []
    for row in Z:
        # Calculate the original anomaly score
        original_score = -float(model.score_samples(row.reshape(1, -1))[0])
        contributions = []
        for i, feature_name in enumerate(feature_names):

            # Create a modified copy of the input row
            modified_row = row.copy()

            # Zero represents the median after RobustScaler centering
            modified_row[i] = 0.0

            # Calculate the anomaly score after neutralizing this feature
            modified_score = -float(
                model.score_samples(modified_row.reshape(1, -1))[0]
            )

            # Positive contribution means this feature increased the anomaly score
            contribution = original_score - modified_score

            contributions.append({
                "feature": feature_name,
                "contribution": float(contribution),
                "score_after_removal": float(modified_score),
            })

        # Rank features by their positive contribution
        contributions.sort(
            key=lambda x: x["contribution"],
            reverse=True
        )

        explanations.append(contributions[:top_n])
    return explanations

def score_fit_rows(key, model, Z):
    """scores for the rows the model was fitted on, on the SAME scale as score_rows.

    LOF with novelty=True counts a training point as its own neighbour when you score it again,
    which makes the training scores look too good. negative_outlier_factor_ is the leave-one-out version
    and is on the same scale as score_samples (both are -LOF), so -negative_outlier_factor_ is comparable
    with score_rows for unseen rows. (Checked against the installed scikit-learn, see smoke_test.py.)
    Z must be exactly the rows passed to fit_model, in the same order.
    """
    if key == "lof":
        if len(Z) != len(model.negative_outlier_factor_):
            raise ValueError("score_fit_rows needs the exact rows the LOF model was fitted on.")
        return -model.negative_outlier_factor_
    return score_rows(key, model, Z)

# model bundle: everything needed to score new raw rows
def make_bundle(key, preprocessor, model, params, threshold, feature_names, random_state):
    return {
        "bundle_version": 1,
        "key": key,
        "display_name": DISPLAY_NAMES[key],
        "preprocessor": preprocessor,
        "model": model,                      # None for robust_baseline (it lives in the preprocessor)
        "params": dict(params),
        "threshold": float(threshold),
        "features": list(feature_names),     # names AND order the preprocessor expects
        "score_direction": "higher = more anomalous; flag when score > threshold",
        "random_state": random_state,
        "sklearn_version": sklearn.__version__,
    }


def bundle_metadata(bundle):
    """json-safe description of a bundle (no fitted objects)."""
    return {
        "bundle_version": bundle["bundle_version"],
        "key": bundle["key"],
        "display_name": bundle["display_name"],
        "preprocessing": PREPROCESSING_TEXT[bundle["key"]],
        "params": {k: str(v) for k, v in bundle["params"].items()},
        "threshold": bundle["threshold"],
        "score_direction": bundle["score_direction"],
        "n_features": len(bundle["features"]),
        "features": bundle["features"],
        "random_state": bundle["random_state"],
        "sklearn_version": bundle["sklearn_version"],
        "model_class": type(bundle["model"]).__name__ if bundle["model"] is not None else "none (robust z in preprocessor)",
        "inference": "predict_raw(bundle, raw_frame) from pipelines.py (needs pipelines.py importable when loading)",
    }


def predict_raw(bundle, raw_frame):
    """raw sensor rows -> (anomaly scores, 0/1 flags). Same steps as 02 script + 03 script.

    raw_frame needs the same raw columns the training csv had (feature engineering is applied here).
    """
    data = add_features(raw_frame)
    missing = [c for c in bundle["features"] if c not in data.columns]
    if missing:
        raise ValueError(f"raw input is missing columns needed by the model: {missing}")
    x = data[bundle["features"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    z = bundle["preprocessor"].transform(x)
    scores = score_rows(bundle["key"], bundle["model"], z)
    return scores, (scores > bundle["threshold"]).astype(int)


def predict_raw_with_top_features(bundle, raw_frame, top_n=5):
    """
    Return anomaly scores, prediction flags, and feature explanations.

    Autoencoder:
        Uses reconstruction error per feature.

    LOF:
        Uses feature ablation to estimate local feature contributions.
        Each feature is temporarily replaced with its transformed
        training median, and the change in anomaly score is measured.
    """

    if raw_frame.empty:
        raise ValueError("raw_frame must contain at least one sensor row.")

    data = add_features(raw_frame)
    missing = [
        c for c in bundle["features"]
        if c not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    raw_values = data[bundle["features"]].apply(
        pd.to_numeric, errors="coerce"
    )

    x = raw_values.to_numpy(dtype=float)
    # Apply the original fitted preprocessing pipeline
    z = bundle["preprocessor"].transform(x)

    # Calculate original anomaly scores and predictions
    scores = score_rows(bundle["key"], bundle["model"], z)
    flags = (scores > bundle["threshold"]).astype(int)
    explanations = []

    # --------------------------------------------------
    # Autoencoder Feature Explanation
    # --------------------------------------------------
    if bundle["key"] == "autoencoder":

        reconstructed = bundle["model"].predict(z)
        errors = (reconstructed - z) ** 2
        imputer = bundle["preprocessor"].named_steps.get("impute")
        medians = getattr(
            imputer,
            "statistics_",
            np.full(len(bundle["features"]), np.nan)
        )

        for row_idx in range(len(raw_values)):
            total_error = float(errors[row_idx].sum())
            indices = np.argsort(errors[row_idx])[::-1][:top_n]
            row_features = []

            for feature_idx in indices:

                value = raw_values.iloc[row_idx, feature_idx]
                baseline = medians[feature_idx]

                value_num = None if pd.isna(value) else float(value)
                baseline_num = (
                    None if pd.isna(baseline) else float(baseline)
                )

                if value_num is None or baseline_num is None:
                    direction = "unknown"
                elif value_num > baseline_num:
                    direction = "high"
                elif value_num < baseline_num:
                    direction = "low"
                else:
                    direction = "normal"

                row_features.append({
                    "name": bundle["features"][feature_idx],
                    "contribution": (
                        float(errors[row_idx, feature_idx] / total_error)
                        if total_error > 0 else 0.0
                    ),
                    "value": value_num,
                    "baseline": baseline_num,
                    "direction": direction,
                })

            explanations.append(row_features)

    # --------------------------------------------------
    # LOF Feature Explanation
    # --------------------------------------------------
    elif bundle["key"] == "lof":
        model = bundle["model"]
        feature_names = bundle["features"]
        # Get the fitted imputer's training medians
        imputer = bundle["preprocessor"].named_steps.get("impute")

        if imputer is None:
            raise ValueError(
                "LOF explanation requires an imputer named 'impute'."
            )
        medians = imputer.statistics_

        # Transform the reference medians using the fitted preprocessing
        reference_raw = np.asarray(medians, dtype=float).reshape(1, -1)
        reference_z = bundle["preprocessor"].transform(reference_raw)[0]
        for row_idx in range(len(raw_values)):
            original_row = np.asarray(z[row_idx], dtype=float).copy()
            original_score = float(scores[row_idx])
            row_features = []

            for feature_idx, feature_name in enumerate(feature_names):

                # Replace one feature with its transformed training median
                modified_row = original_row.copy()
                modified_row[feature_idx] = reference_z[feature_idx]

                modified_z = modified_row.reshape(1, -1)

                # Recalculate the anomaly score without this feature's deviation
                modified_score = float(
                    score_rows("lof", model, modified_z)[0]
                )

                # Positive contribution means the feature increased anomaly score
                contribution = original_score - modified_score

                value = raw_values.iloc[row_idx, feature_idx]
                baseline = medians[feature_idx]

                value_num = None if pd.isna(value) else float(value)
                baseline_num = (
                    None if pd.isna(baseline) else float(baseline)
                )

                if value_num is None or baseline_num is None:
                    direction = "unknown"
                elif value_num > baseline_num:
                    direction = "high"
                elif value_num < baseline_num:
                    direction = "low"
                else:
                    direction = "normal"

                row_features.append({
                    "name": feature_name,
                    "contribution": float(contribution),
                    "value": value_num,
                    "baseline": baseline_num,
                    "direction": direction,
                    "score_after_neutralizing": modified_score,
                })
            # Rank features by their contribution to the anomaly score
            row_features.sort(
                key=lambda item: item["contribution"],
                reverse=True)
            explanations.append(row_features[:top_n])
    else:
        explanations = [
            [] for _ in range(len(raw_values))]
    return scores, flags, explanations