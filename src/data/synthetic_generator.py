import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

RANDOM_SEED = 42

EXCLUDED_COLUMNS = [
    "record_id",
    "File",
    "Date",
    "data_source",
    "data_type",
    "label",
    "fault_type",
]

NON_NEGATIVE_COLUMNS = [
    "speed_rpm",
    "speed_Hz",
    "Voltage",
    "temp_mot",
    "temp_amb",
]


def get_numeric_features(df):
    return [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
    ]


def generate_synthetic_data(
    real_data,
    n_samples=3000,
    k=5,                 # 3, 5, 10, 15
    noise_scale=0.15,    # 0.05 ,0.10, i already tried 
    match_std=False,   
):
    """
    Neighbour-based interpolation (SMOTE-like) + small correlated noise.

    Each synthetic row is a blend of a real row and one of its k nearest
    neighbours (in standardized space), so samples stay inside the same
    operating regime and never bridge the gaps between regimes.

    Parameters
    ----------
    real_data   : DataFrame with the real measurements
    n_samples   : number of synthetic rows to generate
    k           : number of nearest neighbours to choose from
    noise_scale : noise size as a fraction of each feature's std
    match_std   : rescale synthetic features so their std matches the real std
    """
    rng = np.random.default_rng(RANDOM_SEED)

    features = get_numeric_features(real_data)
    real_values = real_data[features].astype(float).reset_index(drop=True)

    means = real_values.mean()
    stds = real_values.std().replace(0, 1)

    # Standardize
    Z = ((real_values - means) / stds).values

    # Nearest neighbours (column 0 is the point itself)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, neighbours = nn.kneighbors(Z)

    idx_a = rng.integers(0, len(Z), n_samples)
    pick = rng.integers(1, k + 1, n_samples)
    idx_b = neighbours[idx_a, pick]

    alpha = rng.uniform(0.0, 1.0, size=(n_samples, 1))
    synth_z = Z[idx_a] * alpha + Z[idx_b] * (1 - alpha)

    # Correlated noise in standardized space
    cov = LedoitWolf().fit(Z).covariance_
    noise = rng.multivariate_normal(
        mean=np.zeros(len(features)),
        cov=cov,
        size=n_samples,
    )
    synth_z += noise * noise_scale

    # Optional: restore the spread lost by interpolation
    if match_std:
        z_mean = synth_z.mean(axis=0)
        z_std = synth_z.std(axis=0)
        z_std[z_std == 0] = 1
        synth_z = (synth_z - z_mean) / z_std * Z.std(axis=0) + Z.mean(axis=0)

    # Back to original units
    synthetic = pd.DataFrame(
        synth_z * stds.values + means.values,
        columns=features,
    )

    # Physical consistency
    if "speed_rpm" in synthetic.columns and "speed_Hz" in synthetic.columns:
        synthetic["speed_Hz"] = synthetic["speed_rpm"] / 60

    # Non-negative measurements
    for column in NON_NEGATIVE_COLUMNS:
        if column in synthetic.columns:
            synthetic[column] = synthetic[column].clip(lower=0)
    return synthetic
