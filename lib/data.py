"""Data loading and pool construction for predictive modeling.

Reads parcel-stat TSVs from the LACUNA derivatives layout, builds mask-presence
indices and restricted subject pools, and prepares demographics design matrices.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

ATLAS_CONFIGS = [
    ("schaefer2018parcels100networks7", r"\s+"),
    ("tian2020parcels16",               "\t"),
    ("nettekoven32",                    r"\s+"),
    ("xtract",                          "\t"),
]


def _parcelstats_path(derivatives_dir, label, method, atlas, desc):
    fname = (
        f"group_ses-01_label-{label}_method-{method}_atlas-{atlas}"
        f"_desc-{desc}_parcelstats.tsv"
    )
    return Path(derivatives_dir) / fname


def _read_parcelstats(path, sep):
    df = pd.read_csv(path, sep=sep, engine="python", index_col="participant_id")
    for col in ("session_id", "label"):
        if col in df.columns:
            df = df.drop(columns=col)
    return df


def load_label(derivatives_dir, label, method, desc, atlases=None):
    """Load and concatenate parcel-stat TSVs across atlases for one label."""
    if atlases is None:
        atlases = ATLAS_CONFIGS
    frames = []
    for atlas, sep in atlases:
        path = _parcelstats_path(derivatives_dir, label, method, atlas, desc)
        frames.append(_read_parcelstats(path, sep))
    return pd.concat(frames, axis=1)


def mask_presence_index(
    derivatives_dir, label, method="rd", desc="damagepct", atlas="xtract"
):
    """Subject index for one canonical TSV (atlases share coverage per label/method)."""
    sep = "\t"
    for name, atlas_sep in ATLAS_CONFIGS:
        if name == atlas:
            sep = atlas_sep
            break
    path = _parcelstats_path(derivatives_dir, label, method, atlas, desc)
    return _read_parcelstats(path, sep).index


def restricted_pool(derivatives_dir, clinical_df, all_of, any_of=None):
    """Subjects in clinical_df present for every label in all_of (and any in any_of)."""
    pool = clinical_df.index
    for label in all_of:
        pool = pool.intersection(mask_presence_index(derivatives_dir, label))
    if any_of:
        union = None
        for label in any_of:
            idx = mask_presence_index(derivatives_dir, label)
            union = idx if union is None else union.union(idx)
        pool = pool.intersection(union)
    return pool


def build_demographics_matrix(clinical_df, exclude_cohort=False):
    """One-hot encode categorical demographics and stack with numerical covariates."""
    covariate_cols = [
        c for c in clinical_df.columns
        if c.startswith("demographics-")
        or c.startswith("stroke-")
        or c.startswith("vascularrisk-")
    ]
    if exclude_cohort and "demographics-cohort" in covariate_cols:
        covariate_cols = [c for c in covariate_cols if c != "demographics-cohort"]

    categorical_candidates = [
        "demographics-ethnicity",
        "demographics-education_STROKOG",
        "demographics-cohort",
    ]
    categorical_cols = [c for c in categorical_candidates if c in covariate_cols]
    numerical_cols = [c for c in covariate_cols if c not in categorical_cols]

    encoder = OneHotEncoder(
        drop="first", sparse_output=False, handle_unknown="ignore"
    )
    encoded_cats = encoder.fit_transform(clinical_df[categorical_cols])
    X_num = clinical_df[numerical_cols].values
    X = np.hstack([X_num, encoded_cats])

    feature_names = list(numerical_cols)
    for col, cats in zip(categorical_cols, encoder.categories_):
        # drop='first' removes the first category; remaining are the kept ones.
        for cat in cats[1:]:
            feature_names.append(f"{col}={cat}")

    return X, feature_names, encoder, categorical_cols, numerical_cols


def identify_binary_targets(clinical_df):
    """Numeric cognition columns (excluding zscore) with exactly 2 unique non-null values."""
    numeric_cols = clinical_df.select_dtypes(include=[np.number]).columns
    candidates = [c for c in numeric_cols if "cognition" in c and "zscore" not in c]
    return [c for c in candidates if clinical_df[c].dropna().nunique() == 2]
