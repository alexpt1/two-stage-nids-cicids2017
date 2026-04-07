import os
import glob
import pandas as pd
import numpy as np
import torch
from scipy import stats as sp_stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


LABEL_COL = "Label"
DROP_COLS = ["Fwd Header Length.1"]

BENIGN_LABEL = "BENIGN"


def _load_and_clean_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    for col in DROP_COLS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df


def load_cicids2017_raw(data_dir: str):
    #Using only Monday for normal traffic VAE training, and all attack days for Stage 2 classification.
    monday_path = os.path.join(data_dir, "Monday-WorkingHours.pcap_ISCX.csv")
    attack_paths = sorted([
        p for p in glob.glob(os.path.join(data_dir, "*.csv"))
        if "Monday" not in p
    ])

    print("Loading Monday (normal traffic)...")
    df_monday = _load_and_clean_csv(monday_path)
    print(f"  Monday: {len(df_monday):,} records")

    attack_dfs = []
    for path in attack_paths:
        name = os.path.basename(path)
        df = _load_and_clean_csv(path)
        print(f"  {name}: {len(df):,} records")
        attack_dfs.append(df)

    df_attacks = pd.concat(attack_dfs, ignore_index=True)
    print(f"Total attack-day records: {len(df_attacks):,}")

    return df_monday, df_attacks


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c != LABEL_COL]


VARIANCE_FLOOR = 1e-8
SKEW_THRESHOLD = 10.0
CLIP_LOWER_PCT = 1.0
CLIP_UPPER_PCT = 99.0


def audit_features(
    df_reference: pd.DataFrame,
    feature_cols: list,
    variance_floor: float = VARIANCE_FLOOR,
    skew_threshold: float = SKEW_THRESHOLD,
):
    """
    Audit features on a reference DataFrame (Monday normal traffic).
    1. Drop columns with variance < variance_floor (near-constant).
    2. Identify columns with |skew| > skew_threshold for log1p transform.

    Returns:
        surviving_cols:      list of column names kept after dropping low-variance
        log_transform_cols:  list of column names that need log1p transform
        audit_report:        dict with full audit details for logging
    """
    X = df_reference[feature_cols]

    variances = X.var()
    low_var_mask = variances < variance_floor
    dropped_cols = list(variances[low_var_mask].index)
    surviving_cols = [c for c in feature_cols if c not in dropped_cols]

    if dropped_cols:
        print(f"\n  Feature audit: dropping {len(dropped_cols)} near-zero-variance columns:")
        for col in dropped_cols:
            print(f"    {col:40s} var={variances[col]:.2e}")
    else:
        print("\n  Feature audit: no near-zero-variance columns found.")

    X_surviving = X[surviving_cols]
    skewness = X_surviving.apply(lambda c: float(sp_stats.skew(c, nan_policy="omit")))
    high_skew_mask = skewness.abs() > skew_threshold
    log_transform_cols = list(skewness[high_skew_mask].index)

    if log_transform_cols:
        print(f"  Feature audit: log1p-transforming {len(log_transform_cols)} highly skewed columns:")
        for col in log_transform_cols:
            print(f"    {col:40s} skew={skewness[col]:+.1f}")
    else:
        print("  Feature audit: no highly skewed columns found.")

    print(f"  Feature audit: {len(feature_cols)} original → "
          f"{len(dropped_cols)} dropped → "
          f"{len(log_transform_cols)} log-transformed → "
          f"{len(surviving_cols)} surviving features\n")

    audit_report = {
        "original_count": len(feature_cols),
        "dropped_cols": dropped_cols,
        "dropped_variances": {c: float(variances[c]) for c in dropped_cols},
        "log_transform_cols": log_transform_cols,
        "log_transform_skewness": {c: float(skewness[c]) for c in log_transform_cols},
        "surviving_count": len(surviving_cols),
    }

    return surviving_cols, log_transform_cols, audit_report


def apply_feature_transforms(
    df: pd.DataFrame,
    surviving_cols: list,
    log_transform_cols: list,
) -> np.ndarray:
    """
    Apply the audit-determined transforms to any DataFrame:
    1. Select only surviving columns.
    2. log1p-transform the identified columns.
    Returns a float32 numpy array.
    """
    X = df[surviving_cols].copy()
    for col in log_transform_cols:
        X[col] = np.log1p(X[col].clip(lower=0))
    return X.values.astype(np.float32)


def compute_clip_bounds(
    X: np.ndarray,
    lower_pct: float = CLIP_LOWER_PCT,
    upper_pct: float = CLIP_UPPER_PCT,
):
    """
    Compute per-column clip bounds from reference data (Monday normal traffic).
    Returns two 1D arrays (lower, upper) of length n_features.
    """
    lower = np.percentile(X, lower_pct, axis=0).astype(np.float32)
    upper = np.percentile(X, upper_pct, axis=0).astype(np.float32)
    n_clipped_cols = int((upper > lower).sum())
    print(f"  Outlier clipping: bounds computed on Monday "
          f"(p{lower_pct:g}–p{upper_pct:g}), {n_clipped_cols}/{len(lower)} columns have non-degenerate ranges.")
    return lower, upper


def apply_clip(X: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Clip each column of X to the provided per-column bounds.
    """
    return np.clip(X, lower, upper).astype(np.float32)


def preprocess_cicids2017(
    df_monday: pd.DataFrame,
    df_attacks: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_state: int = 42,
):
    feature_cols = get_feature_cols(df_monday)

    surviving_cols, log_transform_cols, audit_report = audit_features(
        df_monday, feature_cols,
    )

    df_attacks = df_attacks.copy()
    df_attacks["is_attack"] = (df_attacks[LABEL_COL] != BENIGN_LABEL).astype(int)

    X_monday = apply_feature_transforms(df_monday, surviving_cols, log_transform_cols)

    X_attack_full = apply_feature_transforms(df_attacks, surviving_cols, log_transform_cols)
    y_attack_full = df_attacks["is_attack"].values
    labels_attack_full = df_attacks[LABEL_COL].values.astype(str)

    clip_lower, clip_upper = compute_clip_bounds(X_monday)
    X_monday = apply_clip(X_monday, clip_lower, clip_upper)
    X_attack_full = apply_clip(X_attack_full, clip_lower, clip_upper)

    (
        X_attack_train, X_attack_test,
        y_attack_train, y_attack_test,
        labels_attack_train, labels_attack_test,
    ) = train_test_split(
        X_attack_full, y_attack_full, labels_attack_full,
        test_size=test_size,
        stratify=y_attack_full,
        random_state=random_state,
    )

    (
        X_attack_train, X_attack_val,
        y_attack_train, y_attack_val,
        labels_attack_train, labels_attack_val,
    ) = train_test_split(
        X_attack_train, y_attack_train, labels_attack_train,
        test_size=val_size,
        stratify=y_attack_train,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_monday_scaled = scaler.fit_transform(X_monday)

    X_val_scaled  = scaler.transform(X_attack_val)
    X_test_scaled = scaler.transform(X_attack_test)

    print(f"  Post-clip/scale Monday range: "
          f"min={X_monday_scaled.min():+.2f}, max={X_monday_scaled.max():+.2f}")

    input_dim = X_monday_scaled.shape[1]

    X_train_tensor = torch.tensor(X_monday_scaled, dtype=torch.float32)
    X_val_tensor   = torch.tensor(X_val_scaled,    dtype=torch.float32)
    y_val_tensor   = torch.tensor(y_attack_val,    dtype=torch.int64)
    X_test_tensor  = torch.tensor(X_test_scaled,   dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_attack_test,   dtype=torch.int64)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor),
        batch_size=512, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor),
        batch_size=2048, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=2048, shuffle=False,
    )

    n_train_normal  = len(X_monday)
    n_val_total     = len(y_attack_val)
    n_test_total    = len(y_attack_test)

    print(f"Dataset split sizes | vae_train={n_train_normal:,}, "
          f"val={n_val_total:,}, test={n_test_total:,}")
    print(f"Input dimension: {input_dim}")

    feature_meta = {
        "surviving_cols": surviving_cols,
        "log_transform_cols": log_transform_cols,
        "audit_report": audit_report,
        "clip_lower": clip_lower.tolist(),
        "clip_upper": clip_upper.tolist(),
        "clip_lower_pct": CLIP_LOWER_PCT,
        "clip_upper_pct": CLIP_UPPER_PCT,
    }

    split_info = {
        "val_labels_raw": labels_attack_val,
        "test_labels_raw": labels_attack_test,
    }

    return train_loader, val_loader, test_loader, input_dim, scaler, feature_cols, feature_meta, split_info