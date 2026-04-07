import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_utils import apply_feature_transforms, apply_clip, LABEL_COL, BENIGN_LABEL


ATTACK_CATEGORY_MAP = {
    "BENIGN": "benign",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DoS",
    "PortScan": "Probe",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack \ufffd Brute Force": "WebAttack",
    "Web Attack \ufffd XSS": "WebAttack",
    "Web Attack \ufffd Sql Injection": "WebAttack",
    "Bot": "Bot",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}

RARE_CLASSES = {"Infiltration", "Heartbleed"}


def map_attack_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["attack_category"] = df[LABEL_COL].map(ATTACK_CATEGORY_MAP).fillna("unknown")
    return df


def load_stage2_data(
    df_attacks: pd.DataFrame,
    scaler,
    feature_meta: dict,
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_state: int = 42,
):
    surviving_cols     = feature_meta["surviving_cols"]
    log_transform_cols = feature_meta["log_transform_cols"]
    clip_lower         = np.asarray(feature_meta["clip_lower"], dtype=np.float32)
    clip_upper         = np.asarray(feature_meta["clip_upper"], dtype=np.float32)

    df = map_attack_categories(df_attacks)

    df_attack_only = df[df["attack_category"] != "benign"].reset_index(drop=True)

    unknown_count = int((df_attack_only["attack_category"] == "unknown").sum())
    if unknown_count > 0:
        print(f"WARNING: {unknown_count} rows have attack_category='unknown' (label not in map). Dropping.")
        df_attack_only = df_attack_only[df_attack_only["attack_category"] != "unknown"].reset_index(drop=True)

    print(f"Attack rows for Stage 2: {len(df_attack_only):,}")
    print("Category distribution:")
    print(df_attack_only["attack_category"].value_counts().to_string())

    X_np = apply_feature_transforms(df_attack_only, surviving_cols, log_transform_cols)
    X_np = apply_clip(X_np, clip_lower, clip_upper)
    X_np = scaler.transform(X_np).astype(np.float32)

    le = LabelEncoder()
    y_raw = df_attack_only["attack_category"].values
    le.fit(y_raw)
    y = le.transform(y_raw)

    class_counts = pd.Series(y_raw).value_counts()
    too_rare_for_stratify = class_counts[class_counts < 3].index.tolist()
    if too_rare_for_stratify:
        print(f"Note: classes with <3 samples cannot be stratified safely: {too_rare_for_stratify}")
        print("Falling back to non-stratified split.")
        stratify_outer = None
    else:
        stratify_outer = y

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_np, y,
        test_size=test_size,
        stratify=stratify_outer,
        random_state=random_state,
    )

    stratify_inner = y_trainval if stratify_outer is not None else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size,
        stratify=stratify_inner,
        random_state=random_state,
    )

    print(f"\nStage 2 split sizes | train={len(y_train):,}, val={len(y_val):,}, test={len(y_test):,}")
    print(f"Classes: {list(le.classes_)}")

    rare_present = [c for c in le.classes_ if c in RARE_CLASSES]
    if rare_present:
        print(f"Note: rare classes kept (will likely fall below confidence threshold "
              f"and route to novel_anomaly): {rare_present}")

    return X_train, y_train, X_val, y_val, X_test, y_test, le