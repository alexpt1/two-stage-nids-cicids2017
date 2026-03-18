import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


ATTACK_CATEGORY_MAP = {
    "normal": "normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "worm": "DoS", "mailbomb": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpattack": "R2L", "snmpguess": "R2L",
    "httptunnel": "R2L", "xlock": "R2L", "xsnoop": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}


def map_attack_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["attack_category"] = df["label"].map(ATTACK_CATEGORY_MAP).fillna("unknown")
    return df


def load_stage2_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    preprocessor,
    val_size: float = 0.2,
    random_state: int = 42,
):
    feature_cols = list(range(41))

    df_train = map_attack_categories(df_train.copy())
    df_test  = map_attack_categories(df_test.copy())

    df_train_attacks = df_train[df_train["attack_category"] != "normal"].reset_index(drop=True)
    df_test_attacks  = df_test[df_test["attack_category"]  != "normal"].reset_index(drop=True)

    print(f"Attack rows — train: {len(df_train_attacks)}, test: {len(df_test_attacks)}")
    print("Train category distribution:\n",
          df_train_attacks["attack_category"].value_counts().to_string())
    print("Test category distribution:\n",
          df_test_attacks["attack_category"].value_counts().to_string())

    X_train_proc = preprocessor.transform(df_train_attacks[feature_cols])
    X_test_proc  = preprocessor.transform(df_test_attacks[feature_cols])

    X_train_np = X_train_proc.toarray() if hasattr(X_train_proc, "toarray") else X_train_proc
    X_test_np  = X_test_proc.toarray()  if hasattr(X_test_proc,  "toarray") else X_test_proc

    le = LabelEncoder()
    y_train_raw = df_train_attacks["attack_category"].values
    y_test_raw  = df_test_attacks["attack_category"].values

    le.fit(y_train_raw)
    y_train = le.transform(y_train_raw)
    y_test = np.array([
        le.transform([c])[0] if c in le.classes_ else -1
        for c in y_test_raw
    ])

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_np, y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=random_state,
    )

    print(f"\nStage 2 split sizes | train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    return X_train, y_train, X_val, y_val, X_test_np, y_test, le


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size: int = 512):
    def to_loader(X, y, shuffle):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

    return (
        to_loader(X_train, y_train, shuffle=True),
        to_loader(X_val,   y_val,   shuffle=False),
        to_loader(X_test,  y_test,  shuffle=False),
    )