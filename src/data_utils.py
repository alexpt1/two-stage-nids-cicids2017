import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_nsl_kdd_raw(data_dir: str):
    """
    Load NSL-KDD train and test text files.
    data_dir should contain KDDTrain+.txt and KDDTest+.txt
    Returns: (df_train, df_test)
    """
    train_path = os.path.join(data_dir, "KDDTrain+.txt")
    test_path = os.path.join(data_dir, "KDDTest+.txt")

    col_names = list(range(43))

    try:
        df_train = pd.read_csv(train_path, header=None, names=col_names)
        df_test = pd.read_csv(test_path, header=None, names=col_names)
    except Exception:
        df_train = pd.read_csv(train_path, header=None, names=col_names, sep=",")
        df_test = pd.read_csv(test_path, header=None, names=col_names, sep=",")

    df_train = df_train.rename(columns={41: "label", 42: "difficulty"})
    df_test = df_test.rename(columns={41: "label", 42: "difficulty"})
    return df_train, df_test


def preprocess_nsl_kdd(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    reshuffle_all: bool = False,
):
    """
    Preprocess NSL-KDD:
    - split normal vs attack
    - train/val/test (official split by default)
    - one-hot encode categoricals
    - standardise numerical features
    Returns: (train_loader, val_loader, test_loader, input_dim, preprocessor)
    """
    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train["is_attack"] = (df_train["label"] != "normal").astype(int)
    df_test["is_attack"] = (df_test["label"] != "normal").astype(int)

    feature_cols = list(range(41))
    categorical_idxs = [1, 2, 3]
    numeric_idxs = [i for i in feature_cols if i not in categorical_idxs]

    if reshuffle_all:
        df_all = pd.concat([df_train, df_test], ignore_index=True)
        X = df_all[feature_cols]
        y = df_all["is_attack"].values

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=val_size,
            stratify=y_train_full,
            random_state=random_state,
        )
    else:
        X_train_full = df_train[feature_cols]
        y_train_full = df_train["is_attack"].values
        X_test = df_test[feature_cols]
        y_test = df_test["is_attack"].values

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=val_size,
            stratify=y_train_full,
            random_state=random_state,
        )

    X_train_normal = X_train[y_train == 0]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_idxs),
            ("num", StandardScaler(), numeric_idxs),
        ]
    )

    X_train_normal_processed = preprocessor.fit_transform(X_train_normal)

    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    X_train_normal_np = (
        X_train_normal_processed.toarray()
        if hasattr(X_train_normal_processed, "toarray")
        else X_train_normal_processed
    )
    X_val_np = X_val_processed.toarray() if hasattr(X_val_processed, "toarray") else X_val_processed
    X_test_np = X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed

    input_dim = X_train_normal_np.shape[1]

    X_train_tensor = torch.tensor(X_train_normal_np, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.int64)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)

    train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=1024, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=1024, shuffle=False)

    train_total = len(X_train)
    val_total = len(X_val)
    test_total = len(X_test)
    train_normals = len(X_train_normal)
    print(
        f"Dataset split sizes | train_total={train_total}, "
        f"train_normals={train_normals}, val={val_total}, test={test_total}"
    )

    return train_loader, val_loader, test_loader, input_dim, preprocessor
