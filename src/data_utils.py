import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_nsl_kdd_raw(data_dir: str) -> pd.DataFrame:
    """
    Load NSL-KDD train and test text files into a single pandas DataFrame.
    data_dir should contain KDDTrain+.txt and KDDTest+.txt
    """
    train_path = os.path.join(data_dir, "KDDTrain+.txt")
    test_path = os.path.join(data_dir, "KDDTest+.txt")

    # NSL-KDD is space-separated (or comma in some mirrors). Try first, then fall back.
    col_names = list(range(43))  # 41 features + label + difficulty

    try:
        df_train = pd.read_csv(train_path, header=None, names=col_names)
        df_test = pd.read_csv(test_path, header=None, names=col_names)
    except Exception:
        df_train = pd.read_csv(train_path, header=None, names=col_names, sep=",")
        df_test = pd.read_csv(test_path, header=None, names=col_names, sep=",")

    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df.rename(columns={41: "label", 42: "difficulty"})
    return df


def preprocess_nsl_kdd(df: pd.DataFrame,
                       test_size: float = 0.2,
                       val_size: float = 0.1,
                       random_state: int = 42):
    """
    Preprocess NSL-KDD:
    - split normal vs attack
    - train/val/test
    - one-hot encode categoricals
    - standardise numerical features
    Returns: (train_loader, val_loader, test_loader, input_dim, threshold_info)
    """
    # Shallow copy
    df = df.copy()

    # Binary label: 0 = normal, 1 = attack
    df["is_attack"] = (df["label"] != "normal").astype(int)

    # Separate features and label
    feature_cols = list(range(41))  # 0..40
    X = df[feature_cols]
    y = df["is_attack"].values

    # Identify categorical columns by NSL-KDD spec
    categorical_idxs = [1, 2, 3]  # protocol_type, service, flag
    numeric_idxs = [i for i in feature_cols if i not in categorical_idxs]

    # Split train/test for evaluation (stratified)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Within train, create validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, stratify=y_train_full, random_state=random_state
    )

    # For VAE training, we ONLY use normal samples (label 0) from train set
    X_train_normal = X_train[y_train == 0]

    # Column transformer: one-hot for categoricals, standard scaling for numerics
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_idxs),
            ("num", StandardScaler(), numeric_idxs),
        ]
    )

    # Fit on training normal data only
    X_train_normal_processed = preprocessor.fit_transform(X_train_normal)

    # Transform other splits
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Convert to dense numpy arrays (for small-medium datasets, OK)
    X_train_normal_np = X_train_normal_processed.toarray() if hasattr(X_train_normal_processed, "toarray") else X_train_normal_processed
    X_val_np = X_val_processed.toarray() if hasattr(X_val_processed, "toarray") else X_val_processed
    X_test_np = X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed

    input_dim = X_train_normal_np.shape[1]

    # Torch tensors
    X_train_tensor = torch.tensor(X_train_normal_np, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.int64)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=1024, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=1024, shuffle=False)

    return train_loader, val_loader, test_loader, input_dim, preprocessor