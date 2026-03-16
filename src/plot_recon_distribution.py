import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from data_utils import load_nsl_kdd_raw
from vae_model import VAE
import sklearn
from torch.serialization import add_safe_globals
from thresholding import load_threshold


def _infer_hidden_dims_from_state_dict(state_dict):
    hidden_dims = []
    layer_idx = 0
    while True:
        key = f"encoder.{layer_idx}.weight"
        if key not in state_dict:
            break
        hidden_dims.append(state_dict[key].shape[0])
        layer_idx += 2
    if not hidden_dims:
        raise ValueError("Could not infer hidden_dims from checkpoint state_dict.")
    return tuple(hidden_dims)


def _build_test_loader_with_saved_preprocessor(
    df_train,
    df_test,
    preprocessor,
    reshuffle_all: bool = False,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train["is_attack"] = (df_train["label"] != "normal").astype(int)
    df_test["is_attack"] = (df_test["label"] != "normal").astype(int)
    feature_cols = list(range(41))

    if reshuffle_all:
        from pandas import concat

        df_all = concat([df_train, df_test], ignore_index=True)
        X = df_all[feature_cols]
        y = df_all["is_attack"].values
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        _X_train, _X_val, _y_train, _y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=val_size,
            stratify=y_train_full,
            random_state=random_state,
        )
    else:
        X_test = df_test[feature_cols]
        y_test = df_test["is_attack"].values

    X_test_processed = preprocessor.transform(X_test)
    X_test_np = X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=1024, shuffle=False)
    return test_loader, X_test_tensor.shape[1]


def load_model(model_path, device):
    
    add_safe_globals([
        sklearn.compose.ColumnTransformer,
        sklearn.preprocessing.OneHotEncoder,
        sklearn.preprocessing.StandardScaler
    ])

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    state_dict = checkpoint["model_state_dict"]
    input_dim = int(checkpoint.get("input_dim", state_dict["decoder.4.bias"].shape[0]))
    latent_dim = int(checkpoint.get("latent_dim", state_dict["fc_mu.bias"].shape[0]))
    hidden_dims = _infer_hidden_dims_from_state_dict(state_dict)
    model = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def get_reconstruction_errors(model, loader, device):
    errors = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            x_recon, mu, logvar = model(x)
            err = ((x_recon - x) ** 2).mean(dim=1).cpu().numpy()
            errors.extend(err)
            labels.extend(y.numpy())

    return np.array(errors), np.array(labels)


def plot_distribution(errors, labels, threshold=None):
    normal = errors[labels == 0]
    attack = errors[labels == 1]

    bins = np.logspace(np.log10(errors.min()), np.log10(errors.max()), 80)

    fig = plt.figure(figsize=(10, 6))
    fig.canvas.manager.set_window_title("Reconstruction Error Distribution - NSL-KDD")

    plt.hist(normal, bins=bins, alpha=0.6, density=True, label="Normal")
    plt.hist(attack, bins=bins, alpha=0.6, density=True, label="Attack")
    if threshold is not None:
        plt.axvline(threshold, color="red", linestyle="--", linewidth=2, label="Threshold")
    plt.xscale("log")
    plt.xlabel("Reconstruction Error (log scale)")
    plt.ylabel("Density")
    plt.title("Distribution of Reconstruction Error\nNormal vs Attack (Log-Scaled Bins)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot reconstruction error distribution with optional persisted threshold."
    )
    parser.add_argument("--data-dir", default="../data/nsl_kdd")
    parser.add_argument("--model-path", default="../models/vae_nsl_kdd.pt")
    parser.add_argument("--threshold-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reshuffle-all", action="store_true")
    args = parser.parse_args()

    device = (
        args.device
        if args.device is not None
        else ("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    )
    print(f"Using : {device}")

    df_train, df_test = load_nsl_kdd_raw(args.data_dir)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    preprocessor = checkpoint.get("preprocessor")
    if preprocessor is None:
        raise ValueError(
            "Checkpoint does not contain 'preprocessor'. "
            "Use a checkpoint produced by the current training script."
        )
    test_loader, transformed_input_dim = _build_test_loader_with_saved_preprocessor(
        df_train,
        df_test,
        preprocessor=preprocessor,
        reshuffle_all=args.reshuffle_all,
    )

    model = load_model(args.model_path, device)
    model_input_dim = int(checkpoint.get("input_dim", transformed_input_dim))
    if transformed_input_dim != model_input_dim:
        raise ValueError(
            f"Feature dimension mismatch after preprocessing: got {transformed_input_dim}, "
            f"but checkpoint expects input_dim={model_input_dim}."
        )

    errors, labels = get_reconstruction_errors(model, test_loader, device)

    threshold = None
    if args.threshold_path:
        threshold_payload = load_threshold(args.threshold_path)
        threshold = float(threshold_payload["value"])
        print(f"Loaded threshold from {args.threshold_path}: {threshold:.6f}")

    plot_distribution(errors, labels, threshold=threshold)
