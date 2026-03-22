import json
import argparse
import numpy as np
import torch
import sklearn
from pathlib import Path
from datetime import datetime, timezone
from torch.serialization import add_safe_globals

from data_utils import load_nsl_kdd_raw
from vae_model import VAE
from thresholding import calibrate_threshold, save_threshold, load_threshold


def _infer_hidden_dims(state_dict):
    hidden_dims = []
    layer_idx = 0
    while True:
        key = f"encoder.{layer_idx}.weight"
        if key not in state_dict:
            break
        hidden_dims.append(state_dict[key].shape[0])
        layer_idx += 2
    return tuple(hidden_dims)


def recalibrate_threshold(
    data_dir: str = "data/nsl_kdd",
    vae_run_dir: str = "outputs/default",
    method: str = "percentile",
    percentile: float = 95.0,
    k: float = 3.0,
    window_size: int = 5000,
    normal_only: bool = True,
    random_state: int = 42,
    device: str = "cpu",
):
    add_safe_globals([
        sklearn.compose.ColumnTransformer,
        sklearn.preprocessing.OneHotEncoder,
        sklearn.preprocessing.StandardScaler,
    ])

    vae_run_path   = Path(vae_run_dir)
    vae_model_path = vae_run_path / "model.pt"
    threshold_path = vae_run_path / "threshold.json"

    checkpoint   = torch.load(vae_model_path, map_location=device, weights_only=False)
    preprocessor = checkpoint["preprocessor"]
    state_dict   = checkpoint["model_state_dict"]

    input_dim   = int(checkpoint["input_dim"])
    latent_dim  = int(checkpoint["latent_dim"])
    hidden_dims = _infer_hidden_dims(state_dict)

    vae = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)
    vae.load_state_dict(state_dict)
    vae.to(device).eval()

    print("Loading calibration data...")
    df_train, df_test = load_nsl_kdd_raw(data_dir)
    feature_cols = list(range(41))

    if normal_only:
        df_window = df_train[df_train["label"] == "normal"].sample(
            n=min(window_size, (df_train["label"] == "normal").sum()),
            random_state=random_state,
        )
        print(f"Using {len(df_window)} normal training samples for recalibration")
    else:
        df_window = df_train.sample(
            n=min(window_size, len(df_train)),
            random_state=random_state,
        )
        print(f"Using {len(df_window)} mixed training samples for recalibration")

    X_proc = preprocessor.transform(df_window[feature_cols])
    X_np   = X_proc.toarray() if hasattr(X_proc, "toarray") else X_proc
    X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)

    with torch.no_grad():
        vae_out = vae(X_tensor)
        x_recon = vae_out[0]
        recon_errors = ((x_recon - X_tensor) ** 2).mean(dim=1).cpu().numpy()

    if threshold_path.exists():
        old_payload = load_threshold(threshold_path)
        old_threshold = float(old_payload["value"])
    else:
        old_threshold = None

    new_threshold = calibrate_threshold(
        recon_errors,
        method=method,
        percentile=percentile,
        k=k,
    )

    payload = {
        "method": method,
        "percentile": percentile,
        "k": k,
        "value": new_threshold,
        "window_size": window_size,
        "normal_only": normal_only,
        "previous_value": old_threshold,
        "drift_delta": float(new_threshold - old_threshold) if old_threshold is not None else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recalibrated": True,
    }

    save_threshold(threshold_path, payload)

    print(f"Previous threshold : {old_threshold:.6f}" if old_threshold else "Previous threshold : None")
    print(f"New threshold      : {new_threshold:.6f}")
    if old_threshold is not None:
        delta = new_threshold - old_threshold
        pct_change = (delta / old_threshold) * 100
        print(f"Drift delta        : {delta:+.6f}  ({pct_change:+.2f}%)")
    print(f"Method             : {method}")
    print(f"Saved to           : {threshold_path}")

    return new_threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recalibrate VAE anomaly threshold on new traffic window.")
    parser.add_argument("--data-dir",     default="data/nsl_kdd")
    parser.add_argument("--vae-run-dir",  default="outputs/default")
    parser.add_argument("--method",       default="percentile", choices=["percentile", "sigma"])
    parser.add_argument("--percentile",   type=float, default=95.0)
    parser.add_argument("--k",            type=float, default=3.0)
    parser.add_argument("--window-size",  type=int,   default=5000)
    parser.add_argument("--normal-only",  action="store_true", default=True)
    parser.add_argument("--random-state", type=int,   default=42)
    parser.add_argument("--device",       default="cpu")
    args = parser.parse_args()

    recalibrate_threshold(
        data_dir=args.data_dir,
        vae_run_dir=args.vae_run_dir,
        method=args.method,
        percentile=args.percentile,
        k=args.k,
        window_size=args.window_size,
        normal_only=args.normal_only,
        random_state=args.random_state,
        device=args.device,
    )