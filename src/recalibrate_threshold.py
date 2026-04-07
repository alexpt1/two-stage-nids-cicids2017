import json
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timezone

from data_utils import load_cicids2017_raw, apply_feature_transforms, apply_clip
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
    data_dir: str = "data/cicids2017",
    vae_run_dir: str = "outputs/default",
    method: str = "percentile",
    percentile: float = 95.0,
    k: float = 3.0,
    window_size: int = 5000,
    random_state: int = 42,
    device: str = "cpu",
):
    vae_run_path   = Path(vae_run_dir)
    vae_model_path = vae_run_path / "model.pt"
    threshold_path = vae_run_path / "threshold.json"

    checkpoint   = torch.load(vae_model_path, map_location=device, weights_only=False)
    scaler       = checkpoint["scaler"]
    feature_meta = checkpoint["feature_meta"]
    state_dict   = checkpoint["model_state_dict"]

    surviving_cols     = feature_meta["surviving_cols"]
    log_transform_cols = feature_meta["log_transform_cols"]
    clip_lower         = np.asarray(feature_meta["clip_lower"], dtype=np.float32)
    clip_upper         = np.asarray(feature_meta["clip_upper"], dtype=np.float32)

    input_dim   = int(checkpoint["input_dim"])
    latent_dim  = int(checkpoint["latent_dim"])
    hidden_dims = _infer_hidden_dims(state_dict)

    vae = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)
    vae.load_state_dict(state_dict)
    vae.to(device).eval()

    print("Loading Monday normal traffic for recalibration...")
    df_monday, _ = load_cicids2017_raw(data_dir)

    n_sample = min(window_size, len(df_monday))
    df_window = df_monday.sample(n=n_sample, random_state=random_state)
    print(f"Using {len(df_window):,} Monday normal samples for recalibration")

    X_np = apply_feature_transforms(df_window, surviving_cols, log_transform_cols)
    X_np = apply_clip(X_np, clip_lower, clip_upper)
    X_np = scaler.transform(X_np).astype(np.float32)
    X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)

    with torch.no_grad():
        x_recon, _, _ = vae(X_tensor)
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
    parser = argparse.ArgumentParser(description="Recalibrate VAE anomaly threshold on Monday normal traffic.")
    parser.add_argument("--data-dir",     default="data/cicids2017")
    parser.add_argument("--vae-run-dir",  default="outputs/default")
    parser.add_argument("--method",       default="percentile", choices=["percentile", "sigma"])
    parser.add_argument("--percentile",   type=float, default=95.0)
    parser.add_argument("--k",            type=float, default=3.0)
    parser.add_argument("--window-size",  type=int,   default=5000)
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
        random_state=args.random_state,
        device=args.device,
    )