import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader, TensorDataset

from data_utils import (
    load_cicids2017_raw,
    apply_feature_transforms,
    apply_clip,
    LABEL_COL,
    BENIGN_LABEL,
)
from vae_model import VAE
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


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    input_dim  = int(checkpoint["input_dim"])
    latent_dim = int(checkpoint["latent_dim"])
    hidden_dims = _infer_hidden_dims_from_state_dict(state_dict)
    model = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, checkpoint


def build_loader(df_attacks, scaler, feature_meta, max_rows, random_state, batch_size=2048):
    surviving_cols     = feature_meta["surviving_cols"]
    log_transform_cols = feature_meta["log_transform_cols"]
    clip_lower         = np.asarray(feature_meta["clip_lower"], dtype=np.float32)
    clip_upper         = np.asarray(feature_meta["clip_upper"], dtype=np.float32)

    if max_rows and len(df_attacks) > max_rows:
        print(f"Subsampling {max_rows:,} rows from {len(df_attacks):,} attack-day rows...")
        df_attacks = df_attacks.sample(n=max_rows, random_state=random_state)

    y = (df_attacks[LABEL_COL] != BENIGN_LABEL).astype(int).values
    n_benign = int((y == 0).sum())
    n_attack = int((y == 1).sum())
    print(f"Test set: {n_benign:,} benign, {n_attack:,} attack")

    X_np = apply_feature_transforms(df_attacks, surviving_cols, log_transform_cols)
    X_np = apply_clip(X_np, clip_lower, clip_upper)
    X_np = scaler.transform(X_np).astype(np.float32)

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=False)
    return loader


def get_reconstruction_errors(model, loader, device):
    errors = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            x_recon, _, _ = model(x)
            err = ((x_recon - x) ** 2).mean(dim=1).cpu().numpy()
            errors.extend(err)
            labels.extend(y.numpy())
    return np.array(errors), np.array(labels)


def plot_distribution(errors, labels, threshold=None, save_path=None):
    normal = errors[labels == 0]
    attack = errors[labels == 1]

    e_min = errors[errors > 0].min()
    e_max = errors.max()
    bins = np.logspace(np.log10(e_min), np.log10(e_max), 80)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(normal, bins=bins, alpha=0.6, density=True, label=f"Benign (n={len(normal):,})")
    ax.hist(attack, bins=bins, alpha=0.6, density=True, label=f"Attack (n={len(attack):,})")
    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", linewidth=2,
                   label=f"Threshold ({threshold:.4f})")
    ax.set_xscale("log")
    ax.set_xlabel("Reconstruction Error (log scale)")
    ax.set_ylabel("Density")
    ax.set_title("VAE Reconstruction Error Distribution\nBenign vs Attack - CICIDS2017")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot VAE reconstruction error distribution on CICIDS2017 attack-day data."
    )
    parser.add_argument("--data-dir",      default="data/cicids2017")
    parser.add_argument("--vae-run-dir",   default="outputs/default")
    parser.add_argument("--threshold-path", default=None,
                        help="Path to threshold.json. Defaults to <vae-run-dir>/threshold.json")
    parser.add_argument("--max-rows",      type=int, default=300000,
                        help="Subsample attack-day rows to keep runtime manageable (default 300k)")
    parser.add_argument("--random-state",  type=int, default=42)
    parser.add_argument("--save-path",     default=None,
                        help="If set, save the figure to this path (e.g. figures/recon_dist.png)")
    parser.add_argument("--device",        default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from pathlib import Path
    vae_run_path  = Path(args.vae_run_dir)
    model_path    = vae_run_path / "model.pt"
    threshold_path = Path(args.threshold_path) if args.threshold_path else vae_run_path / "threshold.json"

    model, checkpoint = load_model(model_path, device)
    scaler       = checkpoint["scaler"]
    feature_meta = checkpoint["feature_meta"]

    print("Loading CICIDS2017 attack-day data...")
    _df_monday, df_attacks = load_cicids2017_raw(args.data_dir)

    loader = build_loader(
        df_attacks, scaler, feature_meta,
        max_rows=args.max_rows,
        random_state=args.random_state,
    )

    print("Computing reconstruction errors...")
    errors, labels = get_reconstruction_errors(model, loader, device)

    threshold = None
    if threshold_path.exists():
        payload   = load_threshold(threshold_path)
        threshold = float(payload["value"])
        print(f"Loaded threshold: {threshold:.6f} ({payload.get('method')})")
    else:
        print("No threshold.json found - plotting without threshold line.")

    plot_distribution(errors, labels, threshold=threshold, save_path=args.save_path)