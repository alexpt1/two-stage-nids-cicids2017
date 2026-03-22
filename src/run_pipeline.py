import json
import pickle
import argparse
from pathlib import Path

import time
import numpy as np
import sklearn
import torch
from torch.serialization import add_safe_globals

from data_utils import load_nsl_kdd_raw
from data_utils_stage2 import map_attack_categories
from vae_model import VAE
from thresholding import load_threshold


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


def run_pipeline(
    data_dir: str = "data/nsl_kdd",
    vae_run_dir: str = "outputs/default",
    clf_run_dir: str = "outputs/clf_default",
    confidence_threshold: float = 0.6,
    random_state: int = 42,
    device: str | None = None,
    n_samples: int = 2000,
    recalibrate: bool = False,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    add_safe_globals([
        sklearn.compose.ColumnTransformer,
        sklearn.preprocessing.OneHotEncoder,
        sklearn.preprocessing.StandardScaler,
    ])

    vae_run_path   = Path(vae_run_dir)
    vae_model_path = vae_run_path / "model.pt"
    threshold_path = vae_run_path / "threshold.json"

    print(f"Loading VAE from      : {vae_model_path}")
    checkpoint   = torch.load(vae_model_path, map_location=device, weights_only=False)
    preprocessor = checkpoint["preprocessor"]
    state_dict   = checkpoint["model_state_dict"]

    input_dim   = int(checkpoint["input_dim"])
    latent_dim  = int(checkpoint["latent_dim"])
    hidden_dims = _infer_hidden_dims(state_dict)

    vae = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)
    vae.load_state_dict(state_dict)
    vae.to(device).eval()

    if recalibrate:
        print("Recalibrating threshold on recent traffic window...")
        from recalibrate_threshold import recalibrate_threshold as do_recalibrate
        do_recalibrate(
            data_dir=data_dir,
            vae_run_dir=vae_run_dir,
            method="percentile",
            percentile=95.0,
            window_size=5000,
            device=device,
        )

    threshold_payload = load_threshold(threshold_path)
    threshold = float(threshold_payload["value"])
    print(f"Anomaly threshold     : {threshold:.6f} ({threshold_payload.get('method')})")

    clf_run_path = Path(clf_run_dir)
    with (clf_run_path / "model.pkl").open("rb") as f:
        payload = pickle.load(f)
    clf = payload["clf"] if isinstance(payload, dict) else payload
    with (clf_run_path / "label_encoder.pkl").open("rb") as f:
        le = pickle.load(f)
    print(f"Classifier classes    : {list(le.classes_)}")

    print("\nLoading NSL-KDD test data...")
    df_train, df_test = load_nsl_kdd_raw(data_dir)
    df_test = map_attack_categories(df_test)
    feature_cols = list(range(41))

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(df_test), size=min(n_samples, len(df_test)), replace=False)
    df_sample = df_test.iloc[idx].reset_index(drop=True)

    X_raw = df_sample[feature_cols]
    true_categories = df_sample["attack_category"].values

    X_proc = preprocessor.transform(X_raw)
    X_np   = X_proc.toarray() if hasattr(X_proc, "toarray") else X_proc
    X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)

    t_start = time.perf_counter()
    with torch.no_grad():
        x_recon, mu, logvar = vae(X_tensor)
        recon_errors = ((x_recon - X_tensor) ** 2).mean(dim=1).cpu().numpy()
    t_vae = time.perf_counter() - t_start

    anomaly_mask = recon_errors > threshold
    n_normal  = int((~anomaly_mask).sum())
    n_anomaly = int(anomaly_mask.sum())
    print(f"\nStage 1 results | normal={n_normal}, anomaly={n_anomaly} "
          f"(alert rate={n_anomaly/len(anomaly_mask):.3f})")

    verdicts = np.array(["normal"] * len(df_sample), dtype=object)

    t_clf = 0.0
    if n_anomaly > 0:
        X_anomaly = X_np[anomaly_mask]
        t_clf_start = time.perf_counter()
        proba     = clf.predict_proba(X_anomaly)
        t_clf = time.perf_counter() - t_clf_start
        pred_idx  = np.argmax(proba, axis=1)
        confidence = proba[np.arange(len(proba)), pred_idx]

        pred_labels = le.inverse_transform(pred_idx).astype(object)
        pred_labels[confidence < confidence_threshold] = "novel_anomaly"

        verdicts[anomaly_mask] = pred_labels

    unique, counts = np.unique(verdicts, return_counts=True)
    print("\nFinal verdict distribution:")
    for label, count in zip(unique, counts):
        print(f"  {label:20s}: {count:5d}  ({count/len(verdicts)*100:.1f}%)")

    y_true_binary = (true_categories != "normal").astype(int)
    y_pred_binary = (verdicts != "normal").astype(int)
    correct = (y_true_binary == y_pred_binary).sum()
    print(f"\nBinary accuracy (normal vs attack): {correct/len(verdicts):.4f}")

    n_total = len(df_sample)
    t_total = t_vae + t_clf
    vae_us  = (t_vae / n_total) * 1e6
    clf_us  = (t_clf / n_anomaly) * 1e6 if n_anomaly > 0 else 0.0
    total_us = (t_total / n_total) * 1e6

    print(f"\nLatency (per sample):")
    print(f"  VAE inference      : {vae_us:.2f} us/sample")
    print(f"  RF inference       : {clf_us:.2f} us/sample (anomalies only)")
    print(f"  Total pipeline     : {total_us:.2f} us/sample")
    print(f"  Throughput         : {1e6/total_us:,.0f} samples/sec")

    return verdicts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full two-stage NIDS pipeline.")
    parser.add_argument("--data-dir",             default="data/nsl_kdd")
    parser.add_argument("--vae-run-dir",          default="outputs/default")
    parser.add_argument("--clf-run-dir",          default="outputs/clf_default")
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--device",               default=None)
    parser.add_argument("--n-samples",            type=int,   default=2000)
    parser.add_argument("--random-state",         type=int,   default=42)
    parser.add_argument("--recalibrate", action="store_true", help="Recalibrate threshold before running pipeline")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        vae_run_dir=args.vae_run_dir,
        clf_run_dir=args.clf_run_dir,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        n_samples=args.n_samples,
        random_state=args.random_state,
        recalibrate=args.recalibrate,
    )