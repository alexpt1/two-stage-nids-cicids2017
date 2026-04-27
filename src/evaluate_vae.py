import json
import torch
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    confusion_matrix,
)
from pathlib import Path
import argparse
from datetime import datetime, timezone
from torch.utils.data import DataLoader, TensorDataset
import pickle

from data_utils import load_cicids2017_raw, preprocess_cicids2017
from vae_model import VAE
from thresholding import calibrate_threshold, save_threshold, load_threshold


def recall_at_fpr(fpr_curve, tpr_curve, target_fpr):
    idx = np.searchsorted(fpr_curve, target_fpr, side="right") - 1
    idx = max(0, min(idx, len(tpr_curve) - 1))
    return float(tpr_curve[idx])


def compute_anomaly_scores(x, x_recon, mu, logvar, scoring: str, beta: float = 1.0):
    recon = ((x_recon - x) ** 2).mean(dim=1)
    if scoring == "mse":
        return recon.cpu().numpy()
    kl_per_sample = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    return (recon + beta * kl_per_sample).cpu().numpy()


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


def evaluate_vae_cicids2017(
    data_dir: str = "data/cicids2017",
    run_dir: str = "outputs/default",
    model_path: str | None = None,
    threshold_path: str | None = None,
    reuse_threshold: bool = False,
    threshold_method: str = "roc_optimal",
    threshold_k: float = 3.0,
    threshold_percentile: float = 95,
    device: str | None = None,
    scoring: str = "mse",
):
    #Score the VAE on the test split, calibrate or reuse a threshold, and write metrics.json to run_dir.
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    resolved_run_dir   = Path(run_dir)
    resolved_model_path = Path(model_path) if model_path else resolved_run_dir / "model.pt"

    checkpoint   = torch.load(resolved_model_path, map_location=device, weights_only=False)
    state_dict   = checkpoint["model_state_dict"]
    scaler       = checkpoint["scaler"]
    feature_cols = checkpoint["feature_cols"]
    beta_score   = float(checkpoint.get("beta", 1.0))

    latent_dim  = int(checkpoint.get("latent_dim", state_dict["fc_mu.weight"].shape[0]))
    hidden_dims = _infer_hidden_dims_from_state_dict(state_dict)
    input_dim   = int(checkpoint["input_dim"])

    model = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    print("Loading CICIDS2017 data...")
    df_monday, df_attacks = load_cicids2017_raw(data_dir)

    _, val_loader, test_loader, _, _, _, _, _ = preprocess_cicids2017(
        df_monday,
        df_attacks,
        random_state=42,
    )

    val_errors = []
    val_labels = []
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            x_recon, mu, logvar = model(x_val)
            err = compute_anomaly_scores(x_val, x_recon, mu, logvar, scoring=scoring, beta=beta_score)
            val_errors.extend(err)
            val_labels.extend(y_val.numpy())

    val_errors = np.array(val_errors)
    val_labels = np.array(val_labels)

    resolved_threshold_path = (
        Path(threshold_path) if threshold_path else resolved_run_dir / "threshold.json"
    )

    if reuse_threshold and resolved_threshold_path.exists():
        threshold_payload = load_threshold(resolved_threshold_path)
        threshold = float(threshold_payload["value"])
        print(f"Loaded threshold: {threshold:.6f} ({threshold_payload.get('method')})")
    else:
        normal_val_errors = val_errors[val_labels == 0]
        threshold = calibrate_threshold(
            val_errors if threshold_method == "roc_optimal" else normal_val_errors,
            method=threshold_method,
            percentile=threshold_percentile,
            k=threshold_k,
            val_labels=val_labels if threshold_method == "roc_optimal" else None,
        )
        threshold_payload = {
            "method":      threshold_method,
            "percentile":  threshold_percentile,
            "k":           threshold_k,
            "value":       threshold,
            "created_at":  datetime.now(timezone.utc).isoformat(),
        }
        save_threshold(resolved_threshold_path, threshold_payload)
        print(f"Calibrated threshold ({threshold_method}): {threshold:.6f}")

    test_errors = []
    test_labels = []
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            x_recon, mu, logvar = model(x_test)
            err = compute_anomaly_scores(x_test, x_recon, mu, logvar, scoring=scoring, beta=beta_score)
            test_errors.extend(err)
            test_labels.extend(y_test.numpy())

    test_errors = np.array(test_errors)
    test_labels = np.array(test_labels)
    y_pred      = (test_errors > threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, y_pred, average="binary", pos_label=1
    )
    fpr_curve, tpr_curve, _ = roc_curve(test_labels, test_errors)
    auc    = roc_auc_score(test_labels, test_errors)
    pr_auc = average_precision_score(test_labels, test_errors)

    recall_at_fpr_1pct  = recall_at_fpr(fpr_curve, tpr_curve, 0.01)
    recall_at_fpr_5pct  = recall_at_fpr(fpr_curve, tpr_curve, 0.05)
    recall_at_fpr_10pct = recall_at_fpr(fpr_curve, tpr_curve, 0.10)

    tn, fp, fn, tp = confusion_matrix(test_labels, y_pred).ravel()
    alert_rate     = round(float((y_pred == 1).sum() / len(y_pred)), 6)

    print(f"Test Precision : {precision:.4f}")
    print(f"Test Recall    : {recall:.4f}")
    print(f"Test F1-score  : {f1:.4f}")
    print(f"Test ROC-AUC   : {auc:.4f}")
    print(f"Test PR-AUC    : {pr_auc:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Alert Rate     : {alert_rate:.6f}")
    print(f"Recall @ FPR 5%: {recall_at_fpr_5pct:.4f}")
    print(f"FNR            : {fn / (fn + tp):.4f}")
    print(f"Alert Fatigue  : {tp / (tp + fp):.4f}")
    print(f"Threshold      : {threshold_method} | value: {threshold:.6f}")

    metrics_payload = {
        "model_path":        str(resolved_model_path),
        "threshold_path":    str(resolved_threshold_path),
        "threshold":         float(threshold),
        "threshold_method":  threshold_method,
        "precision":         float(precision),
        "recall":            float(recall),
        "f1":                float(f1),
        "roc_auc":           float(auc),
        "pr_auc":            float(pr_auc),
        "alert_rate":        alert_rate,
        "confusion_matrix":  {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "recall_at_fpr_1pct":  recall_at_fpr_1pct,
        "recall_at_fpr_5pct":  recall_at_fpr_5pct,
        "recall_at_fpr_10pct": recall_at_fpr_10pct,
        "fnr":               float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "tpr":               float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "scoring":           scoring,
        "evaluated_at":      datetime.now(timezone.utc).isoformat(),
    }

    metrics_path = resolved_run_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VAE on CICIDS2017.")
    parser.add_argument("--data-dir",             default="data/cicids2017")
    parser.add_argument("--run-dir",              default="outputs/default")
    parser.add_argument("--model-path",           default=None)
    parser.add_argument("--threshold-path",       default=None)
    parser.add_argument("--reuse-threshold",      action="store_true")
    parser.add_argument("--threshold-method",     default="roc_optimal",choices=["percentile", "sigma", "roc_optimal"])
    parser.add_argument("--threshold-k",          type=float, default=3.0)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--device",               default=None)
    parser.add_argument("--scoring",              choices=["mse", "elbo"], default="mse")
    args = parser.parse_args()

    evaluate_vae_cicids2017(
        data_dir=args.data_dir,
        run_dir=args.run_dir,
        model_path=args.model_path,
        threshold_path=args.threshold_path,
        reuse_threshold=args.reuse_threshold,
        threshold_method=args.threshold_method,
        threshold_k=args.threshold_k,
        threshold_percentile=args.threshold_percentile,
        device=args.device,
        scoring=args.scoring,
    )