import json
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
import sklearn
from torch.serialization import add_safe_globals
from pathlib import Path
import argparse
from datetime import datetime, timezone
from torch.utils.data import DataLoader, TensorDataset

from data_utils import load_nsl_kdd_raw
from vae_model import VAE
from thresholding import calibrate_threshold, save_threshold, load_threshold

def recall_at_fpr(fpr_curve, tpr_curve, target_fpr):
    idx = np.searchsorted(fpr_curve, target_fpr, side='right') - 1
    idx = max(0, min(idx, len(tpr_curve) - 1))
    return float(tpr_curve[idx])

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


def _build_eval_loaders_with_saved_preprocessor(
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

    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    X_val_np = X_val_processed.toarray() if hasattr(X_val_processed, "toarray") else X_val_processed
    X_test_np = X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed

    X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.int64)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)

    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=1024, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=1024, shuffle=False)

    train_normals = int((y_train == 0).sum())
    print(
        f"Dataset split sizes | train_total={len(y_train)}, "
        f"train_normals={train_normals}, val={len(y_val)}, test={len(y_test)}"
    )

    return val_loader, test_loader, X_val_tensor.shape[1]


def evaluate_vae_nsl_kdd(
    data_dir: str = "../data/nsl_kdd",
    run_dir: str = "../outputs/default",
    model_path: str | None = None,
    threshold_path: str | None = None,
    reuse_threshold: bool = False,
    threshold_method: str = "percentile",
    threshold_percentile: float = 95,
    device: str | None = None,
    reshuffle_all: bool = False,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    resolved_run_dir = Path(run_dir)
    resolved_model_path = Path(model_path) if model_path is not None else resolved_run_dir / "model.pt"

    add_safe_globals([
        sklearn.compose.ColumnTransformer,
        sklearn.preprocessing.OneHotEncoder,
        sklearn.preprocessing.StandardScaler
    ])

    df_train, df_test = load_nsl_kdd_raw(data_dir)
    checkpoint = torch.load(resolved_model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    checkpoint_preprocessor = checkpoint.get("preprocessor")
    if checkpoint_preprocessor is None:
        raise ValueError(
            "Checkpoint does not contain 'preprocessor'. "
            "Use a checkpoint produced by the current training script."
        )

    val_loader, test_loader, transformed_input_dim = _build_eval_loaders_with_saved_preprocessor(
        df_train,
        df_test,
        checkpoint_preprocessor,
        reshuffle_all=reshuffle_all,
    )

    latent_dim = int(checkpoint.get("latent_dim", state_dict["fc_mu.weight"].shape[0]))
    hidden_dims = _infer_hidden_dims_from_state_dict(state_dict)
    input_dim = int(checkpoint.get("input_dim", transformed_input_dim))
    if transformed_input_dim != input_dim:
        raise ValueError(
            f"Feature dimension mismatch after preprocessing: got {transformed_input_dim}, "
            f"but checkpoint expects input_dim={input_dim}."
        )

    model = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    val_errors = []
    val_labels = []
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            x_recon, mu, logvar = model(x_val)
            err = ((x_recon - x_val) ** 2).mean(dim=1).cpu().numpy()
            val_errors.extend(err)
            val_labels.extend(y_val.numpy())

    val_errors = np.array(val_errors)
    val_labels = np.array(val_labels)

    resolved_threshold_path = (
        Path(threshold_path)
        if threshold_path is not None
        else resolved_run_dir / "threshold.json"
    )

    if reuse_threshold and resolved_threshold_path.exists():
        threshold_payload = load_threshold(resolved_threshold_path)
        threshold = float(threshold_payload["value"])
        print(
            f"Loaded anomaly threshold from {resolved_threshold_path}: "
            f"{threshold:.6f} ({threshold_payload.get('method', 'unknown')})"
        )
    else:
        normal_val_errors = val_errors[val_labels == 0]
        threshold = calibrate_threshold(
            normal_val_errors,
            method=threshold_method,
            percentile=threshold_percentile,
        )
        threshold_payload = {
            "method": threshold_method,
            "percentile": threshold_percentile,
            "value": threshold,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        save_threshold(resolved_threshold_path, threshold_payload)
        print(
            f"Calibrated anomaly threshold ({threshold_method}, "
            f"p{threshold_percentile:g}) and saved to {resolved_threshold_path}: "
            f"{threshold:.6f}"
        )

    test_errors = []
    test_labels = []
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            x_recon, mu, logvar = model(x_test)
            err = ((x_recon - x_test) ** 2).mean(dim=1).cpu().numpy()
            test_errors.extend(err)
            test_labels.extend(y_test.numpy())

    test_errors = np.array(test_errors)
    test_labels = np.array(test_labels)

    y_pred = (test_errors > threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, y_pred, average="binary", pos_label=1
    )
    fpr_curve, tpr_curve, _ = roc_curve(test_labels, test_errors)
    auc = roc_auc_score(test_labels, test_errors)
    pr_auc = average_precision_score(test_labels, test_errors)

    recall_at_fpr_1pct  = recall_at_fpr(fpr_curve, tpr_curve, 0.01)
    recall_at_fpr_5pct  = recall_at_fpr(fpr_curve, tpr_curve, 0.05)
    recall_at_fpr_10pct = recall_at_fpr(fpr_curve, tpr_curve, 0.10)

    tn, fp, fn, tp = confusion_matrix(test_labels, y_pred).ravel()
    alert_rate = round(float((y_pred == 1).sum() / len(y_pred)), 6)

    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1-score:  {f1:.4f}")
    print(f"Test ROC-AUC:   {auc:.4f}")
    print(f"Test PR-AUC:    {pr_auc:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Alert Rate: {alert_rate:.6f}")
    print(f"Recall at FPR 1%:  {recall_at_fpr_1pct:.4f}")
    print(f"Recall at FPR 5%:  {recall_at_fpr_5pct:.4f}")
    print(f"Recall at FPR 10%: {recall_at_fpr_10pct:.4f}")
    print(f"FNR (at p95 threshold): {fn / (fn + tp):.4f}")

    metrics_payload = {
        "model_path": str(resolved_model_path),
        "threshold_path": str(resolved_threshold_path),
        "threshold": float(threshold),
        "threshold_method": threshold_method,
        "threshold_percentile": float(threshold_percentile),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
        "pr_auc": float(pr_auc),
        "alert_rate": alert_rate,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "recall_at_fpr_1pct":  recall_at_fpr_1pct,
        "recall_at_fpr_5pct":  recall_at_fpr_5pct,
        "recall_at_fpr_10pct": recall_at_fpr_10pct,
        "fnr": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path = resolved_run_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VAE on NSL-KDD.")
    parser.add_argument("--data-dir", default="../data/nsl_kdd")
    parser.add_argument("--run-dir", default="../outputs/default")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--threshold-path", default=None)
    parser.add_argument("--reuse-threshold", action="store_true")
    parser.add_argument("--threshold-method", default="percentile")
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reshuffle-all", action="store_true")
    args = parser.parse_args()

    evaluate_vae_nsl_kdd(
        data_dir=args.data_dir,
        run_dir=args.run_dir,
        model_path=args.model_path,
        threshold_path=args.threshold_path,
        reuse_threshold=args.reuse_threshold,
        threshold_method=args.threshold_method,
        threshold_percentile=args.threshold_percentile,
        device=args.device,
        reshuffle_all=args.reshuffle_all,
    )