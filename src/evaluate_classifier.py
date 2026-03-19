import json
import pickle
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch
from torch.serialization import add_safe_globals
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from data_utils import load_nsl_kdd_raw
from data_utils_stage2 import load_stage2_data


def evaluate_classifier(
    data_dir: str = "data/nsl_kdd",
    clf_run_dir: str = "outputs/clf_default",
    confidence_threshold: float = 0.6,
    val_size: float = 0.2,
    random_state: int = 42,
):
    clf_run_path = Path(clf_run_dir)
    model_path   = clf_run_path / "model.pkl"
    le_path      = clf_run_path / "label_encoder.pkl"
    config_path  = clf_run_path / "config.json"
    metrics_path = clf_run_path / "metrics.json"

    with config_path.open("r") as f:
        config = json.load(f)

    vae_model_path = Path(config["vae_run_dir"]) / "model.pt"
    print(f"Loading VAE checkpoint from : {vae_model_path}")
    print(f"Loading classifier from     : {model_path}")

    add_safe_globals([
        sklearn.compose.ColumnTransformer,
        sklearn.preprocessing.OneHotEncoder,
        sklearn.preprocessing.StandardScaler,
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(vae_model_path, map_location=device, weights_only=False)
    preprocessor = checkpoint.get("preprocessor")
    if preprocessor is None:
        raise ValueError("VAE checkpoint does not contain a preprocessor.")

    with model_path.open("rb") as f:
        payload = pickle.load(f)
        clf = payload["clf"] if isinstance(payload, dict) else payload
        selector = payload.get("selector") if isinstance(payload, dict) else None

    with le_path.open("rb") as f:
        le = pickle.load(f)

    print("Loading NSL-KDD raw data...")
    df_train, df_test = load_nsl_kdd_raw(data_dir)

    _, _, _, _, X_test, y_test, _ = load_stage2_data(
        df_train, df_test, preprocessor,
        val_size=val_size,
        random_state=random_state,
    )

    valid_mask = y_test != -1
    if not valid_mask.all():
        n_dropped = int((~valid_mask).sum())
        print(f"Dropping {n_dropped} test samples with unseen attack categories.")
    X_test = X_test[valid_mask]
    y_test = y_test[valid_mask]

    proba = clf.predict_proba(X_test)
    y_pred_raw = np.argmax(proba, axis=1)
    confidence = proba[np.arange(len(proba)), y_pred_raw]

    novel_mask = confidence < confidence_threshold
    y_pred = y_pred_raw.copy().astype(object)
    y_pred[novel_mask] = "novel_anomaly"

    novel_rate = float(novel_mask.sum() / len(novel_mask))
    print(f"\nNovel anomaly rate (confidence < {confidence_threshold}): {novel_rate:.4f}")

    known_mask = ~novel_mask
    y_test_known = y_test[known_mask]
    y_pred_known = y_pred_raw[known_mask]

    class_names = list(le.classes_)
    macro_f1 = f1_score(y_test_known, y_pred_known, average="macro")

    print("\nClassification Report (known-class predictions only):")
    print(classification_report(y_test_known, y_pred_known, target_names=class_names))

    cm = confusion_matrix(y_test_known, y_pred_known)
    print("Confusion Matrix (rows=true, cols=predicted):")
    print("Classes:", class_names)
    print(cm)

    per_class_recall = {}
    for i, cls in enumerate(class_names):
        mask = y_test_known == i
        if mask.sum() == 0:
            per_class_recall[cls] = None
        else:
            per_class_recall[cls] = float((y_pred_known[mask] == i).sum() / mask.sum())

    print("\nPer-class recall:")
    for cls, rec in per_class_recall.items():
        print(f"  {cls:10s}: {rec:.4f}" if rec is not None else f"  {cls:10s}: N/A")

    metrics_payload = {
        "clf_run_dir": str(clf_run_dir),
        "confidence_threshold": confidence_threshold,
        "macro_f1": float(macro_f1),
        "novel_anomaly_rate": novel_rate,
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "n_test_samples": int(len(y_test)),
        "n_novel": int(novel_mask.sum()),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 attack classifier.")
    parser.add_argument("--data-dir",             default="data/nsl_kdd")
    parser.add_argument("--clf-run-dir",          default="outputs/clf_default")
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--val-size",             type=float, default=0.2)
    parser.add_argument("--random-state",         type=int,   default=42)
    args = parser.parse_args()

    evaluate_classifier(
        data_dir=args.data_dir,
        clf_run_dir=args.clf_run_dir,
        confidence_threshold=args.confidence_threshold,
        val_size=args.val_size,
        random_state=args.random_state,
    )