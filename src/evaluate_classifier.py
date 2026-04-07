import json
import pickle
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from data_utils import load_cicids2017_raw
from data_utils_stage2 import load_stage2_data


def evaluate_classifier(
    data_dir: str = "data/cicids2017",
    clf_run_dir: str = "outputs/clf_default",
    confidence_threshold: float = 0.6,
    val_size: float = 0.1,
    test_size: float = 0.2,
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(vae_model_path, map_location=device, weights_only=False)
    scaler       = checkpoint.get("scaler")
    feature_meta = checkpoint.get("feature_meta")
    if scaler is None or feature_meta is None:
        raise ValueError(
            "VAE checkpoint missing 'scaler' or 'feature_meta'. "
            "Retrain the VAE with the current train_vae.py."
        )

    with model_path.open("rb") as f:
        payload = pickle.load(f)
        clf = payload["clf"] if isinstance(payload, dict) else payload

    with le_path.open("rb") as f:
        le = pickle.load(f)

    print("Loading CICIDS2017 attack data...")
    _df_monday, df_attacks = load_cicids2017_raw(data_dir)

    _, _, _, _, X_test, y_test, _ = load_stage2_data(
        df_attacks,
        scaler=scaler,
        feature_meta=feature_meta,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    proba = clf.predict_proba(X_test)
    y_pred_raw = np.argmax(proba, axis=1)
    confidence = proba[np.arange(len(proba)), y_pred_raw]

    novel_mask = confidence < confidence_threshold
    novel_rate = float(novel_mask.sum() / len(novel_mask))
    print(f"\nNovel anomaly rate (confidence < {confidence_threshold}): {novel_rate:.4f}")

    known_mask = ~novel_mask
    y_test_known = y_test[known_mask]
    y_pred_known = y_pred_raw[known_mask]

    class_names = list(le.classes_)

    present_labels = sorted(set(np.concatenate([y_test_known, y_pred_known])))
    present_target_names = [class_names[i] for i in present_labels]

    macro_f1 = f1_score(
        y_test_known, y_pred_known,
        labels=present_labels,
        average="macro",
        zero_division=0,
    )

    print("\nClassification Report (known-class predictions only):")
    print(classification_report(
        y_test_known, y_pred_known,
        labels=present_labels,
        target_names=present_target_names,
        zero_division=0,
    ))

    cm = confusion_matrix(y_test_known, y_pred_known, labels=list(range(len(class_names))))
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
        if rec is None:
            print(f"  {cls:15s}: N/A")
        else:
            print(f"  {cls:15s}: {rec:.4f}")

    per_class_novel = {}
    for i, cls in enumerate(class_names):
        mask = y_test == i
        n_total = int(mask.sum())
        if n_total == 0:
            per_class_novel[cls] = None
        else:
            n_novel = int((mask & novel_mask).sum())
            per_class_novel[cls] = {
                "n_total": n_total,
                "n_routed_to_novel": n_novel,
                "novel_rate": float(n_novel / n_total),
            }

    print("\nPer-class routing to novel_anomaly (all test samples):")
    for cls, stats in per_class_novel.items():
        if stats is None:
            print(f"  {cls:15s}: N/A")
        else:
            print(f"  {cls:15s}: {stats['n_routed_to_novel']:6d}/{stats['n_total']:6d}  "
                  f"({stats['novel_rate']*100:.1f}%)")

    metrics_payload = {
        "clf_run_dir": str(clf_run_dir),
        "confidence_threshold": confidence_threshold,
        "macro_f1": float(macro_f1),
        "novel_anomaly_rate": novel_rate,
        "per_class_recall": per_class_recall,
        "per_class_novel_routing": per_class_novel,
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
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 attack classifier on CICIDS2017.")
    parser.add_argument("--data-dir",             default="data/cicids2017")
    parser.add_argument("--clf-run-dir",          default="outputs/clf_default")
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--val-size",             type=float, default=0.1)
    parser.add_argument("--test-size",            type=float, default=0.2)
    parser.add_argument("--random-state",         type=int,   default=42)
    args = parser.parse_args()

    evaluate_classifier(
        data_dir=args.data_dir,
        clf_run_dir=args.clf_run_dir,
        confidence_threshold=args.confidence_threshold,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )