import json
import pickle
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

from data_utils import load_cicids2017_raw
from data_utils_stage2 import load_stage2_data


def cross_validate_classifier(
    data_dir: str = "data/cicids2017",
    vae_run_dir: str = "outputs/default",
    clf_run_dir: str = "outputs/clf_default",
    n_splits: int = 5,
    n_estimators: int = 200,
    random_state: int = 42,
    val_size: float = 0.1,
    test_size: float = 0.2,
):
    #Run stratified k-fold CV on the training split and save per-fold and summary metrics to cv_metrics.json.
    clf_run_path = Path(clf_run_dir)
    clf_run_path.mkdir(parents=True, exist_ok=True)

    vae_model_path = Path(vae_run_dir) / "model.pt"
    print(f"Loading VAE checkpoint from: {vae_model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(vae_model_path, map_location=device, weights_only=False)

    scaler       = checkpoint.get("scaler")
    feature_meta = checkpoint.get("feature_meta")
    if scaler is None or feature_meta is None:
        raise ValueError(
            "VAE checkpoint missing 'scaler' or 'feature_meta'. "
            "Retrain the VAE with the current train_vae.py."
        )

    print("Loading CICIDS2017 attack data...")
    _df_monday, df_attacks = load_cicids2017_raw(data_dir)

    X_train, y_train, _, _, _, _, le = load_stage2_data(
        df_attacks,
        scaler=scaler,
        feature_meta=feature_meta,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    class_names = list(le.classes_)
    all_labels  = list(range(len(class_names)))

    print(f"\nRunning {n_splits}-fold stratified CV on training split "
          f"(n={len(y_train):,}, classes={class_names})")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_results = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        #class_weight="balanced" recomputes weights per fold subset, not on the full training distribution - can produce inconsistent weighting across folds if class proportions vary significantly.
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_va)

        present_labels = sorted(set(np.concatenate([y_va, y_pred])))
        macro_f1 = f1_score(y_va, y_pred, labels=present_labels, average="macro", zero_division=0)

        per_class_f1 = f1_score(y_va, y_pred, labels=all_labels, average=None, zero_division=0)
        per_class_f1_dict = {class_names[i]: float(per_class_f1[i]) for i in range(len(class_names))}

        print(f"\n[Fold {fold_idx}/{n_splits}] train={len(y_tr):,} val={len(y_va):,}  macro_F1={macro_f1:.4f}")
        for cls, f in per_class_f1_dict.items():
            print(f"    {cls:15s} F1={f:.4f}")

        fold_results.append({
            "fold": fold_idx,
            "n_train": int(len(y_tr)),
            "n_val":   int(len(y_va)),
            "macro_f1": float(macro_f1),
            "per_class_f1": per_class_f1_dict,
        })

    macro_f1_values = np.array([r["macro_f1"] for r in fold_results])
    mean_f1 = float(macro_f1_values.mean())
    std_f1  = float(macro_f1_values.std(ddof=1)) if len(macro_f1_values) > 1 else 0.0

    print("\n" + "=" * 55)
    print(f"CROSS-VALIDATION SUMMARY ({n_splits}-fold stratified)")
    print("=" * 55)
    for r in fold_results:
        print(f"  Fold {r['fold']}: macro_F1 = {r['macro_f1']:.4f}")
    print("-" * 55)
    print(f"  Mean macro_F1 : {mean_f1:.4f}")
    print(f"  Std  macro_F1 : {std_f1:.4f}")
    print("=" * 55)

    payload = {
        "clf_run_dir": str(clf_run_dir),
        "vae_run_dir": str(vae_run_dir),
        "n_splits": n_splits,
        "n_estimators": n_estimators,
        "class_weight": "balanced",
        "random_state": random_state,
        "val_size": val_size,
        "test_size": test_size,
        "class_names": class_names,
        "n_train_total": int(len(y_train)),
        "fold_results": fold_results,
        "mean_macro_f1": mean_f1,
        "std_macro_f1":  std_f1,
        "evaluated_at":  datetime.now(timezone.utc).isoformat(),
    }

    out_path = clf_run_path / "cv_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nCV metrics saved to {out_path}")

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified k-fold CV for Stage 2 classifier on CICIDS2017.")
    parser.add_argument("--data-dir",     default="data/cicids2017")
    parser.add_argument("--vae-run-dir",  default="outputs/vae_20260405_213312")
    parser.add_argument("--clf-run-dir",  default="outputs/clf_20260415_144411")
    parser.add_argument("--n-splits",     type=int,   default=5)
    parser.add_argument("--n-estimators", type=int,   default=200)
    parser.add_argument("--random-state", type=int,   default=42)
    parser.add_argument("--val-size",     type=float, default=0.1)
    parser.add_argument("--test-size",    type=float, default=0.2)
    args = parser.parse_args()

    cross_validate_classifier(
        data_dir=args.data_dir,
        vae_run_dir=args.vae_run_dir,
        clf_run_dir=args.clf_run_dir,
        n_splits=args.n_splits,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        val_size=args.val_size,
        test_size=args.test_size,
    )