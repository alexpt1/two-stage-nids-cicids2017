import json
import pickle
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from data_utils import load_cicids2017_raw
from data_utils_stage2 import load_stage2_data


def train_classifier(
    data_dir: str = "data/cicids2017",
    vae_run_dir: str = "outputs/default",
    outputs_root: str = "outputs",
    model_type: str = "rf",
    n_estimators: int = 200,
    random_state: int = 42,
    val_size: float = 0.1,
    test_size: float = 0.2,
):
    run_id = "clf_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(outputs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

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

    X_train, y_train, X_val, y_val, X_test, y_test, le = load_stage2_data(
        df_attacks,
        scaler=scaler,
        feature_meta=feature_meta,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    print(f"\nTraining {model_type.upper()} classifier with class_weight='balanced'...")

    if model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        clf = XGBClassifier(
            n_estimators=n_estimators,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'. Choose 'rf' or 'xgb'.")

    clf.fit(X_train, y_train)

    val_preds = clf.predict(X_val)
    print("\nValidation report:")
    present_labels = sorted(set(np.concatenate([y_val, val_preds])))
    present_target_names = [le.classes_[i] for i in present_labels]
    print(classification_report(
        y_val, val_preds,
        labels=present_labels,
        target_names=present_target_names,
        zero_division=0,
    ))

    model_path  = run_dir / "model.pkl"
    le_path     = run_dir / "label_encoder.pkl"
    config_path = run_dir / "config.json"

    with model_path.open("wb") as f:
        pickle.dump(clf, f)

    with le_path.open("wb") as f:
        pickle.dump(le, f)

    config = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "cicids2017",
        "model_type": model_type,
        "n_estimators": n_estimators,
        "class_weight": "balanced",
        "random_state": random_state,
        "val_size": val_size,
        "test_size": test_size,
        "vae_run_dir": str(vae_run_dir),
        "classes": list(le.classes_),
        "input_dim": int(X_train.shape[1]),
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nRun directory : {run_dir}")
    print(f"Model saved   : {model_path}")
    print(f"Label encoder : {le_path}")
    print(f"Config saved  : {config_path}")

    return str(run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 2 attack classifier on CICIDS2017.")
    parser.add_argument("--data-dir",      default="data/cicids2017")
    parser.add_argument("--vae-run-dir",   default="outputs/default")
    parser.add_argument("--outputs-root",  default="outputs")
    parser.add_argument("--model-type",    default="rf", choices=["rf", "xgb"])
    parser.add_argument("--n-estimators",  type=int, default=200)
    parser.add_argument("--random-state",  type=int, default=42)
    parser.add_argument("--val-size",      type=float, default=0.1)
    parser.add_argument("--test-size",     type=float, default=0.2)
    args = parser.parse_args()

    train_classifier(
        data_dir=args.data_dir,
        vae_run_dir=args.vae_run_dir,
        outputs_root=args.outputs_root,
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        val_size=args.val_size,
        test_size=args.test_size,
    )