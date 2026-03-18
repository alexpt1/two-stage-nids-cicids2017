import json
import pickle
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch
from torch.serialization import add_safe_globals
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from data_utils import load_nsl_kdd_raw
from data_utils_stage2 import load_stage2_data


def train_classifier(
    data_dir: str = "data/nsl_kdd",
    vae_run_dir: str = "outputs/default",
    outputs_root: str = "outputs",
    model_type: str = "rf",
    n_estimators: int = 200,
    random_state: int = 42,
    val_size: float = 0.2,
):
    run_id = "clf_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(outputs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    vae_model_path = Path(vae_run_dir) / "model.pt"
    print(f"Loading VAE checkpoint from: {vae_model_path}")

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

    print("Loading NSL-KDD raw data...")
    df_train, df_test = load_nsl_kdd_raw(data_dir)

    X_train, y_train, X_val, y_val, X_test, y_test, le = load_stage2_data(
        df_train, df_test, preprocessor,
        val_size=val_size,
        random_state=random_state,
    )
    
    print(f"\nTraining {model_type.upper()} classifier...")

    if model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    #RandomizedSearchCV can be added here if desired - achieved nearly optimal result but took much longer to train"
    #if model_type == "rf":
        #from sklearn.model_selection import RandomizedSearchCV

        #base_rf = RandomForestClassifier(
        #    class_weight="balanced",
        #    random_state=random_state,
        #    n_jobs=-1,
        #)

        #param_dist = {
        #    "n_estimators": [100, 200, 300, 500],
        #    "max_depth": [None, 20, 40, 60],
        #    "min_samples_leaf": [1, 2, 4],
        #    "min_samples_split": [2, 5, 10],
        #    "max_features": ["sqrt", "log2"],
        #}

        #search = RandomizedSearchCV(
        #    base_rf,
        #    param_distributions=param_dist,
        #    n_iter=20,
        #    scoring="f1_macro",
        #    cv=3,
        #    random_state=random_state,
        #    n_jobs=-1,
        #    verbose=2,
        #)
        #search.fit(X_train, y_train)
        #clf = search.best_estimator_
        #print(f"\nBest parameters: {search.best_params_}")
        #print(f"Best CV macro F1: {search.best_score_:.4f}")
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
    print(classification_report(y_val, val_preds, target_names=le.classes_))

    model_path = run_dir / "model.pkl"
    le_path    = run_dir / "label_encoder.pkl"
    config_path = run_dir / "config.json"

    with model_path.open("wb") as f:
        pickle.dump(clf, f)

    with le_path.open("wb") as f:
        pickle.dump(le, f)

    config = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": model_type,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "val_size": val_size,
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
    parser = argparse.ArgumentParser(description="Train Stage 2 attack classifier.")
    parser.add_argument("--data-dir",      default="data/nsl_kdd")
    parser.add_argument("--vae-run-dir",   default="outputs/default")
    parser.add_argument("--outputs-root",  default="outputs")
    parser.add_argument("--model-type",    default="rf", choices=["rf", "xgb"])
    parser.add_argument("--n-estimators",  type=int, default=200)
    parser.add_argument("--random-state",  type=int, default=42)
    parser.add_argument("--val-size",      type=float, default=0.2)
    args = parser.parse_args()

    train_classifier(
        data_dir=args.data_dir,
        vae_run_dir=args.vae_run_dir,
        outputs_root=args.outputs_root,
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        val_size=args.val_size,
    )