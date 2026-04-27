import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


def calibrate_threshold(errors, method: str = "percentile", percentile: float = 95, k: float = 3.0, val_labels=None):
    errors_np = np.asarray(errors, dtype=float)
    if errors_np.size == 0:
        raise ValueError("Cannot calibrate threshold from an empty error array.")

    if method == "percentile":
        return float(np.percentile(errors_np, percentile))

    elif method == "sigma":
        mu = float(np.mean(errors_np))
        sigma = float(np.std(errors_np))
        return mu + k * sigma

    elif method == "roc_optimal":
        if val_labels is None:
            raise ValueError("val_labels must be provided for roc_optimal thresholding.")
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(val_labels, errors_np)
        youden_j = tpr - fpr
        best_idx = int(np.argmax(youden_j))
        return float(thresholds[best_idx])

    else:
        raise ValueError(f"Unsupported thresholding method: {method}. Choose from: percentile, sigma, roc_optimal")


def save_threshold(path, payload):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_payload = dict(payload)
    if "created_at" not in serializable_payload:
        serializable_payload["created_at"] = datetime.now(timezone.utc).isoformat()

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(serializable_payload, f, indent=2)


def load_threshold(path):
    in_path = Path(path)
    with in_path.open("r", encoding="utf-8") as f:
        return json.load(f)