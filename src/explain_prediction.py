import json
import pickle
import argparse
import numpy as np
import torch
import sklearn
import shap
from pathlib import Path
from torch.serialization import add_safe_globals

from data_utils import load_nsl_kdd_raw
from data_utils_stage2 import map_attack_categories, ATTACK_CATEGORY_MAP
from vae_model import VAE
from thresholding import load_threshold


FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

COST_WEIGHTS = {"DoS": 10, "Probe": 5, "R2L": 20, "U2R": 50}

def get_severity(confidence: float, attack_class: str) -> str:
    cost = COST_WEIGHTS.get(attack_class, 1)
    score = confidence * cost
    if score >= 40:
        return "CRITICAL"
    elif score >= 10:
        return "HIGH"
    elif score >= 8:
        return "MEDIUM"
    return "LOW"

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


def explain_predictions(
    data_dir: str = "data/nsl_kdd",
    vae_run_dir: str = "outputs/default",
    clf_run_dir: str = "outputs/clf_default",
    confidence_threshold: float = 0.6,
    n_samples: int = 5,
    n_background: int = 100,
    random_state: int = 42,
    device: str = "cpu",
):
    add_safe_globals([
        sklearn.compose.ColumnTransformer,
        sklearn.preprocessing.OneHotEncoder,
        sklearn.preprocessing.StandardScaler,
    ])

    vae_run_path   = Path(vae_run_dir)
    vae_model_path = vae_run_path / "model.pt"
    threshold_path = vae_run_path / "threshold.json"

    checkpoint   = torch.load(vae_model_path, map_location=device, weights_only=False)
    preprocessor = checkpoint["preprocessor"]
    state_dict   = checkpoint["model_state_dict"]

    input_dim   = int(checkpoint["input_dim"])
    latent_dim  = int(checkpoint["latent_dim"])
    hidden_dims = _infer_hidden_dims(state_dict)

    vae = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)
    vae.load_state_dict(state_dict)
    vae.to(device).eval()

    threshold_payload = load_threshold(threshold_path)
    threshold = float(threshold_payload["value"])

    clf_run_path = Path(clf_run_dir)
    with (clf_run_path / "model.pkl").open("rb") as f:
        payload = pickle.load(f)
    clf = payload["clf"] if isinstance(payload, dict) else payload
    with (clf_run_path / "label_encoder.pkl").open("rb") as f:
        le = pickle.load(f)

    df_train, df_test = load_nsl_kdd_raw(data_dir)
    df_test = map_attack_categories(df_test)
    feature_cols = list(range(41))

    rng = np.random.default_rng(random_state)

    X_all_proc = preprocessor.transform(df_test[feature_cols])
    X_all_np   = X_all_proc.toarray() if hasattr(X_all_proc, "toarray") else X_all_proc
    X_tensor   = torch.tensor(X_all_np, dtype=torch.float32).to(device)

    with torch.no_grad():
        vae_out = vae(X_tensor)
        x_recon = vae_out[0]
        recon_errors = ((x_recon - X_tensor) ** 2).mean(dim=1).cpu().numpy()

    anomaly_mask = recon_errors > threshold
    X_anomaly_np = X_all_np[anomaly_mask]
    recon_errors_anomaly = recon_errors[anomaly_mask]

    if len(X_anomaly_np) == 0:
        print("No anomalies detected in test set.")
        return

    proba      = clf.predict_proba(X_anomaly_np)
    pred_idx   = np.argmax(proba, axis=1)
    confidence = proba[np.arange(len(proba)), pred_idx]

    confident_mask  = confidence >= confidence_threshold
    X_confident     = X_anomaly_np[confident_mask]
    pred_confident  = pred_idx[confident_mask]
    conf_confident  = confidence[confident_mask]
    recon_confident = recon_errors_anomaly[confident_mask]

    if len(X_confident) == 0:
        print("No high-confidence predictions to explain.")
        return

    sample_idx = rng.choice(len(X_confident), size=min(n_samples, len(X_confident)), replace=False)
    X_explain  = X_confident[sample_idx]
    y_explain  = pred_confident[sample_idx]
    c_explain  = conf_confident[sample_idx]
    r_explain  = recon_confident[sample_idx]

    bg_idx        = rng.choice(len(X_anomaly_np), size=min(n_background, len(X_anomaly_np)), replace=False)
    X_background  = X_anomaly_np[bg_idx]

    print("Computing SHAP values via TreeExplainer...")
    explainer   = shap.TreeExplainer(clf, data=X_background, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_explain)

    cat_feature_names = []
    categorical_col_names = ["protocol_type", "service", "flag"]
    for col_name, cat_vals in zip(categorical_col_names, preprocessor.named_transformers_["cat"].categories_):
        for v in cat_vals:
            cat_feature_names.append(f"{col_name}={v}")
    numeric_feature_names = [FEATURE_NAMES[i] for i in range(41) if i not in [1, 2, 3]]
    all_feature_names = cat_feature_names + numeric_feature_names
    
    #print(f"DEBUG: all_feature_names length = {len(all_feature_names)}, input_dim = {input_dim}")

    print(f"\n{'='*60}")
    print(f"SHAP EXPLANATIONS - Top 5 features per prediction")
    print(f"{'='*60}")

    results = []
    for i in range(len(X_explain)):
        class_idx  = y_explain[i]
        class_name = le.classes_[class_idx]
        conf       = c_explain[i]

        if isinstance(shap_values, list):
            sv = np.array(shap_values[class_idx][i]).flatten()
        else:
            n_classes = len(le.classes_)
            n_features = input_dim
            sv_full = np.array(shap_values[i]).flatten()
            sv = sv_full[class_idx * n_features:(class_idx + 1) * n_features]
        
        top5_idx    = np.argsort(np.abs(sv)).flatten()[::-1][:5]
        top_features = []
        for fi in top5_idx.tolist():
            fname = all_feature_names[fi] if fi < len(all_feature_names) else f"feature_{fi}"
            top_features.append({
                "feature": fname,
                "shap_value": float(sv[fi]),
                "direction": "increases risk" if sv[fi] > 0 else "decreases risk",
            })

        recon_err  = float(r_explain[i])
        severity   = get_severity(conf, class_name)
        top_pos    = [f for f in top_features if f["shap_value"] > 0]
        top_neg    = [f for f in top_features if f["shap_value"] < 0]
        pos_names  = ", ".join(f["feature"] for f in top_pos[:2])
        neg_names  = ", ".join(f["feature"] for f in top_neg[:1])
        plain_eng  = f"Flagged as {class_name} primarily due to elevated {pos_names}"
        if neg_names:
            plain_eng += f", with {neg_names} reducing suspicion"

        print(f"\nSample {i+1} | Predicted: {class_name} | Confidence: {conf:.3f} | Severity: {severity}")
        print(f"  VAE reconstruction error : {recon_err:.6f}  (threshold={threshold:.6f})")
        print(f"  Verdict                  : {plain_eng}")
        print(f"  Top contributing features:")
        for f in top_features:
            raw_val = float(X_explain[i][top5_idx.tolist()[top_features.index(f)]])
            direction_symbol = "+" if f["shap_value"] > 0 else "-"
            print(f"    [{direction_symbol}] {f['feature']:35s} value={raw_val:+.4f}  (SHAP={f['shap_value']:+.4f})")

        results.append({
            "sample_index": int(sample_idx[i]),
            "predicted_class": class_name,
            "confidence": float(conf),
            "severity": severity,
            "vae_reconstruction_error": recon_err,
            "plain_english_verdict": plain_eng,
            "top_features": top_features,
        })

    out_path = Path(clf_run_dir) / "shap_explanations.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Explanations saved to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP explanations for Stage 2 classifier.")
    parser.add_argument("--data-dir",             default="data/nsl_kdd")
    parser.add_argument("--vae-run-dir",          default="outputs/default")
    parser.add_argument("--clf-run-dir",          default="outputs/clf_default")
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--n-samples",            type=int,   default=5)
    parser.add_argument("--n-background",         type=int,   default=100)
    parser.add_argument("--random-state",         type=int,   default=42)
    parser.add_argument("--device",               default="cpu")
    args = parser.parse_args()

    explain_predictions(
        data_dir=args.data_dir,
        vae_run_dir=args.vae_run_dir,
        clf_run_dir=args.clf_run_dir,
        confidence_threshold=args.confidence_threshold,
        n_samples=args.n_samples,
        n_background=args.n_background,
        random_state=args.random_state,
        device=args.device,
    )