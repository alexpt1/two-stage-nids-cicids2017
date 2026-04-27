import json
import pickle
import argparse
import numpy as np
import torch
import shap
from pathlib import Path

from data_utils import load_cicids2017_raw, apply_feature_transforms, apply_clip
from data_utils_stage2 import map_attack_categories
from vae_model import VAE
from thresholding import load_threshold


COST_WEIGHTS = {
    "DoS":          20, #Updated from 10, so that it can reach HIGH severity
    "Probe":         5,
    "BruteForce":   15,
    "WebAttack":    20,
    "Bot":          25,
    "Infiltration": 50,
    "Heartbleed":   50,
}


def get_severity(confidence: float, attack_class: str) -> str:
    cost = COST_WEIGHTS.get(attack_class, 1)
    score = confidence * cost
    if score >= 40:
        return "CRITICAL"
    elif score >= 15:
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


def _batched_recon_errors(vae, X_np, device, batch_size=4096):
    errors = np.empty(len(X_np), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(X_np), batch_size):
            end = min(start + batch_size, len(X_np))
            batch = torch.tensor(X_np[start:end], dtype=torch.float32).to(device)
            x_recon, _, _ = vae(batch)
            err = ((x_recon - batch) ** 2).mean(dim=1).cpu().numpy()
            errors[start:end] = err
    return errors


def explain_predictions(
    data_dir: str = "data/cicids2017",
    vae_run_dir: str = "outputs/default",
    clf_run_dir: str = "outputs/clf_default",
    confidence_threshold: float = 0.6,
    n_samples: int = 5,
    n_background: int = 100,
    max_data_rows: int = 200000,
    random_state: int = 42,
    device: str = "cpu",
):
    #Run the two-stage pipeline on attack-day data and produce stratified SHAP explanations with severity verdicts.
    
    vae_run_path   = Path(vae_run_dir)
    vae_model_path = vae_run_path / "model.pt"
    threshold_path = vae_run_path / "threshold.json"

    checkpoint   = torch.load(vae_model_path, map_location=device, weights_only=False)
    scaler       = checkpoint["scaler"]
    feature_meta = checkpoint["feature_meta"]
    state_dict   = checkpoint["model_state_dict"]

    input_dim   = int(checkpoint["input_dim"])
    latent_dim  = int(checkpoint["latent_dim"])
    hidden_dims = _infer_hidden_dims(state_dict)

    surviving_cols     = feature_meta["surviving_cols"]
    log_transform_cols = feature_meta["log_transform_cols"]
    clip_lower         = np.asarray(feature_meta["clip_lower"], dtype=np.float32)
    clip_upper         = np.asarray(feature_meta["clip_upper"], dtype=np.float32)

    feature_names = list(surviving_cols)

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

    print("Loading CICIDS2017 attack-day data...")
    _df_monday, df_attacks = load_cicids2017_raw(data_dir)
    df_attacks = map_attack_categories(df_attacks)

    rng = np.random.default_rng(random_state)

    if len(df_attacks) > max_data_rows:
        print(f"Subsampling {max_data_rows:,} rows from {len(df_attacks):,} attack-day rows for SHAP analysis")
        sub_idx = rng.choice(len(df_attacks), size=max_data_rows, replace=False)
        df_attacks = df_attacks.iloc[sub_idx].reset_index(drop=True)

    X_all_np = apply_feature_transforms(df_attacks, surviving_cols, log_transform_cols)
    X_all_np = apply_clip(X_all_np, clip_lower, clip_upper)
    X_all_np = scaler.transform(X_all_np).astype(np.float32)

    print(f"Running VAE inference on {len(X_all_np):,} rows (batched)...")
    recon_errors = _batched_recon_errors(vae, X_all_np, device)

    anomaly_mask = recon_errors > threshold
    X_anomaly_np = X_all_np[anomaly_mask]
    recon_errors_anomaly = recon_errors[anomaly_mask]

    if len(X_anomaly_np) == 0:
        print("No anomalies detected.")
        return

    print(f"Stage 1 flagged {len(X_anomaly_np):,} anomalies")

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

    print(f"Stage 2 produced {len(X_confident):,} confident predictions (>= {confidence_threshold})")

    classes_present = np.unique(pred_confident)
    per_class = max(1, n_samples // len(classes_present))
    chosen = []
    for cls in classes_present:
        cls_idx = np.where(pred_confident == cls)[0]
        n_pick = min(per_class, len(cls_idx))
        chosen.extend(rng.choice(cls_idx, size=n_pick, replace=False).tolist())
    if len(chosen) < n_samples:
        remaining = list(set(range(len(X_confident))) - set(chosen))
        extra = min(n_samples - len(chosen), len(remaining))
        chosen.extend(rng.choice(remaining, size=extra, replace=False).tolist())
    sample_idx = np.array(chosen[:n_samples])
    X_explain  = X_confident[sample_idx]
    y_explain  = pred_confident[sample_idx]
    c_explain  = conf_confident[sample_idx]
    r_explain  = recon_confident[sample_idx]

    bg_idx        = rng.choice(len(X_anomaly_np), size=min(n_background, len(X_anomaly_np)), replace=False)
    X_background  = X_anomaly_np[bg_idx]

    print("Computing SHAP values via TreeExplainer...")
    explainer   = shap.TreeExplainer(clf, data=X_background, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_explain)

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
            n_features = input_dim
            sv_full = np.array(shap_values[i]).flatten()
            sv = sv_full[class_idx * n_features:(class_idx + 1) * n_features]

        top5_idx = np.argsort(np.abs(sv))[::-1][:5]
        top_features = []
        for fi in top5_idx.tolist():
            fname = feature_names[fi] if fi < len(feature_names) else f"feature_{fi}"
            top_features.append({
                "feature": fname,
                "shap_value": float(sv[fi]),
                "direction": "increases risk" if sv[fi] > 0 else "decreases risk",
            })

        recon_err  = float(r_explain[i])
        severity   = get_severity(conf, class_name)
        top_pos    = [f for f in top_features if f["shap_value"] > 0]
        top_neg    = [f for f in top_features if f["shap_value"] < 0]
        def _feature_phrase(f):
            raw = float(X_explain[i][feature_names.index(f["feature"])])
            verb = "elevated" if raw > 0 else "low"
            return f"{verb} {f['feature']}"
        pos_phrases = ", ".join(_feature_phrase(f) for f in top_pos[:2])
        neg_names   = ", ".join(f["feature"] for f in top_neg[:1])
        plain_eng   = f"Flagged as {class_name} primarily due to {pos_phrases}"
        if neg_names:
            plain_eng += f", with {neg_names} contributing against this classification"

        print(f"\nSample {i+1} | Predicted: {class_name} | Confidence: {conf:.3f} | Severity: {severity}")
        print(f"  VAE reconstruction error : {recon_err:.6f}  (threshold={threshold:.6f})")
        print(f"  Verdict                  : {plain_eng}")
        print(f"  Top contributing features:")
        for rank, fi in enumerate(top5_idx.tolist()):
            f = top_features[rank]
            raw_val = float(X_explain[i][fi])
            direction_symbol = "+" if f["shap_value"] > 0 else "-"
            print(f"    [{direction_symbol}] {f['feature']:35s} value={raw_val:+.4f}  (SHAP={f['shap_value']:+.4f})  ({'contributing' if f['shap_value'] > 0 else 'reducing risk'})")

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
    parser = argparse.ArgumentParser(description="SHAP explanations for Stage 2 classifier on CICIDS2017.")
    parser.add_argument("--data-dir",             default="data/cicids2017")
    parser.add_argument("--vae-run-dir",          default="outputs/default")
    parser.add_argument("--clf-run-dir",          default="outputs/clf_default")
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--n-samples",            type=int,   default=5)
    parser.add_argument("--n-background",         type=int,   default=100)
    parser.add_argument("--max-data-rows",        type=int,   default=200000,
                        help="Subsample this many rows from attack-day data to keep memory/runtime manageable")
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
        max_data_rows=args.max_data_rows,
        random_state=args.random_state,
        device=args.device,
    )