# Two-Stage NIDS - CICIDS2017

A two-stage Network Intrusion Detection System built for SOC use, targeting two concrete problems: alert fatigue and attack triage. Stage 1 filters traffic down to a manageable anomaly set. Stage 2 classifies what those anomalies actually are.

This repository ports the original NSL-KDD pipeline to CICIDS2017, a larger and more operationally representative dataset.

---

## Architecture

**Stage 1 - Variational Autoencoder (anomaly detection)**

Trained exclusively on Monday's benign traffic (529K flows). The VAE learns a compact representation of normal network behaviour. At inference, flows whose reconstruction error exceeds a calibrated threshold are flagged as anomalous. Only flagged samples proceed to Stage 2, which keeps analyst load low.

**Stage 2 - Random Forest (attack classification)**

A multi-class Random Forest classifies VAE-flagged anomalies into specific attack categories. Predictions below a confidence threshold route to a `novel_anomaly` bucket for manual review. This also catches rare classes (Infiltration, Heartbleed) where the model lacks enough training examples to classify confidently.

---

## Pipeline features

- Feature audit on Monday data: near-zero-variance columns dropped, highly skewed columns log1p-transformed
- Outlier clipping to p1-p99 before scaling, preventing StandardScaler distortion
- Threshold calibration: ROC-optimal (Youden's J), percentile, and sigma methods
- Confidence-based abstention routing low-confidence predictions to `novel_anomaly`
- SHAP explanations with severity scoring and plain-English verdicts
- Cost-weighted detection analysis quantifying FP analyst load vs FN security cost
- Inference latency profiling per stage
- Dynamic threshold recalibration on fresh Monday traffic windows
- Stratified SHAP sampling for diverse per-class explanations

---

## Dataset

CICIDS2017 is released by the **Canadian Institute for Cybersecurity (CIC) at the University of New Brunswick**. It is publicly available for researchers at [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html).

### License

Freely available for research use. The dataset was designed specifically for the development and evaluation of intrusion detection systems, making it directly appropriate for this project. Redistribution requires citation of the original paper:

> Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", ICISSP, Portugal, 2018.

The dataset was purpose-built for IDS development and evaluation, making it directly suitable for this work.

### Data split

| Split | Source | Records | Purpose |
|---|---|---|---|
| VAE train | Monday only | 529,481 | Normal traffic baseline |
| Stage 2 train/val/test | Tuesday to Friday | 2,298,395 | Attack classification and evaluation |

### Attack categories after mapping

| Category | Source labels |
|---|---|
| DoS | DoS Hulk, GoldenEye, slowloris, Slowhttptest, DDoS |
| Probe | PortScan |
| BruteForce | FTP-Patator, SSH-Patator |
| WebAttack | Web Attack Brute Force, XSS, SQL Injection |
| Bot | Bot |
| Infiltration | Infiltration (rare) |
| Heartbleed | Heartbleed (rare) |

---

## Experiment log

All experiments changed one variable against the locked baseline. Four sweeps returned negative results, confirming the baseline is well-tuned.

### Stage 1 - VAE experiments

| Variable | Values tested | Winner | Notes |
|---|---|---|---|
| Latent dim | 8, 16, **32**, 64 | **32 (baseline)** | latent_dim=16 wins PR-AUC by 0.7% but loses precision and alert rate |
| Beta ceiling | 0.1, 0.5, **1.0** | **1.0 (baseline)** | Lower beta degrades precision; KL regularisation is doing useful work |
| Hidden dims | 128-64, 256-128-64, 512-256, **256-128** | **256-128 (baseline)** | No alternative improves on any metric |
| Anomaly scoring | **MSE**, ELBO | **MSE (baseline)** | ELBO produces 54% alert rate vs 26% for MSE |
| Threshold method | **roc_optimal**, p95, p99, sigma | **roc_optimal** primary, p95 documented as SOC alternative | p95 halves alert rate at the cost of 17 points of recall |

### Stage 2 - classifier

Random Forest with `class_weight='balanced'`, n_estimators=200. Near-perfect validation and test accuracy across all major classes. Rare classes (Infiltration, Heartbleed) correctly abstain via the confidence threshold in most cases.

---

## Baseline results (locked)

Run ID: `vae_20260405_213312`

### Stage 1 - VAE

| Metric | Value |
|---|---|
| ROC-AUC | 0.9031 |
| PR-AUC | 0.7526 |
| F1 | 0.7328 |
| Precision | 0.7081 |
| Recall | 0.7593 |
| Alert rate | 0.2596 |
| FP count | 34,834 |

Configuration: latent_dim=32, hidden_dims=(256, 128), beta_max=1.0, MSE scoring, ROC-optimal threshold, early stopping patience=10.

### Stage 2 - classifier

| Metric | Value |
|---|---|
| Macro F1 (test set) | 0.999 |
| Weighted F1 (test set) | ≈1.00 |
| Novel anomaly rate | ~0.01% |
| CV macro F1 mean (5-fold) | 0.9859 |
| CV macro F1 std (5-fold) | 0.0071 |

5-fold stratified CV was run on the training split only (n=400,719). The tight std of 0.0071 confirms results are stable across splits and not an artefact of a favourable random seed. Infiltration is the only volatile class across folds, expected given only 36 training samples.

### Cost-weighted analysis

| Contribution | Value |
|---|---|
| FN security cost | 155 (0.4%) |
| FP analyst cost | 36,639 (99.6%) |
| Total cost C | 36,794 |

Cost weights: DoS=20, Probe=5, BruteForce=15, WebAttack=20, Bot=25, Infiltration=50, Heartbleed=50. The FP dominance motivates Stage 1 precision as the primary optimisation target.

---

## Repository structure

```
src/
  data_utils.py              Stage 1 preprocessing (audit, log1p, clip, scale)
  data_utils_stage2.py       Stage 2 attack-category mapping and splits
  vae_model.py               VAE architecture and loss function
  train_vae.py               Stage 1 training with early stopping and LR scheduling
  evaluate_vae.py            Stage 1 evaluation and threshold calibration
  train_classifier.py        Stage 2 training
  evaluate_classifier.py     Stage 2 evaluation with novel-routing diagnostics
  run_pipeline.py            End-to-end pipeline with latency profiling
  explain_prediction.py      Stratified SHAP explanations with severity scoring
  recalibrate_threshold.py   Threshold recalibration on Monday traffic windows
  cost_metrics.py            Cost-weighted FP/FN analysis
  plot_recon_distribution.py Reconstruction error distribution plot
  thresholding.py            Threshold calibration utilities
data/cicids2017/             CICIDS2017 CSVs (not included)
outputs/                     Run directories (vae_*, clf_*)
figures/                     Exported plots for reporting
```

---

## Usage

**Train Stage 1:**
```powershell
.\.venv\Scripts\python.exe src\train_vae.py --data-dir data/cicids2017
```

**Evaluate Stage 1:**
```powershell
.\.venv\Scripts\python.exe src\evaluate_vae.py --data-dir data/cicids2017 --run-dir outputs/vae_<timestamp>
```

**Train Stage 2** (requires trained VAE):
```powershell
.\.venv\Scripts\python.exe src\train_classifier.py --data-dir data/cicids2017 --vae-run-dir outputs/vae_<timestamp>
```

**Evaluate Stage 2:**
```powershell
.\.venv\Scripts\python.exe src\evaluate_classifier.py --data-dir data/cicids2017 --clf-run-dir outputs/clf_<timestamp>
```

**Run full pipeline:**
```powershell
.\.venv\Scripts\python.exe src\run_pipeline.py --data-dir data/cicids2017 --vae-run-dir outputs/vae_<timestamp> --clf-run-dir outputs/clf_<timestamp>
```

**SHAP explanations** (stratified, 2 samples per class):
```powershell
.\.venv\Scripts\python.exe src\explain_prediction.py --data-dir data/cicids2017 --vae-run-dir outputs/vae_<timestamp> --clf-run-dir outputs/clf_<timestamp> --n-samples 14
```

**Cost-weighted analysis** (run after both evaluate scripts):
```powershell
.\.venv\Scripts\python.exe src\cost_metrics.py --clf-metrics outputs/clf_<timestamp>/metrics.json --vae-metrics outputs/vae_<timestamp>/metrics.json
```

**Reconstruction error distribution plot:**
```powershell
mkdir figures
.\.venv\Scripts\python.exe src\plot_recon_distribution.py --data-dir data/cicids2017 --vae-run-dir outputs/vae_<timestamp> --save-path figures/recon_dist.png
```

**Threshold recalibration** (on fresh Monday traffic):
```powershell
.\.venv\Scripts\python.exe src\recalibrate_threshold.py --data-dir data/cicids2017 --vae-run-dir outputs/vae_<timestamp>
```

---

## Requirements

Python 3.10+, PyTorch, scikit-learn, scipy, pandas, numpy, SHAP, matplotlib, tqdm. Tested on Windows with CPU. CUDA supported via `--device cuda`.

---

## Related work

The earlier NSL-KDD implementation is locked and lives in a separate repository. That pipeline established the core two-stage architecture; this repository adapts it to a larger, more modern dataset with a fully numeric feature space and a richer attack taxonomy.
