# Two-Stage NIDS - CICIDS2017

A two-stage Network Intrusion Detection System built for SOC use, with the dual goals of reducing alert fatigue and improving triage. This repository ports the original NSL-KDD pipeline to the larger and more modern CICIDS2017 dataset.

## Architecture

**Stage 1 - Variational Autoencoder (anomaly detection).** Trained exclusively on Monday's benign traffic, the VAE flags traffic whose reconstruction error exceeds a calibrated threshold. Only anomaly-flagged samples proceed to Stage 2, dramatically reducing analyst load.

**Stage 2 - Random Forest (attack classification).** A multi-class Random Forest categorises VAE-flagged anomalies into specific attack types (DoS, Probe, BruteForce, WebAttack, Bot). Predictions falling below a confidence threshold are routed to a `novel_anomaly` bucket for analyst review, which also captures rare classes (Infiltration, Heartbleed) the model has insufficient examples to learn.

## Pipeline features

- Stage 1 anomaly detection with calibrated thresholds (ROC-optimal, percentile, sigma)
- Stage 2 attack classification with confidence-based abstention
- SHAP-based explainability with severity scoring
- Cost-weighted detection metrics
- Inference latency profiling
- Dynamic threshold recalibration on new traffic windows

## Dataset

CICIDS2017 - eight CSV files spanning Monday to Friday, ~2.8 million flow records, 77 numeric features (78 after column de-duplication).

- **Monday:** 529K benign records → VAE training
- **Tuesday–Friday:** 2.3M mixed records → Stage 2 training and end-to-end evaluation

## Repository structure

```
src/
  data_utils.py            Stage 1 preprocessing (audit, log1p, clip, scale)
  data_utils_stage2.py     Stage 2 attack-category mapping and splits
  vae_model.py             VAE architecture and loss
  train_vae.py             Stage 1 training
  evaluate_vae.py          Stage 1 evaluation and threshold calibration
  train_classifier.py      Stage 2 training
  evaluate_classifier.py   Stage 2 evaluation
  run_pipeline.py          End-to-end pipeline with latency profiling
  explain_prediction.py    SHAP explanations for Stage 2 predictions
  recalibrate_threshold.py Threshold recalibration on new traffic windows
  cost_metrics.py          Cost-weighted detection analysis
  thresholding.py          Threshold calibration methods
data/cicids2017/           CICIDS2017 CSVs (not included)
outputs/                   Run directories (vae_*, clf_*)
```

## Usage

Train Stage 1:

```powershell
.\.venv\Scripts\python.exe src\train_vae.py --data-dir data/cicids2017
```

Evaluate Stage 1:

```powershell
.\.venv\Scripts\python.exe src\evaluate_vae.py --data-dir data/cicids2017 --run-dir outputs/vae_<timestamp>
```

Train Stage 2 (requires a trained VAE):

```powershell
.\.venv\Scripts\python.exe src\train_classifier.py --data-dir data/cicids2017 --vae-run-dir outputs/vae_<timestamp>
```

## Baseline (locked)

Run ID: `vae_20260405_213312`

| Metric | Value |
|---|---|
| ROC-AUC | 0.9031 |
| PR-AUC | 0.7526 |
| F1 | 0.7328 |
| Precision | 0.7081 |
| Recall | 0.7593 |
| Alert rate | 0.2596 |

Configuration: latent_dim=32, hidden_dims=(256, 128), β_max=1.0, MSE scoring, ROC-optimal threshold.

## Requirements

Python 3.10+, PyTorch, scikit-learn, scipy, pandas, numpy, SHAP, tqdm. Tested on Windows with CPU; CUDA supported.

## Related work

This project is the second of two implementations. The earlier NSL-KDD version is locked and lives in a separate repository.