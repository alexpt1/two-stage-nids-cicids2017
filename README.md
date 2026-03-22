# Two-Stage Network Intrusion Detection System

A two-stage NIDS built on the NSL-KDD dataset. Stage 1 uses a Variational Autoencoder (VAE) trained exclusively on normal traffic to flag anomalies. Stage 2 uses a Random Forest classifier to categorise flagged traffic into known attack types, with a confidence-based abstention mechanism for novel or ambiguous threats.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Stage 1 - Train VAE](#stage-1---train-vae)
  - [Stage 1 - Evaluate VAE](#stage-1---evaluate-vae)
  - [Stage 2 - Train Classifier](#stage-2---train-classifier)
  - [Stage 2 - Evaluate Classifier](#stage-2---evaluate-classifier)
  - [Full Pipeline](#full-pipeline)
- [Results](#results)
  - [Stage 1 Results](#stage-1-results)
  - [Stage 2 Results](#stage-2-results)
  - [End-to-End Pipeline](#end-to-end-pipeline)
- [Optimisation Log](#optimisation-log)
  - [Stage 2 Classifier Optimisation](#stage-2-classifier-optimisation)
  - [Stage 1 VAE Optimisation](#stage-1-vae-optimisation)
- [Limitations](#limitations)

---

## Overview

```
Network Traffic
      │
      ▼
┌─────────────────────┐
│   Stage 1: VAE      │  ← Trained on normal traffic only
│   Anomaly Detector  │
└─────────────────────┘
      │
      ├── recon_error ≤ threshold ──→  ✅ NORMAL  (no alert)
      │
      └── recon_error > threshold ──→  ⚠️  ANOMALY
                                             │
                                             ▼
                                  ┌─────────────────────┐
                                  │  Stage 2: Random    │
                                  │  Forest Classifier  │
                                  └─────────────────────┘
                                             │
                               ┌─────────────┴─────────────┐
                        confidence ≥ 0.6             confidence < 0.6
                               │                           │
                        Attack category              🔍 novel_anomaly
                    (DoS / Probe / R2L / U2R)         (flag for review)
```

---

## Architecture

### Stage 1 - Variational Autoencoder

- Trained exclusively on **normal traffic** (unsupervised)
- Learns a compressed latent representation of normal network behaviour
- At inference, anomalies produce high reconstruction error
- Threshold calibrated using **ROC-optimal (Youden's J)** on validation set
- KL annealing applied over first 10 epochs (`beta = epoch / 10`)
- Logvar clamped to `[-4.0, 4.0]` for training stability
- Free bits regularisation (`min KL = 0.5`) to prevent posterior collapse

| Hyperparameter | Value |
|---|---|
| Hidden dims | 128 → 64 |
| Latent dim | 32 |
| Epochs | 20 |
| Batch size | 256 |
| Optimiser | Adam (lr=1e-3) |
| Threshold method | ROC-optimal |

### Stage 2 - Random Forest Classifier

- Operates **only on VAE-flagged anomalies**
- Trained on labelled attack traffic from NSL-KDD training set
- Outputs attack category or abstains as `novel_anomaly` if `max(predict_proba) < 0.6`
- Preprocessor reused directly from VAE checkpoint, no separate fitting

| Hyperparameter | Value |
|---|---|
| n_estimators | 200 |
| class_weight | balanced |
| Confidence threshold | 0.6 |

---

## Dataset

**NSL-KDD** — an improved version of the KDD Cup 1999 dataset.

- `KDDTrain+.txt` — training set (~125,973 records)
- `KDDTest+.txt` — test set (~22,544 records)

### Attack Category Mapping

| Category | Example Attacks |
|---|---|
| `DoS` | neptune, smurf, back, teardrop |
| `Probe` | ipsweep, portsweep, nmap, satan |
| `R2L` | guess_passwd, ftp_write, phf, spy |
| `U2R` | buffer_overflow, rootkit, perl |

> **Note:** KDDTest+ contains 178 records with attack labels not present in training. These are correctly handled as `novel_anomaly` at inference time.

Download the dataset from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/nsl.html) and place files in `data/nsl_kdd/`.

---

## Project Structure

```
vae-anomaly-detection/
├── src/
│   ├── vae_model.py                  # VAE architecture and loss function
│   ├── train_vae.py                  # VAE training script
│   ├── evaluate_vae.py               # VAE evaluation, metrics, ROC, thresholding
│   ├── data_utils.py                 # NSL-KDD loading and preprocessing
│   ├── thresholding.py               # Percentile, sigma, and ROC-optimal threshold methods
│   ├── plot_recon_distribution.py    # Reconstruction error visualisation
│   ├── data_utils_stage2.py          # Attack-only data prep for classifier
│   ├── train_classifier.py           # Random Forest training
│   ├── evaluate_classifier.py        # Classifier metrics and novel anomaly rate
│   └── run_pipeline.py               # End-to-end Stage 1 to Stage 2 inference
├── data/
│   └── nsl_kdd/
│       ├── KDDTrain+.txt
│       └── KDDTest+.txt
├── outputs/
│   ├── vae_<run_id>/
│   │   ├── model.pt                  # Saved weights + preprocessor
│   │   ├── config.json               # Hyperparameters
│   │   ├── train_log.json            # Per-epoch losses
│   │   ├── threshold.json            # Calibrated anomaly threshold
│   │   └── metrics.json              # Full evaluation results
│   └── clf_<run_id>/
│       ├── model.pkl                 # Saved Random Forest
│       ├── label_encoder.pkl         # Category to integer mapping
│       ├── config.json               # Training configuration
│       └── metrics.json              # Evaluation results
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

All commands assume you are in the project root with the virtual environment activated.

### Stage 1 - Train VAE

```powershell
.\.venv\Scripts\python.exe src\train_vae.py `
  --data-dir data/nsl_kdd `
  --outputs-root outputs
```

### Stage 1 - Evaluate VAE

```powershell
$VAE_RUN=(Get-ChildItem outputs -Directory | Where-Object { $_.Name -like "vae_*" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName

.\.venv\Scripts\python.exe src\evaluate_vae.py `
  --data-dir data/nsl_kdd `
  --run-dir "$VAE_RUN"
```

### Stage 2 - Train Classifier

```powershell
.\.venv\Scripts\python.exe src\train_classifier.py `
  --data-dir data/nsl_kdd `
  --outputs-root outputs `
  --vae-run-dir "$VAE_RUN"
```

### Stage 2 - Evaluate Classifier

```powershell
$CLF_RUN=(Get-ChildItem outputs -Directory | Where-Object { $_.Name -like "clf_*" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName

.\.venv\Scripts\python.exe src\evaluate_classifier.py `
  --data-dir data/nsl_kdd `
  --clf-run-dir "$CLF_RUN"
```

### Full Pipeline

```powershell
.\.venv\Scripts\python.exe src\run_pipeline.py `
  --data-dir data/nsl_kdd `
  --vae-run-dir "$VAE_RUN" `
  --clf-run-dir "$CLF_RUN" `
  --n-samples 2000
```

---

## Results

### Stage 1 Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.9502 |
| PR-AUC | 0.9548 |
| Precision | 0.9356 |
| Recall | 0.7721 |
| F1-score | 0.8461 |
| FNR | 0.2279 |
| Recall @ FPR 5% | 0.7321 |
| Alert rate | 46.98% |
| Threshold (ROC-optimal) | 0.072078 |

### Stage 2 Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| DoS | 0.97 | 0.955 | 0.962 |
| Probe | 0.77 | 0.925 | 0.840 |
| R2L | 0.99 | 0.693 | 0.817 |
| U2R | 0.75 | 0.846 | 0.795 |
| **Macro avg** | **0.87** | **0.85** | **0.85** |

- Novel anomaly rate (confidence < 0.6): **22.6%**
- 178 test samples with unseen attack labels correctly abstained on

### End-to-End Pipeline

#### 2,000-Sample Draw from KDDTest+

| Verdict | Count | % |
|---|---|---|
| normal | 1,059 | 52.9% |
| DoS | 533 | 26.7% |
| Probe | 204 | 10.2% |
| novel_anomaly | 182 | 9.1% |
| R2L | 21 | 1.1% |
| U2R | 1 | 0.1% |

**Binary accuracy (normal vs attack): 84.6%**

#### Full KDDTest+ (22,544 samples)

| Verdict | Count | % |
|---|---|---|
| normal | 12,039 | 53.4% |
| DoS | 5,896 | 26.2% |
| Probe | 2,383 | 10.6% |
| novel_anomaly | 1,980 | 8.8% |
| R2L | 226 | 1.0% |
| U2R | 20 | 0.1% |

**Binary accuracy (normal vs attack): 83.9%**

The results are consistent across both sample sizes, confirming the 2,000-sample draw was representative and the model generalises stably across the full test set. The 1,980 novel_anomaly abstentions include the 178 unseen attack labels plus ambiguous samples the RF correctly declined to commit to.

---

## Optimisation Log

### Stage 2 Classifier Optimisation

The following approaches were systematically evaluated against the Stage 2 baseline (RF 200, `balanced`):

| Approach | Macro F1 | R2L Recall | Outcome |
|---|---|---|---|
| RF 200, `balanced` ⭐ | **0.85** | **0.693** | Baseline, best overall |
| RF 500, `balanced` | 0.84 | 0.689 | Marginal degradation |
| RF 500, `balanced_subsample` | 0.82 | 0.662 | Worse |
| XGBoost 200 | 0.78 | 0.701 | Worse overall |
| SMOTE + RF 200 | 0.76 | 0.650 | Worse, U2R collapsed |
| RandomizedSearchCV (best: 500, depth=40) | 0.84 | 0.666 | Near-baseline, slower |
| Probability calibration (isotonic) | 0.81 | 0.696 | Novel rate halved but F1 dropped |
| Feature selection (77 to 39 features) | 0.83 | 0.621 | Removed useful R2L features |

The baseline was optimal in every trial. R2L recall is limited by NSL-KDD's feature overlap between R2L and Probe, a known dataset-level constraint.

### Stage 1 VAE Optimisation

The following experiments were run iteratively to improve VAE anomaly detection performance:

| Experiment | ROC-AUC | PR-AUC | F1 | Outcome | Note |
|---|---|---|---|---|---|
| Original baseline (256→128, latent=16, L1, b=0.1) | 0.915 | 0.929 | 0.756 | Baseline | Starting point |
| Beta ceiling: 0.1 to 1.0, free bits floor, warm-up fix | 0.935 | 0.941 | 0.837 | Kept | Fixed posterior collapse, KL ratio 0.000 to 0.70 |
| Latent dim: 16 to 8 | 0.933 | 0.940 | 0.682 | Reverted | Under-compression, below baseline |
| Latent dim: 16 to 32 | 0.938 | 0.943 | 0.706 | Kept | Best of sweep, sweet spot |
| Latent dim: 16 to 64 | 0.934 | 0.940 | 0.662 | Reverted | Over-parameterised, attacks reconstructed too well |
| Reconstruction loss: L1 to MSE | 0.952 | 0.956 | 0.837 | Kept | Largest single gain, quadratic penalty sharpens gap |
| Hidden dims: 256→128→64 (deeper) | 0.951 | 0.955 | 0.837 | Reverted | Marginal, narrow 128→64 wins instead |
| Hidden dims: 256→128 to 128→64 | 0.955 | 0.959 | 0.833 | Kept | Tighter normal representation, simpler wins |
| Training epochs: 20 to 40 | 0.951 | 0.955 | 0.834 | Reverted | Drift from epoch 20+, KL dominates late |
| Anomaly score: MSE to ELBO | 0.880 | 0.922 | 0.781 | Reverted | Free bits inflate KL uniformly, no discriminative signal |
| Threshold: percentile to sigma (mu+3sigma) | 0.951 | 0.956 | 0.265 | Reverted | Long tail in normal errors, threshold overshoots |
| Threshold: percentile to ROC-optimal | 0.951 | 0.955 | 0.846 | Kept | Best F1, maximises Youden's J on val set |

---

## Limitations

- **Dataset age:** NSL-KDD is derived from 1999 traffic. Modern attack patterns differ significantly, limiting real-world generalisation.
- **R2L recall ceiling:** R2L and Probe share overlapping features in NSL-KDD's feature space. No classifier configuration meaningfully resolved this.
- **Unseen attack types:** KDDTest+ contains 178 records with labels absent from training. These are routed to `novel_anomaly`, which is the correct behaviour but means they are never classified.
- **Static threshold:** The VAE threshold is calibrated once at evaluation time. A production system would require periodic recalibration as traffic distributions shift.
- **Natural next step:** Retraining on CICIDS2017 would provide more realistic and contemporary attack coverage.
