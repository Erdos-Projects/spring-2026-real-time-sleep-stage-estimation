# Towards Automated Sleep Analysis: Stage Classification and Apnea Prediction - Spring 2026

A team research project developing machine learning models for **real-time sleep stage classification** and **sleep apnea detection** from polysomnography (PSG) wearable signals.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Tasks](#tasks)
  - [Task 1: Sleep Apnea Detection](#task-1-sleep-apnea-detection)
  - [Task 2: Sleep Stage Classification](#task-2-sleep-stage-classification)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Models](#models)

---

## Project Overview

This project builds end-to-end ML pipelines for two clinical tasks using multi-channel physiological recordings:

1. **Sleep Apnea Detection** — Binary classification predicting apnea events 10 seconds ahead
2. **Sleep Stage Classification** — 5-class classification (Wake, N1, N2, N3, REM) at 5-second epoch resolution

Both pipelines include feature engineering, hyperparameter optimization via Optuna, model ensembling, and temporal post-processing.

---

## Dataset

**DREAMT** (located at `https://physionet.org/content/dreamt/2.1.0/.
`)

| Property | Value |
|---|---|
| Subjects | 100 |
| Sampling rate | 100 Hz |
| Epoch length | 5 seconds |
| Labels | Wake (W), N1, N2, N3, REM (R) + AHI apnea events |

**Signal channels include:**
- EEG: F3, F4, C3, C4, O1, O2, Cz, Fz, Pz
- EOG: Left/Right eye movement
- EMG: Chin electromyography
- Respiratory: Chest/abdominal effort, airflow, SaO2 (oxygen saturation)
- Cardiac: ECG-derived features, IBI (inter-beat interval)
- EDA: Electrodermal activity

**Train/Val/Test split:** 64% / 16% / 20% — stratified **across subjects** (no subject leakage).

---

## Project Structure

```
.
├── config/
│   ├── dir-config.yaml                  # Data directory paths
│   └── sleep-model.yaml                 # Full model & feature configuration
├── notebooks/
│   ├── 0.1-preprocessing.ipynb          # Raw data cleaning & feature extraction
│   ├── 1.0-apnea-detection-stratified-train-test-split.ipynb   # Apnea data splits
│   ├── 1.1-apnea-detection-tree-based-model.ipynb              # Apnea LightGBM training
│   ├── 2.1-sleepstage_basemodels.ipynb  # Sleep stage XGBoost / LightGBM
│   ├── 2.2-sleepstage_nnmodels.ipynb    # Sleep stage LSTM / BiLSTM / CNN1D
│   └── 2.3-sleepstage_test.ipynb        # Final test set evaluation
├── models/
│   ├── bilstm_seq10_f1_0.5673/          # Best BiLSTM checkpoint
│   ├── lstm_seq30_f1_0.5924/            # LSTM checkpoint
│   ├── cnn1d_seq30_f1_0.5553/           # CNN1D checkpoint
│   ├── xgb_featured/                    # XGBoost model + hyperparameters
│   └── lgb_featured/                    # LightGBM model + hyperparameters
├── src/
│   ├── sleep_data_utils.py              # Dataset loading, subject splits, label encoding
│   ├── sleep_model_utils.py             # Training loop, evaluation, feature pipeline
│   ├── data_prepatation.py              # Stratified train/val/test split creation
│   ├── feature_extraction_apnea.py      # Parallel per-epoch feature computation
│   ├── apnea_feature_tiers.py           # Curated feature tier definitions
│   ├── eda_functions.py                 # Univariate scoring, correlation analysis
│   ├── hyperparameter_optimization.py   # Optuna tuning with SMOTE
│   └── hyperparameter_optimization_nosmote.py  # Optuna tuning without SMOTE
├── Dockerfile                           # PyTorch + CUDA 12 environment
├── docker-compose.yaml
├── requirements.txt
├── setup.py
└── ruff.toml                            # Linting configuration
```

---

## Tasks

### Task 1: Sleep Apnea Detection

**Goal:** Binary classification — predict whether an apnea event will occur within the next 10 seconds (2 epochs ahead).

**Pipeline (`notebooks/1.x`):**

1. **Preprocessing** (`0.1`) — Extract 5-second epoch features from 100 Hz signals; merge with subject metadata (AHI severity, medical history).
2. **Data preparation** (`1.0`) — Stratified split by AHI severity category (normal/mild/moderate/severe). Add lagged features (top 20 features × lags (0, 25, 50, 75 secs)). Remove correlated (>0.8) and near-zero variance features.
3. **Training** (`1.1`) — LightGBM trained with Optuna (50 trials, 5-fold CV), optimizing AUC-PR. Threshold tuned on precision-recall curve to maximize F1.

**Feature configuration:**
- Lag window: up to 15 steps (75 seconds of history)
- Feature tier options: all / tier1 / tier1+2 / tier1+2+3

---

### Task 2: Sleep Stage Classification

**Goal:** 5-class classification — assign each 5-second epoch to Wake, N1, N2, N3, or REM.

**Pipeline (`notebooks/2.x`):**

1. **Feature engineering** — 227 hand-crafted features per epoch:
   - Time-domain: mean, std, min, max, range, slope per channel
   - Frequency-domain: bandpower in delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), sigma (12–15 Hz), beta (13–30 Hz), gamma (30–100 Hz), high-gamma bands
   - HRV metrics: SDNN, RMSSD, PNN50 from IBI
   - EEG channel ratios (C4/Cz, etc.)
   - Lag features: top 7 features × 3 lags → 289 total features

2. **Tree models** (`2.1`) — XGBoost and LightGBM tuned with Optuna (5-fold CV, macro F1):
   - XGBoost: 630 estimators, max_depth=6, lr=0.027
   - LightGBM: 507 estimators, num_leaves=55, lr=0.029

3. **Neural networks** (`2.2`) — Sequence models on fixed-length windows:
   - **BiLSTM:** Bidirectional LSTM, hidden_dim=32, seq_len=10
   - **LSTM:** Unidirectional LSTM, hidden_dim=64, seq_len=20
   - **CNN1D:** Temporal CNN, channels=(32,64,64), kernel=5, seq_len=30
   - Grid search over seq_len ∈ {10, 20, 30}, lr, dropout, hidden size (27 experiments)
   - Training: early stopping (patience=5), max 30 epochs

4. **Evaluation** (`2.3`) — Ensemble (averaged softmax probabilities) + mode filter post-processing.

---

## Results

### Sleep Stage Classification — Validation Set Comparison

| Model | Macro F1 | Weighted F1 | N1 F1 |
|---|---|---|---|
| **XGBoost** | **0.6677** | **0.7949** | 0.3594 |
| LightGBM | 0.6538 | 0.7905 | 0.3675 |
| LSTM | 0.5924 | 0.6966 | 0.3324 |
| BiLSTM | 0.5673 | 0.6763 | 0.2911 |
| CNN1D | 0.5553 | 0.6620 | 0.3065 |
| Logistic Regression (baseline) | 0.4301 | 0.5449 | 0.2226 |

> XGBoost achieves the best macro and weighted F1 on the validation set. N1 is the hardest stage across all models due to its transitional nature.

### Sleep Stage Classification (Test Set — 86,972 samples)

| Model | Macro F1 | Accuracy |
|---|---|---|
| XGBoost | 0.6565 | 0.7312 |
| LightGBM | 0.6579 | 0.7360 |
| Ensemble (XGB + LGB) | 0.6593 | 0.7353 |
| **Ensemble + mode filter (k=5)** | **0.6658** | **0.7408** |

**Per-stage breakdown (best model — Ensemble + mode filter):**

| Stage | F1 | Precision | Recall |
|---|---|---|---|
| Wake (W) | 0.8280 | 0.8080 | 0.8491 |
| N1 | 0.4186 | 0.3641 | 0.4924 |
| N2 | 0.8390 | 0.9082 | 0.7796 |
| N3 | 0.6493 | 0.7079 | 0.5996 |
| REM (R) | 0.5940 | 0.5409 | 0.6587 |

> N2 and Wake are the most reliably classified stages. N1 remains the hardest due to its transitional nature and overlap with neighboring stages.

### Sleep Apnea Detection (Validation Set)

Performance by max feature lag window:

| Max Feature Lag (secs) | Macro F1 | AUC-ROC | AUC-PR |
|---|---|---|---|
| 0 | 0.439 | 0.810 | 0.396 |
| 25 | 0.430 | 0.799 | 0.385 |
| 50 | 0.451 | 0.820 | 0.416 |
| **75** | **0.460** | **0.825** | **0.430** |

> Best results achieved with 75-second lag window (15 lagged epochs of history). Performance dips at 25-second lag before recovering, suggesting short-range lag features add noise while longer history is informative.

### Sleep Apnea Detection (Test Set)

**AUC-ROC = 0.786 | AUC-PR = 0.467 | Accuracy = 0.88**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| No apnea | 0.93 | 0.94 | 0.93 |
| Apnea | 0.54 | 0.50 | 0.52 |
| Macro avg | 0.73 | 0.72 | 0.72 |
| Weighted avg | 0.88 | 0.88 | 0.88 |

> Class imbalance is evident — the model achieves strong overall accuracy driven by the majority (no apnea) class. Apnea recall of 0.50 indicates room for improvement in catching true positive events.

---

## Installation

### Using Docker (recommended)

```bash
docker-compose up --build
```

The Docker image includes PyTorch with CUDA 12 support.

### Local installation

```bash
pip install -r requirements.txt
pip install -e .
```

**Key dependencies:** `torch`, `xgboost`, `lightgbm`, `optuna`, `scikit-learn`, `pandas`, `numpy`, `scipy`, `shap`, `imbalanced-learn`, `omegaconf`

---

## Usage

Run notebooks in order:

```
0.1  →  Preprocess raw PSG data and extract features
1.0  →  Create apnea train/val/test splits
1.1  →  Train LightGBM apnea detection model
2.1  →  Train XGBoost/LightGBM sleep stage models
2.2  →  Train LSTM/BiLSTM/CNN1D sleep stage models
2.3  →  Final test evaluation and ensembling
```

---

## Configuration

All model and data settings are managed via YAML files in `config/`:

**`config/dir-config.yaml`** — Data directory paths:
```yaml
base: /mnt/dreamt/
compiled: /mnt/dreamt/compiled
processed: /mnt/dreamt/processed
final: /mnt/dreamt/final
```

**`config/sleep-model.yaml`** — Full experiment configuration including:
- Subject list and train/val/test assignment
- Signal channel names and sampling rate
- Feature definitions (227 features across 7 categories)
- Sequence length and epoch boundary settings
- Preprocessing steps (imputation, scaling)

---

## Models

Trained model artifacts are saved in `models/`:

| Directory | Type | Task | Val F1 |
|---|---|---|---|
| `bilstm_seq10_f1_0.5673/` | BiLSTM | Sleep stage | 0.5673 |
| `lstm_seq30_f1_0.5924/` | LSTM | Sleep stage | 0.5924 |
| `cnn1d_seq30_f1_0.5553/` | CNN1D | Sleep stage | 0.5553 |
| `xgb_featured/` | XGBoost | Sleep stage | 0.6565 (test) |
| `lgb_featured/` | LightGBM | Sleep stage | 0.6579 (test) |

Each model directory contains the saved weights/pickle and a `metadata.json` with the best hyperparameters.

---
