# FLabBench - Chemotherapy Side Effect Prediction Pipeline

This repository provides a reproducible pipeline to:

- Extract cohorts from **MIMIC-IV** and a **Code dataset from a MII node** (Erlangen University Hospital, UKEr) for patients with cancer diagnoses undergoing chemotherapy.  
- Train **classical machine learning** and **temporal deep learning** models to predict chemotherapy-induced side effects using longitudinal lab data and demographic features.

## Fork notice

This repository (`ChemoTreeVsDLGAM`) is a **forked extension of [bionetslab/ChemoTreeVsDL](https://github.com/bionetslab/ChemoTreeVsDL)**. The original repo benchmarks tree-based and temporal deep learning models on the same cohorts/features; this fork's goal is to find **explainable (glass-box) models that can compete with tree-based models** (Random Forest, Gradient Boosting, XGBoost, CatBoost) on predictive performance while remaining directly interpretable — i.e. models whose predictions decompose into per-feature contributions instead of requiring post-hoc explainers (SHAP, LIME, etc.) on a black box.

Everything under `ml_model_training/model_wrappers/` beyond the original tree/linear baselines (GAM, EBM, GAMINET, IGANN, NAM, DGAM, MGAM, SSTGAM, HE-EBM, IG-EBM) was added in this fork, alongside supporting feature-engineering (`--feat_type A`, `--lookback_days`) and feature-selection (`--selector_type`) machinery to make those models competitive on the same tabular representation the tree-based baselines use.

## Models

**Classical baseline models** (from the original repo)
- Tree-based: Random Forest (RF), Gradient Boosting (GB), Extreme Gradient Boosting (XGB), CatBoost (CatB)
- Other: Logistic Regression (LR)

**Temporal models** (from the original repo)
- Regular time series: GRU, LSTM, Temporal Convolutional Network (TCN), SAnD  
- Irregular time series: GRU-D, InterpNet, STraTS

**Explainable models added in this fork**

All of the following are trained via `ml_model_training/ml_main.py --model_type <name>` on the same tabular feature matrices as the classical baselines, so their AUC-ROC/AUC-PRC is directly comparable.

| `--model_type` value | Model | Explainability mechanism | Wrapper |
|---|---|---|---|
| `GAM` | Logistic GAM (pygam) | Additive smooth-spline / linear terms per feature | [`gam_wrapper.py`](ml_model_training/model_wrappers/gam_wrapper.py) |
| `EBM` | Explainable Boosting Machine | Additive shape functions learned via cyclic gradient boosting (native missing-value bins) | [`ebm_wrapper.py`](ml_model_training/model_wrappers/ebm_wrapper.py) |
| `GAMINET` | GAMINet | Neural additive model with explicit pairwise interaction subnets, constrained for clarity | [`gaminet_wrapper.py`](ml_model_training/model_wrappers/gaminet_wrapper.py) |
| `IGANN` | Interpretable Generalized Additive Neural Network | Per-feature ELM (extreme learning machine) shape functions combined via boosting | [`igann_wrapper.py`](ml_model_training/model_wrappers/igann_wrapper.py) |
| `NAM` | Neural Additive Model | One small MLP (ExU activation) per feature, summed to a logit | [`nam_wrapper.py`](ml_model_training/model_wrappers/nam_wrapper.py) |
| `DGAM` | Distributed-Lag GAM (thin wrapper on R `mvgam::mvgam()`) | Tensor-product distributed-lag smooth `te(Value, Lag)` per lab, per mvgam's own case study (Clark & Wells 2022) | [`dl_gam_wrapper.py`](ml_model_training/model_wrappers/dl_gam_wrapper.py) |
| `MGAM` | Missingness-Aware GAM | Sparse L0L2 logistic regression over value + missingness-gated interaction terms | [`mgam_wrapper.py`](ml_model_training/model_wrappers/mgam_wrapper.py) |
| `SSTGAM` | Shared-Shape Temporal GAM | Shape functions shared across time bins per feature (spline or tree fit backend) | [`sstgam_wrapper.py`](ml_model_training/model_wrappers/sstgam_wrapper.py) |
| `HE-EBM` | Hierarchical Expert-Based EBM | Five specimen/category EBM experts combined by a learned non-negative logistic combiner | [`he_ebm_wrapper.py`](ml_model_training/model_wrappers/he_ebm_wrapper.py) |
| `IG-EBM` | Interaction-Grouped EBM | EBM with FAST auto-search replaced by forced same-itemid, cross-day-bin interaction pairs (tests whether hand-specifying temporal interactions beats greedy search) | [`ig_ebm_wrapper.py`](ml_model_training/model_wrappers/ig_ebm_wrapper.py) |  

## Cohorts

- Patients with a cancer diagnosis and at least one chemotherapy procedure.  
- **Aplasia:** defined by transfusion procedures or low absolute neutrophil count.  
- **Neutropenic fever:** concurrent neutropenia and fever diagnoses.  
- **Prediction target:** onset of the condition within the prediction window (45 days after discharge for aplasia, 30 days for neutropenic fever) and before the next chemotherapy cycle.  
- **Observation window:** 14 days prior to discharge.  

## Structure

- **config/** – Configuration files
  - `constants.py` – Shared constants and paths
  - `model_params.yaml` – Grid search parameters for classical models
  - `ml_config_params_best.yaml` – Best parameters/cohort for classical models
  - `ts_config_params.yaml` – Grid search parameters for neural network models
  - `ts_config_params_best.yaml` – Best parameters/cohort for neural network models

- **ml_model_training/** – Training scripts for classical models
- **ts_model_training/** – Training scripts for deep learning models


## Usage

1. Create the environment:

```bash
conda env create -f environment.yml
conda activate flabnet_ml_pipeline_env
```
### 2. Train models

#### Classical models
Run the training script with:

```bash
bash train_ml_models.sh [mimic|uker] [prepare|train|full_run]
```

- mimic – for MIMIC-IV cohorts
- uker – for Erlangen University Hospital cohorts

Workflow options:
- prepare – prepare inputs
- train – train models on prepared inputs
- full_run – prepare inputs and train models

Example:

```bash
bash train_ml_models.sh mimic full_run
```

#### Running `ml_main.py` directly

`train_ml_models.sh` is a thin loop around `ml_model_training/ml_main.py`; for a single cohort/model/feat_type combination (e.g. to try one of the explainable models above) it's usually simpler to call it directly:

```bash
python -m ml_model_training.ml_main \
    --dataset MIMIC_IV \
    --cohort mimic_cohort_NF_30_days \
    --model_type EBM \
    --features LAB DEMO \
    --feature_method concatenate \
    --oversampling minority \
    --grid_search 0 \
    --feature_selection 1 \
    --top_k_features 250 \
    --selector_type tree \
    --lookback_days 7 \
    --feat_type VMD \
    --agg_interval 24 \
    --num_folds 5 \
    --prefix example_run \
    --action prepare   # build the feature matrices for every fold first

python -m ml_model_training.ml_main \
    --dataset MIMIC_IV \
    --cohort mimic_cohort_NF_30_days \
    --model_type EBM \
    --features LAB DEMO \
    --feature_method concatenate \
    --oversampling minority \
    --grid_search 0 \
    --feature_selection 1 \
    --top_k_features 250 \
    --selector_type tree \
    --lookback_days 7 \
    --feat_type VMD \
    --agg_interval 24 \
    --num_folds 5 \
    --prefix example_run \
    --load_data 1 \
    --action train      # then train on the cached matrices
```

`--action prepare` only builds and caches the per-fold feature matrices (no training); `--action train` trains `--model_type` on them (`--load_data 1` reuses the cache from `prepare` instead of rebuilding it). See `python -m ml_model_training.ml_main --help` for the full argument list.

#### CLI keywords added in this fork

These arguments to `ml_main.py` were added alongside the explainable models, to control the feature engineering and column-pruning they need to be competitive with the tree-based baselines:

| Argument | Type / choices | Why it's needed |
|---|---|---|
| `--feat_type A` | one of `standard, V, M, D, VMD, VM, VD, MD, A` | Adds the **Aggregated** feature type: a decorrelated 9-feature-per-lab summary (`median, mean, min, max, std, latest_zscore, robust_trend, observed_fraction`) over the whole observation window, instead of one column per (lab, day, VMD-channel). Several of the additive models (GAM, EBM, IGANN, NAM) train faster and generalize better on this compact representation than on the raw per-day VMD matrix. |
| `--lookback_days N` | `int`, default `None` | Restricts every feature type (VMD, `A`, ...) to only the last `N` days before discharge, dropping earlier bins entirely before any feature is computed. Lets you test whether a model's signal is concentrated near discharge, and shrinks the column count for models (GAM, MGAM, DGAM) that scale poorly with the number of daily bins. `None` (default) keeps the full observation window. |
| `--feature_selection {0,1}` | boolean-as-int | Enables/disables feature selection (top-k column pruning) before training. `1` is effectively required for the wide VMD/`A` matrices to keep the additive models' term count tractable. |
| `--top_k_features K` | `int`, default `250` | Number of columns kept by the feature selector when `--feature_selection 1` is set. Bounds the number of shape functions/terms the explainable models have to fit (and, for GAM/DGAM, the PIRLS/REML design matrix size). |
| `--selector_type` | `none, tree, mi, correlation, xgb_gain, lasso` | Which ranking method picks the top-`k` columns: `tree` = RandomForest MDI importance (default), `mi` = mutual information, `correlation` = \|Pearson r\| with the target, `xgb_gain` = XGBoost gain importance, `lasso` = L1-logistic-regression coefficients, `none` = passthrough (no pruning). Different explainable models respond differently to *which* features survive selection, so this is swappable per experiment rather than hard-coded to one ranking method. |

#### Temporal deep models
To run temporal deep models on already prepared inputs:

```bash
bash train_ts_models.sh
```
