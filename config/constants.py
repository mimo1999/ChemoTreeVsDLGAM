from pathlib import Path
import sys
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIMIC_DIR = '/home/jovyan/data_common/mimiciv/'
MIMIC_LABS_DIR = MIMIC_DIR +'hosp/labevents.csv.gz'
UKER_DIR = '/home/jovyan/data_common/ped-data.db'



CONFIG_DIR = Path(__file__).resolve().parents[0]
MODEL_PARAM_PATH = CONFIG_DIR / 'model_params.yaml'
BEST_PARAM_PATH = CONFIG_DIR / 'ml_config_params_best.yaml'

def load_model_params():
    with open(MODEL_PARAM_PATH, "r") as f:
        return yaml.safe_load(f)

def load_best_params():
    with open(BEST_PARAM_PATH, "r") as f:
        return yaml.safe_load(f)

MODEL_PARAMS = load_model_params()
BEST_PARAMS = load_best_params()


num_folds = 5
num_inner_folds = 3

RANDOM_SEED = 42

# Demographic covariates present in every feature matrix in addition to the
# lab-derived columns (see feature_processing/feature_generator.py:
# demo_csv=grp[['age', 'gender']]). GAM-family wrappers need to know which
# static columns are continuous (get a smooth spline term) vs categorical
# (get a factor/dummy term) — defined once here so every wrapper agrees.
STATIC_SPLINE_FEATURES = ("age",)
STATIC_CATEGORICAL_FEATURES = ("gender",)

