"""
Factory for creating ML models and components.

Wrapper imports for optional-dependency models (EBM, GAMINET, IGANN, …) are
deferred to create_model() so that a missing package only raises when that
specific model is requested, not at import time.
"""

import importlib
import os

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from catboost import CatBoostClassifier

from config.constants import MODEL_PARAMS, BEST_PARAMS, RANDOM_SEED
from ml_model_training.model_wrappers.gam_wrapper import LogisticGAMClassifier

# Wrapper-only models whose deps may not be installed — resolved lazily.
_LAZY_WRAPPERS = {
    'ebm':     ('ml_model_training.model_wrappers.ebm_wrapper',     'EBMClassifier'),
    'gaminet': ('ml_model_training.model_wrappers.gaminet_wrapper', 'GAMINetWrapperClassifier'),
    'igann':   ('ml_model_training.model_wrappers.igann_wrapper',   'IGANNWrapperClassifier'),
    'nam':     ('ml_model_training.model_wrappers.nam_wrapper',     'NAMWrapperClassifier'),
    'dgam':    ('ml_model_training.model_wrappers.dl_gam_wrapper', 'DLGAMClassifier'),
    'mgam':    ('ml_model_training.model_wrappers.mgam_wrapper',    'MGAMClassifier'),
    'sstgam':  ('ml_model_training.model_wrappers.sstgam_wrapper',  'SSTGAMClassifier'),
    'he_ebm':  ('ml_model_training.model_wrappers.he_ebm_wrapper',  'HEEBMClassifier'),
    'ig_ebm':  ('ml_model_training.model_wrappers.ig_ebm_wrapper',  'IGEBMClassifier'),
}

# Models whose random_state is handled internally by their wrapper.
_WRAPPER_MODELS = frozenset(['gam', 'ebm', 'gaminet', 'igann', 'nam', 'dgam', 'mgam', 'sstgam', 'he_ebm', 'ig_ebm'])


def _load_wrapper(key: str):
    """Import and return a wrapper class on first use."""
    module_path, class_name = _LAZY_WRAPPERS[key]
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except ImportError as e:
        raise ImportError(
            f"Model '{key}' requires optional dependencies that are not installed: {e}"
        ) from e


class MLModelFactory:
    """Factory class to create ML models."""

    _EAGER_CLASSES = {
        'logistic_regression': LogisticRegression,
        'random_forest':       RandomForestClassifier,
        'gradient_boosting':   HistGradientBoostingClassifier,
        'xgboost':             xgb.XGBClassifier,
        'catboost':            CatBoostClassifier,
        'gam':                 LogisticGAMClassifier,
    }

    @classmethod
    def _get_model_class(cls, model_type_lower: str):
        if model_type_lower in cls._EAGER_CLASSES:
            return cls._EAGER_CLASSES[model_type_lower]
        if model_type_lower in _LAZY_WRAPPERS:
            return _load_wrapper(model_type_lower)
        raise ValueError(
            f"Unknown model type: '{model_type_lower}'. "
            f"Available: {list(cls._EAGER_CLASSES) + list(_LAZY_WRAPPERS)}"
        )

    @classmethod
    def create_model(cls, model_type: str, grid_search: bool = False, results_path: str = None, cohort: str = None):
        """Create an ML model of the specified type."""

        model_type_lower = model_type.lower().replace(' ', '_').replace('-', '_')
        model_class = cls._get_model_class(model_type_lower)

        model_config = MODEL_PARAMS.get(model_type, {})
        if not model_config:
            raise ValueError(f"Model config for '{model_type}' not found in model_params.yaml")

        if grid_search:
            params = model_config.get("grid_search", {})
        else:
            if cohort and model_type in BEST_PARAMS and cohort in BEST_PARAMS[model_type]:
                params = BEST_PARAMS[model_type][cohort]
                print("BEST PARAMs:", params)
                print(f"Using best parameters for {model_type} on cohort {cohort}")
            else:
                params = model_config.get("default", {})
                if cohort:
                    print(f"Best parameters not found for {model_type} on cohort {cohort}, using default parameters")

        # Wrapper models manage random_state internally — don't inject it.
        model_params = {} if model_type_lower in _WRAPPER_MODELS else {'random_state': RANDOM_SEED}
        model_params.update(params)

        if model_type_lower == 'catboost':
            import tempfile
            train_dir = os.path.join(tempfile.gettempdir(), "catboost_run_info")
            os.makedirs(os.path.join(train_dir, "tmp"), exist_ok=True)
            model_params['train_dir'] = train_dir
            model_params['verbose'] = False

        if model_type_lower in ('he_ebm', 'ig_ebm'):
            # HE-EBM/IG-EBM have no hyperparameters of their own for the EBM
            # backbone — they pull BEST_PARAMS['EBM'][cohort] internally, so
            # they need the cohort name rather than tuned params for that part.
            model_params['cohort'] = cohort

        return model_class(**model_params)

    @classmethod
    def get_available_models(cls):
        return list(cls._EAGER_CLASSES) + list(_LAZY_WRAPPERS)

    @classmethod
    def get_model_config(cls, model_type: str):
        return MODEL_PARAMS.get(model_type, {})
