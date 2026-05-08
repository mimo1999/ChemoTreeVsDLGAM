"""
Factory for creating ML models and components.
"""

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from catboost import CatBoostClassifier
import os
from config.constants import MODEL_PARAMS, BEST_PARAMS, RANDOM_SEED
from ml_model_training.gam_wrapper import LogisticGAMClassifier


class MLModelFactory:
    """Factory class to create ML models."""

    MODEL_CLASSES = {
        'logistic_regression': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'gradient_boosting': HistGradientBoostingClassifier,
        'xgboost': xgb.XGBClassifier,
        'catboost': CatBoostClassifier,
        'gam': LogisticGAMClassifier,
    }
    
    @classmethod
    def create_model(cls, model_type: str, grid_search: bool = False, results_path: str = None, cohort: str = None):
        """Create an ML model of the specified type."""
        
        # Normalize model type
        model_type_lower = model_type.lower().replace(' ', '_')
        
        if model_type_lower not in cls.MODEL_CLASSES:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available types: {list(cls.MODEL_CLASSES.keys())}")
        
        # Get model configuration
        model_config = MODEL_PARAMS.get(model_type, {})
        if not model_config:
            raise ValueError(f"Model config for '{model_type}' not found in model_params.yaml")
        
        # Get parameters based on training mode
        if grid_search:
            params = model_config.get("grid_search", {})
        else:
            # Use best parameters if available and cohort is provided
            if cohort and model_type in BEST_PARAMS and cohort in BEST_PARAMS[model_type]:
                params = BEST_PARAMS[model_type][cohort]
                print("BEST PARAMs:", params)
                print(f"Using best parameters for {model_type} on cohort {cohort}")
            else:
                params = model_config.get("default", {})
                if cohort:
                    print(f"Best parameters not found for {model_type} on cohort {cohort}, using default parameters")
        
        # Create model with appropriate parameters
        model_class = cls.MODEL_CLASSES[model_type_lower]

        model_params = {'random_state': RANDOM_SEED}
        model_params.update(params)

        if model_type_lower == 'catboost':
            model_params ['train_dir'] = cls._get_catboost_train_dir(results_path)
            model_params ['verbose'] = False
        
        return model_class(**model_params)

    
    @classmethod
    def _get_catboost_train_dir(cls, results_path: str):
        """Get CatBoost training directory."""
        train_dir = os.path.join(results_path, "catboost_run_info")
        os.makedirs(train_dir, exist_ok=True)
        return train_dir
    
    @classmethod
    def get_available_models(cls):
        """Get list of available model types."""
        return list(cls.MODEL_CLASSES.keys())
    
    @classmethod
    def get_model_config(cls, model_type: str):
        """Get configuration for a specific model type."""
        return MODEL_PARAMS.get(model_type, {})
