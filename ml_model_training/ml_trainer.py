"""
ML Model Trainer for handling training logic.
"""

import pandas as pd
import numpy as np
import pickle
import os
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from config.constants import RANDOM_SEED
from ml_model_training.ml_factory import MLModelFactory
from ml_model_training.ml_evaluator import Loss
import json
from pathlib import Path
        

class MLTrainer:
    """Handles ML model training logic."""
    
    def __init__(self, model_type: str, feature_combination_method: str, 
                 grid_search: bool, results_path: str, cohort: str = None):
        self.model_type = model_type
        self.feature_combination_method = feature_combination_method
        self.grid_search = grid_search
        self.results_path = results_path
        self.cohort = cohort
        self.loss = Loss('cpu', True, True, True, True, True, True, True, True, True, True, True)
    
    def train_model(self, X_train: pd.DataFrame, Y_train: pd.Series, 
                   X_test: pd.DataFrame, Y_test: pd.Series, 
                   fold: int, subj_ids_train: pd.Series):
        """Train model and return evaluation metrics and best parameters."""
        
        print('# number of features for training: ', len(X_train.columns))
        print("==================== Model Training  =======================")
        
        # Preprocess data based on model type
        X_train_processed, X_test_processed = self._preprocess_data(
            X_train, X_test, self.model_type, self.feature_combination_method
        )
        
        # Create model
        model = self._create_model()
        
        # Train model
        if self.grid_search:
            model, best_params = self._train_with_grid_search(
                model, X_train_processed, Y_train, subj_ids_train, fold
            )
        else:
            model, best_params = self._train_direct(
                model, X_train_processed, Y_train
            )
        
        # Evaluate model
        eval_metrics = self._evaluate_model(model, X_test_processed, Y_test, X_train.columns, fold)
        
        return eval_metrics, best_params
    
    def _preprocess_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                        model_type: str, feature_method: str):
        """Preprocess data based on model type."""
        
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()
        
        # Logistic Regression requires scaling
        if model_type in ("Logistic Regression"):
            scaler = StandardScaler()
            X_train_processed = scaler.fit_transform(X_train_processed)
            X_test_processed = scaler.transform(X_test_processed)
        
        # XGBoost with aggregation needs numeric conversion
        elif model_type == "Xgboost" and feature_method == 'aggregate':
            X_train_processed = X_train_processed.apply(pd.to_numeric, errors='coerce')
            X_test_processed = X_test_processed.apply(pd.to_numeric, errors='coerce')
        
        return X_train_processed, X_test_processed
    
    def _create_model(self):
        """Create model using factory."""
        return MLModelFactory.create_model(
            self.model_type, 
            self.grid_search, 
            self.results_path,
            self.cohort
        )
    
    def _train_with_grid_search(self, model, X_train, Y_train, subj_ids_train, fold):
        """Train model with grid search."""
        print(f"[Grid Search Enabled] for model: {self.model_type}")
        
        # Get parameter grid
        model_config = MLModelFactory.get_model_config(self.model_type)
        param_grid = model_config.get("grid_search", {})
        
        # Setup cross-validation with explicit random state
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,  # Use single job for deterministic results
            verbose=1
        )
        
        grid_search.fit(X_train, Y_train, groups=subj_ids_train)
        
        print("Best Parameters:", grid_search.best_params_)
        print("Best Score (CV ROC AUC):", grid_search.best_score_)
        
        # Save inner fold results for this outer fold
        self._save_inner_fold_results(grid_search, fold)
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def _train_direct(self, model, X_train, Y_train, feature_names=None):
        """ Direct training with grid_search = False."""
        fit_kwargs = {}
        if self.model_type == "GAM" and feature_names is not None:
            fit_kwargs["feature_names"] = list(feature_names)
        model.fit(X_train, Y_train, **fit_kwargs)
        model_params = model.get_params()
        
        return model, model_params
    
    def _save_inner_fold_results(self, grid_search, outer_fold):
        """Save inner fold parameter combinations and ROC scores for this outer fold."""

        
        
        results_dir = Path(self.results_path)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        cv_results = grid_search.cv_results_
        
        
        inner_fold_results = {
            'outer_fold': outer_fold,
            'model_type': self.model_type,
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'all_combinations': []
        }
        
        
        for i, params in enumerate(cv_results['params']):
            combination = {
                'combination_id': i,
                'parameters': params,
                'mean_roc_auc': float(cv_results['mean_test_score'][i]),
                'std_roc_auc': float(cv_results['std_test_score'][i]),
                'inner_fold_scores': []
            }
            
            
            for inner_fold_idx in range(3):  # 3 inner folds
                fold_key = f'split{inner_fold_idx}_test_score'
                if fold_key in cv_results:
                    combination['inner_fold_scores'].append({
                        'inner_fold': inner_fold_idx,
                        'roc_auc': float(cv_results[fold_key][i])
                    })
            
            inner_fold_results['all_combinations'].append(combination)
        
        # Save results for this outer fold
        fold_file = results_dir / f'{self.model_type}_inner_folds_fold_{outer_fold}.json'
        with open(fold_file, 'w') as f:
            json.dump(inner_fold_results, f, indent=2)
        
        #print(f"Inner fold results saved to: {fold_file}")
        #print(f"Best params: {grid_search.best_params_}")
        #print(f"Best ROC AUC: {grid_search.best_score_:.4f}")
        #print(f"Total combinations tested: {len(cv_results['params'])}")
    
    def _evaluate_model(self, model, X_test, Y_test, features, fold):
        """Evaluate model and save outputs/importance if applicable."""
        # === Predictions and Evaluation ===
        prob = model.predict_proba(X_test)
        
        # Different logit handling
        if hasattr(model, 'predict_log_proba'):
            logits = model.predict_log_proba(X_test)[:, 1]
        else:
            logits = np.log2(prob[:, 1] / (1 - prob[:, 1] + 1e-8))  # logit from probs
        
        eval_metrics = self.loss(prob[:, 1], np.asarray(Y_test), logits, False, True, False)


        self.save_model(model, fold)
        # Save feature importance 
        if self.model_type in ["Random Forest","CatBoost"]:
            self.save_outputImp(Y_test, prob[:, 1], logits, model.feature_importances_, features, fold)

        return eval_metrics
    
    def save_model(self, model, fold):
        """Save trained model."""
        model_path = os.path.join(
            self.results_path, f"{self.model_type}_model_fold{fold}.joblib"
        )
        try:
            joblib.dump(model, model_path)
        except (TypeError, PicklingError, AttributeError) as exc:
            print(
                f"[save_model] WARNING: could not joblib-dump "
                f"{self.model_type} fold {fold}: {type(exc).__name__}: {exc}. "
                "Continuing without saving the fitted model."
            )

    def save_outputImp(self, labels, prob, logits, importance, features, fold):
        """Save model outputs and feature importance."""
        output_df = pd.DataFrame()
        output_df['Labels'] = labels.values
        output_df['Prob'] = prob
        output_df['Logits'] = np.asarray(logits)


        with open(os.path.join(self.results_path, f'{self.model_type}_output_dict_fold{fold}.pkl'), 'wb') as fp:
               pickle.dump(output_df, fp)
        
        imp_df = pd.DataFrame()
        imp_df['imp'] = importance
        imp_df['feature'] = features
        imp_df.sort_values(by='imp', ascending=False).to_csv(os.path.join(self.results_path, f'{self.model_type}_feature_importance_fold{fold}.csv'), index=False)
