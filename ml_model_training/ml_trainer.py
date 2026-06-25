"""
ML Model Trainer for handling training logic.
"""

import pandas as pd
import numpy as np
import pickle
from pickle import PicklingError
import os
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from config.constants import RANDOM_SEED
from ml_model_training.ml_factory import MLModelFactory
from ml_model_training.ml_evaluator import Loss, _TORCH_AVAILABLE as _LOSS_AVAILABLE
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
        self.loss = Loss('cpu', True, True, True, True, True, True, True, True, True, True, True) if _LOSS_AVAILABLE else None

    def train_model(self, X_train: pd.DataFrame, Y_train: pd.Series,
                   X_test: pd.DataFrame, Y_test: pd.Series,
                   fold: int, subj_ids_train: pd.Series,
                   sample_weight=None, val_set=None):
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
                model, X_train_processed, Y_train, subj_ids_train, fold,
                sample_weight=sample_weight
            )
        else:
            model, best_params = self._train_direct(
                model, X_train_processed, Y_train, sample_weight=sample_weight,
                val_set=val_set
            )

        # Evaluate model
        eval_metrics = self._evaluate_model(model, X_test_processed, Y_test, X_train.columns, fold)

        return eval_metrics, best_params

    def _preprocess_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                        model_type: str, feature_method: str):
        """Preprocess data based on model type."""

        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()

        # Logistic Regression requires scaling. Inputs are expected to be
        # NaN-free on disk (feat_type=A fills empty bins with -1 as a
        # sentinel; feat_type=VMD ffill/bfill/0-imputes during extraction).
        if model_type == "Logistic Regression":
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

    def _train_with_grid_search(self, model, X_train, Y_train, subj_ids_train, fold,
                                sample_weight=None):
        """Train model with grid search."""
        print(f"[Grid Search Enabled] for model: {self.model_type}")

        # Get parameter grid
        model_config = MLModelFactory.get_model_config(self.model_type)
        param_grid = model_config.get("grid_search", {})

        # Setup cross-validation with explicit random state
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

        # MGAM: GridSearchCV refit uses the full oversampled outer-fold set, which is
        # larger than any inner-fold slice and can hit solver instability (CARMA array
        # ownership / degenerate CD states). Skip refit and do it manually via
        # _train_direct with the best params after grid search completes.
        _NO_REFIT_MODELS = {"MGAM"}
        do_refit = self.model_type not in _NO_REFIT_MODELS

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
            verbose=1,
            refit=do_refit,
        )

        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
        grid_search.fit(X_train, Y_train, groups=subj_ids_train, **fit_params)

        print("Best Parameters:", grid_search.best_params_)
        print("Best Score (CV ROC AUC):", grid_search.best_score_)

        # Save inner fold results for this outer fold
        self._save_inner_fold_results(grid_search, fold)

        if not do_refit:
            # Manually refit with best params on the full outer training set.
            best_model = MLModelFactory.create_model(
                self.model_type, grid_search=False,
                results_path=self.results_path, cohort=self.cohort,
            )
            best_model.set_params(**grid_search.best_params_)
            best_model_fitted, _ = self._train_direct(best_model, X_train, Y_train, sample_weight)
            return best_model_fitted, grid_search.best_params_

        return grid_search.best_estimator_, grid_search.best_params_

    def _train_direct(self, model, X_train, Y_train, sample_weight=None, val_set=None):
        """Direct training with grid_search = False."""

        # Wrappers that accept feature_names (GAM, EBM, GAMINET, IGANN, NAM, DGAM, FGAM, MVGAM)
        # receive column names so their internals can label features.
        _WRAPPER_TYPES = ("GAM", "EBM", "GAMINET", "IGANN", "NAM", "DGAM", "FGAM", "MVGAM")
        if self.model_type in _WRAPPER_TYPES and hasattr(X_train, "columns"):
            fit_kwargs = {'feature_names': list(X_train.columns)}
            if sample_weight is not None:
                fit_kwargs['sample_weight'] = sample_weight
            if self.model_type == 'IGANN' and val_set is not None:
                fit_kwargs['val_set'] = val_set
            model.fit(X_train, Y_train, **fit_kwargs)
        else:
            if sample_weight is not None:
                model.fit(X_train, Y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train, Y_train)
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

    def _evaluate_model(self, model, X_test, Y_test, features, fold):
        """Evaluate model and save outputs/importance if applicable."""
        prob = model.predict_proba(X_test)

        if hasattr(model, 'predict_log_proba'):
            logits = model.predict_log_proba(X_test)[:, 1]
        else:
            logits = np.log2(prob[:, 1] / (1 - prob[:, 1] + 1e-8))

        if self.loss is not None:
            eval_metrics = self.loss(prob[:, 1], np.asarray(Y_test), logits, False, True, False)
        else:
            # Torch unavailable — compute equivalent metrics with sklearn only
            from sklearn import metrics as skm
            y_true = np.asarray(Y_test)
            y_prob = prob[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            fpr, tpr, _ = skm.roc_curve(y_true, y_prob)
            auc  = skm.auc(fpr, tpr)
            prec_c, rec_c, _ = skm.precision_recall_curve(y_true, y_prob)
            apr  = skm.auc(rec_c, prec_c)
            base = y_true.mean()
            accur   = skm.accuracy_score(y_true, y_pred)
            f_score = skm.f1_score(y_true, y_pred)
            prec    = skm.precision_score(y_true, y_pred, zero_division=0)
            recall  = skm.recall_score(y_true, y_pred, zero_division=0)
            tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred).ravel()
            spec    = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
            npv_val = tn / (tn + fn) if (tn + fn) > 0 else float('nan')
            eval_metrics = ['NA', auc, apr, base, accur, f_score, prec, recall, spec, npv_val, 'NA', 'NA']

        self.save_model(model, fold)
        if self.model_type in ["Random Forest", "CatBoost"]:
            try:
                self.save_outputImp(Y_test, prob[:, 1], logits, model.feature_importances_, features, fold)
            except OSError as e:
                print(f"[save_outputImp] WARNING: could not save feature importance for fold {fold}: {e}. "
                      "Continuing without saving (likely Windows MAX_PATH exceeded).")

        return eval_metrics

    def save_model(self, model, fold):
        """Save trained model.

        Some wrappers (IGANN) attach un-picklable lambdas to the fitted
        instance, which crash joblib.dump. We catch those errors so the
        rest of the pipeline still runs.
        """
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
        os.makedirs(self.results_path, exist_ok=True)
        output_df = pd.DataFrame()
        output_df['Labels'] = labels.values
        output_df['Prob'] = prob
        output_df['Logits'] = np.asarray(logits)

        with open(os.path.join(self.results_path, f'{self.model_type}_output_dict_fold{fold}.pkl'), 'wb') as fp:
               pickle.dump(output_df, fp)

        imp_df = pd.DataFrame()
        imp_df['imp'] = importance
        imp_df['feature'] = features
        imp_df.sort_values(by='imp', ascending=False).to_csv(
            os.path.join(self.results_path, f'{self.model_type}_feature_importance_fold{fold}.csv'),
            index=False,
        )
