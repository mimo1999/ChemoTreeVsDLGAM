"""
Data loading and processing module for ML/NN training.
"""

import pickle
from typing import Tuple, Any

import numpy as np
import pandas as pd

from feature_processing.feature_selector import (
    TreeFeatureSelector, TopKFeatureSelector, CorrelationFeatureSelector,
    XGBGainSelector, LassoFeatureSelector,
)
from utils.preprocessing_utils import fit_transform_gender, oversample_minority_with_groups
from feature_processing.ml_feature_matrix_builder import FeatureExtractorFactory
from utils.io_utils import load_pickle, save_pickle, fold_file


class DataLoader:
    """Handles data loading and processing for machine learning models."""

    def __init__(self,
                 target_cohort: str,
                 feature_combination_method: str,
                 training_data_type: str,
                 feature_selection_boolen: bool,
                 oversampling_method: str,
                 saved_data_path: str,
                 paths: Any,
                 feat_type: str = 'standard',
                 agg_interval: int = 24,
                 top_k_features: int = 250,
                 selector_type: str = 'tree',
                 lookback_days: int = None):
        """
        Initialize the data loader.

        Args:
            target_cohort: Name of the target cohort
            feature_combination_method: Method for combining features
            training_data_type: Type of training data
            feature_selection_boolen: Whether to apply MI-based feature selection
            oversampling_method: Method for oversampling
            saved_data_path: Path to saved data
            paths: Path configuration object
            feat_type: Feature type ('standard' for traditional, 'V', 'M', 'D', 'VD',
                       'VM', 'MD', 'VMD' for timeseries)
            agg_interval: Aggregation interval in hours (default: 24)
            top_k_features: Number of top features to retain via mutual information
                            when feature_selection_boolen=True (default: 250).
            selector_type: Feature selector to use when feature_selection_boolen=True.
                           'tree'        = RandomForest MDI importance (TreeFeatureSelector),
                           'mi'          = Mutual Information SelectKBest (TopKFeatureSelector),
                           'correlation' = |feature-target correlation| top-K,
                           'xgb_gain'    = XGBoost gain importance,
                           'lasso'       = L1-regularized logistic regression.
            lookback_days: Restrict feature window to the last N calendar days before
                           discharge. None (default) = use the full observation window.
        """
        self.target_cohort = target_cohort
        self.feature_combination_method = feature_combination_method
        self.training_data_type = training_data_type
        self.feature_selection_boolen = feature_selection_boolen
        self.oversampling_method = oversampling_method
        self.saved_data_path = saved_data_path
        self.paths = paths
        self.feat_type = feat_type
        self.agg_interval = agg_interval
        self.top_k_features = top_k_features
        self.selector_type = selector_type
        self.lookback_days = lookback_days

        if feat_type == 'standard':
            self.feature_extractor_type = 'traditional'
        else:
            self.feature_extractor_type = 'timeseries'

        self.feature_extractor = FeatureExtractorFactory.create_extractor(self.feature_extractor_type)


    def load_and_process_data(self, fold: int, load_training_data_boolen: bool = False) -> Tuple | None:
        """
        Load and process training data for a specific fold.

        Args:
            fold: Fold number for cross-validation
            load_training_data_boolen: Whether to load pre-saved data

        Returns:
            Tuple of (X_train, Y_train, X_test, Y_test, X_val, Y_val, subj_ids_train, subj_ids_val)
        """
        if load_training_data_boolen:
            return self._load_pre_saved_data(fold)
        else:
            return self._extract_and_process_data(fold)

    def _load_pre_saved_data(self, fold: int) -> Tuple | None:
        """Load previously saved training data."""
        print("Loading previously saved training data...")
        base = self.paths.training
        x_train_path = fold_file(base, 'X_train', fold)

        if not x_train_path.exists():
            print('Training data not found!')
            return None
        X_train = load_pickle(fold_file(base, 'X_train', fold))
        Y_train = load_pickle(fold_file(base, 'Y_train', fold))
        X_test = load_pickle(fold_file(base, 'X_test', fold))
        Y_test = load_pickle(fold_file(base, 'Y_test', fold))
        X_val = load_pickle(fold_file(base, 'X_val', fold))
        Y_val = load_pickle(fold_file(base, 'Y_val', fold))
        subj_ids_train = load_pickle(fold_file(base, 'Sub_train', fold))
        subj_ids_val = load_pickle(fold_file(base, 'Sub_val', fold))

        return (X_train, Y_train, X_test, Y_test, X_val, Y_val, subj_ids_train, subj_ids_val)

    def _extract_and_process_data(self, fold: int) -> Tuple:
        """Extract and process fresh data from folds."""
        with open(self.paths.folds / f'fold_{fold}.pkl', 'rb') as f:
            train_ids, val_ids, test_ids = pickle.load(f)

        print(f'Feature extractor type: {self.feature_extractor_type}')
        print(f'Feature type: {self.feat_type}')
        if self.lookback_days is not None:
            print(f'Lookback window: last {self.lookback_days} days before discharge')
        if self.feature_selection_boolen:
            sel_label = 'none (passthrough)' if self.selector_type == 'none' else f'{self.selector_type.upper()} top-{self.top_k_features}'
            print(f'Feature selection enabled - {sel_label} will be applied after extraction.')

        X_train, Y_train, subj_ids_train, hadm_ids_train, train_itemids, train_bins = self._extract_data_split(train_ids, "training", fold)
        X_test, Y_test, subj_ids_test, hadm_ids_test = self._extract_data_split(test_ids, "test", fold, itemids=train_itemids, bins=train_bins)
        X_val, Y_val, subj_ids_val, hadm_ids_val = self._extract_data_split(val_ids, "validation", fold, itemids=train_itemids, bins=train_bins)

        if self.feature_selection_boolen:
            X_train, X_test, X_val = self._apply_mi_selection(X_train, Y_train, X_test, X_val)

        self._validate_data_leakage(train_ids, test_ids)
        self._validate_data_leakage(val_ids, test_ids)

        if "DEMO" in self.training_data_type:
            X_train, X_test, X_val = fit_transform_gender(X_train, X_test, X_val)

        # Fill any remaining NaN with 0 (admissions with no observations in the
        # lookback window for a selected itemid). Models like RF do not tolerate
        # NaN natively; 0 is the correct sentinel for "not measured".
        X_train = X_train.fillna(0) if isinstance(X_train, pd.DataFrame) else X_train
        X_test  = X_test.fillna(0)  if isinstance(X_test,  pd.DataFrame) else X_test
        X_val   = X_val.fillna(0)   if isinstance(X_val,   pd.DataFrame) else X_val

        self._save_processed_data(X_train, Y_train, X_test, Y_test, X_val, Y_val, subj_ids_train, subj_ids_val, fold)

        return X_train, Y_train, X_test, Y_test, X_val, Y_val, subj_ids_train, subj_ids_val

    def _extract_data_split(self, ids, split_name: str, fold: int, itemids=None, bins=None):
        """Extract data for a specific split (train/test/val)."""
        result = self.feature_extractor.extract_features(
            target_cohort=self.target_cohort,
            ids=ids,
            feature_combination_method=self.feature_combination_method,
            training_data_types=self.training_data_type,
            fold=fold,
            feature_threshold=self.feature_selection_boolen,
            saved_data_path=self.saved_data_path,
            feat_type=self.feat_type,
            agg_interval=self.agg_interval,
            itemids=itemids,
            bins=bins,
            lookback_days=self.lookback_days,
        )
        return result

    def _apply_mi_selection(self, X_train, Y_train, X_test, X_val):
        """Fit a feature selector on X_train and apply the chosen columns to all splits.

        Non-numeric columns (e.g. un-encoded gender) are kept as-is and
        re-appended after selection so the selector only scores lab features.
        """
        numeric_cols     = X_train.select_dtypes(include='number').columns.tolist()
        non_numeric_cols = X_train.select_dtypes(exclude='number').columns.tolist()

        if self.selector_type == 'none':
            print(
                f"[Feature Selection] selector_type='none' — "
                f"passing all {len(numeric_cols)} numeric + "
                f"{len(non_numeric_cols)} non-numeric columns through unchanged."
            )
            return X_train, X_test, X_val

        if self.selector_type == 'mi':
            selector = TopKFeatureSelector(max_features=self.top_k_features, random_state=42)
            label = 'MI (mutual information)'
        elif self.selector_type in ('correlation', 'corr'):
            selector = CorrelationFeatureSelector(max_features=self.top_k_features, method='pearson', random_state=42)
            label = 'Correlation (|feature-target corr|)'
        elif self.selector_type == 'xgb_gain':
            selector = XGBGainSelector(max_features=self.top_k_features, random_state=42)
            label = 'XGB Gain (XGBoost gain importance)'
        elif self.selector_type == 'lasso':
            selector = LassoFeatureSelector(max_features=self.top_k_features, random_state=42)
            label = 'Lasso (L1-regularized logistic regression)'
        else:  # 'tree'
            selector = TreeFeatureSelector(max_features=self.top_k_features, n_estimators=100, max_depth=5, random_state=42)
            label = 'Tree (RF importance)'

        selector.fit_transform(
            X_train[numeric_cols].values,
            np.ravel(Y_train),
            feature_names=numeric_cols,
        )
        selected_numeric = selector.get_selected_features()
        selected_cols    = selected_numeric + non_numeric_cols

        print(
            f"[{label}] {len(numeric_cols)} numeric features -> "
            f"{len(selected_numeric)} selected (top-{self.top_k_features}); "
            f"{len(non_numeric_cols)} non-numeric columns kept as-is"
        )

        return (
            X_train[selected_cols],
            X_test[selected_cols],
            X_val[selected_cols],
        )

    def _validate_data_leakage(self, train_ids, test_ids):
        """Validate that there's no data leakage between train and test sets."""
        train_patients = set(train_ids[:, 0])
        test_patients = set(test_ids[:, 0])
        train_admissions = set(train_ids[:, 1])
        test_admissions = set(test_ids[:, 1])

        if train_patients.intersection(test_patients):
            print('WARNING: There are common patients in test and train sets!')
        if train_admissions.intersection(test_admissions):
            print('WARNING: There are common admissions in test and train sets!')

    def _save_processed_data(self, X_train, Y_train, X_test, Y_test, X_val, Y_val, subj_ids_train, subj_ids_val, fold):
        """Save processed data for future use."""
        base = self.paths.training
        save_pickle(X_train, fold_file(base, 'X_train', fold))
        save_pickle(Y_train, fold_file(base, 'Y_train', fold))
        save_pickle(X_test, fold_file(base, 'X_test', fold))
        save_pickle(Y_test, fold_file(base, 'Y_test', fold))
        save_pickle(X_val, fold_file(base, 'X_val', fold))
        save_pickle(Y_val, fold_file(base, 'Y_val', fold))
        save_pickle(subj_ids_train, fold_file(base, 'Sub_train', fold))
        save_pickle(subj_ids_val, fold_file(base, 'Sub_val', fold))


# Convenience function for backward compatibility
def load_and_process_data(target_cohort: str,
                         feature_combination_method: str,
                         training_data_type: str,
                         feature_selection_boolen: bool,
                         oversampling_method: str,
                         saved_data_path: str,
                         paths: Any,
                         fold: int,
                         load_training_data_boolen: bool = False,
                         feat_type: str = 'standard',
                         top_k_features: int = 250) -> Tuple | None:
    """
    Convenience function for loading and processing data.

    Creates a DataLoader instance and calls load_and_process_data.
    Useful for backward compatibility or when you don't need to reuse the loader.
    """
    loader = DataLoader(
        target_cohort=target_cohort,
        feature_combination_method=feature_combination_method,
        training_data_type=training_data_type,
        feature_selection_boolen=feature_selection_boolen,
        oversampling_method=oversampling_method,
        saved_data_path=saved_data_path,
        paths=paths,
        feat_type=feat_type,
        top_k_features=top_k_features,
    )

    return loader.load_and_process_data(fold, load_training_data_boolen)
