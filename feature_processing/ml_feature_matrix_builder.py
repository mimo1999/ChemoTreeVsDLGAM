"""
Feature extraction utilities for different data representations.

"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Tuple, List, Optional, Dict, Any
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Approach A — Temporal Bin Aggregation with Missingness Preservation
# ---------------------------------------------------------------------------
#
# Three day-bins partition the 15-day observation window:
#   B1: days 1-5   (bin indices 0..4)
#   B2: days 6-10  (bin indices 5..9)
#   B3: days 11-15 (bin indices 10..14)
#
# For each (admission, itemid, B-bin) we emit five features:
#   mean, delta, observed_fraction, last_value, observation_center
#
# Missing days remain NaN throughout aggregation — there is no forward-fill,
# backward-fill, or zero imputation, so missingness is clinically informative.

A_BINS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("B1", (0, 1, 2, 3, 4)),
    ("B2", (5, 6, 7, 8, 9)),
    ("B3", (10, 11, 12, 13, 14)),
)

A_FEATURES: Tuple[str, ...] = (
    "mean",
    "delta",
    "observed_fraction",
    "last_value",
    "observation_center",
)


def _compute_a_bin_features(values: np.ndarray, bin_size: int) -> Dict[str, np.ndarray]:
    """Compute the five Approach-A features for one (lab, B-bin) block.

    Parameters
    ----------
    values : np.ndarray of shape (n_rows, bin_size)
        Lab values within this B-bin. NaN means the lab was not measured on that day.
    bin_size : int
        Number of calendar days the bin spans (denominator of ``observed_fraction``
        and normalizer for ``observation_center``).

    Returns
    -------
    dict with keys ``mean, delta, observed_fraction, last_value, observation_center``.
    Each is a float64 array of length ``n_rows``. Empty rows produce NaN for all
    features except ``observed_fraction`` (always numeric, 0.0 when empty).
    Single-observation rows have ``delta == 0.0`` per spec.
    """
    if values.shape[1] != bin_size:
        raise ValueError(
            f"values has {values.shape[1]} columns but bin_size={bin_size}"
        )

    n_rows = values.shape[0]
    mask = ~np.isnan(values)
    n_obs = mask.sum(axis=1)
    has_any = n_obs > 0

    observed_fraction = n_obs.astype(np.float64) / float(bin_size)

    mean = np.full(n_rows, np.nan, dtype=np.float64)
    if has_any.any():
        with np.errstate(invalid="ignore"):
            sums = np.where(mask, values, 0.0).sum(axis=1)
            mean[has_any] = sums[has_any] / n_obs[has_any]

    first_idx = np.full(n_rows, -1, dtype=np.int64)
    last_idx = np.full(n_rows, -1, dtype=np.int64)
    if has_any.any():
        first_idx[has_any] = np.argmax(mask[has_any], axis=1)
        last_idx[has_any] = bin_size - 1 - np.argmax(mask[has_any][:, ::-1], axis=1)

    last_value = np.full(n_rows, np.nan, dtype=np.float64)
    delta = np.full(n_rows, np.nan, dtype=np.float64)
    if has_any.any():
        rows = np.where(has_any)[0]
        last_value[rows] = values[rows, last_idx[rows]]
        first_vals = values[rows, first_idx[rows]]
        single = n_obs[rows] == 1
        delta[rows] = np.where(single, 0.0, last_value[rows] - first_vals)

    observation_center = np.full(n_rows, np.nan, dtype=np.float64)
    if has_any.any():
        if bin_size > 1:
            norm_positions = np.arange(bin_size, dtype=np.float64) / float(bin_size - 1)
        else:
            norm_positions = np.array([0.5], dtype=np.float64)
        weighted = np.where(mask, norm_positions[None, :], 0.0).sum(axis=1)
        observation_center[has_any] = weighted[has_any] / n_obs[has_any]

    return {
        "mean": mean,
        "delta": delta,
        "observed_fraction": observed_fraction,
        "last_value": last_value,
        "observation_center": observation_center,
    }


class BaseFeatureExtractor(ABC):
    """Base class for feature extractors."""
    
    @abstractmethod
    def extract_features(self, target_cohort: str, ids: np.ndarray, 
                        feature_combination_method: str, training_data_types: List[str], 
                        fold: int, feature_threshold: bool, saved_data_path: str,
                        feat_type: str = None, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Extract features for given admission IDs.
        
        Returns:
            X_df: Feature matrix
            y_df: Labels
            subj_ids: Subject IDs
            hadm_ids: Admission IDs
        """
        pass


class TraditionalMLExtractor(BaseFeatureExtractor):
    """Traditional ML feature extraction (original getXY approach)."""
    
    def extract_features(self, target_cohort: str, ids: np.ndarray, 
                        feature_combination_method: str, training_data_types: List[str], 
                        fold: int, feature_threshold: bool, saved_data_path: str,
                        feat_type: str = "standard", itemids: list = None, bins: list = None, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Extract traditional ML features ."""
        
        # Validate feat_type for traditional extractor
        if feat_type not in ["standard"]:
            print(f"Warning: feat_type '{feat_type}' not supported for traditional extractor. Using 'standard'.")
            feat_type = "standard"
        
        X_df = pd.DataFrame()   
        y_df = pd.DataFrame()   
        subj_ids_out, hadm_ids_out = [], []

        concat_column_names = None
        agg_interval = kwargs.get('agg_interval', 24)
        labels = pd.read_csv(f'{saved_data_path}/processed_admission_features_csv/{target_cohort}/agg_interval_{agg_interval}h/labels.csv', header=0)
        
        if feature_threshold:
            cc_top_features = pd.read_csv(f'{saved_data_path}/top_features/feat_imp_summary.csv')
            selected_features = cc_top_features.itemid.head(100)
            selected_features = list(map(str, selected_features))
            #print("Feature selection applied (Top 100 features in cancer chemo cohort), Threshold: ", len(selected_features))

        for subj_id, hadm_id in ids:
            y = labels[labels['hadm_id'] == hadm_id]['label']

            # --------------- DYNAMIC Data --------------------------------------
            dyn_path = f'{saved_data_path}/processed_admission_features_csv/{target_cohort}/agg_interval_{agg_interval}h/{hadm_id}/dynamic.csv'
            if not os.path.exists(dyn_path):
                continue  

            dyn = pd.read_csv(dyn_path, header=[0, 1])
            
            # feature selection
            if feature_threshold > 0:
                dyn = dyn.loc[:, ('LAB', dyn['LAB'].columns.intersection(selected_features))]
                
            dyn_columns = (dyn.columns.get_level_values(0)).unique().tolist()
            dyn_training_data_types = list(set(training_data_types) & set(dyn_columns))

            if not dyn_training_data_types:
                dyn_df = pd.DataFrame()
            else:  
                dyn = dyn[dyn_training_data_types]

                # If concatenation 
                concat_cols = []
                if feature_combination_method == 'concatenate':
                    dyn.columns = dyn.columns.droplevel(0)
                    
                    # prepare concatenated column names
                    if concat_column_names is None:
                        cols = dyn.columns
                        time = dyn.shape[0]
                        
                        for t in range(time):
                            cols_t = [x + "_" + str(t) for x in cols]
                            concat_cols.extend(cols_t)
                    
                    dyn = dyn.to_numpy()
                    dyn = dyn.reshape(1, -1)
                    dyn_df = pd.DataFrame(data=dyn, columns=concat_cols)
                    
                # If aggregation
                else:
                    dyn_df = pd.DataFrame()
                    for key in dyn.columns.levels[0]:   
                        dyn_temp = dyn[key]

                        if ((key == "LAB") or (key == "MEDS")):
                            agg = dyn_temp.aggregate("mean")
                            agg = agg.reset_index()
                        else:
                            agg = dyn_temp.aggregate("max")
                            agg = agg.reset_index()
                            
                        if dyn_df.empty:
                            dyn_df = agg
                        else:
                            dyn_df = pd.concat([dyn_df, agg], axis=0)

                    dyn_df = dyn_df.T
                    dyn_df.columns = dyn_df.iloc[0]
                    dyn_df = dyn_df.iloc[1:, :]

            if "DIAG" in training_data_types:
                stat = pd.read_csv(f'{saved_data_path}/processed_admission_features_csv/{target_cohort}/agg_interval_{agg_interval}h/' + str(hadm_id) + '/static.csv', header=[0, 1])
                stat = stat['COND']               
            else:
                stat = pd.DataFrame()

            if "DEMO" in training_data_types:
                demo = pd.read_csv(f'{saved_data_path}/processed_admission_features_csv/{target_cohort}/agg_interval_{agg_interval}h/' + str(hadm_id) + '/demo.csv', header=0)
            else: 
                demo = pd.DataFrame()
            
            X_row = pd.concat([dyn_df, stat, demo], axis=1)
            if not X_row.empty and not y.empty:
                X_df = pd.concat([X_df, X_row], axis=0) if not X_df.empty else X_row
                y_df = pd.concat([y_df, y], axis=0) if not y_df.empty else y
            
                subj_ids_out.append(subj_id)
                hadm_ids_out.append(hadm_id)
    
        if itemids is None and bins is None:
            # Training: return itemids and bins (empty lists for traditional extractor)
            return (
                X_df.reset_index(drop=True),
                y_df.reset_index(drop=True),
                pd.Series(subj_ids_out, name="subject_id"),
                pd.Series(hadm_ids_out, name="hadm_id"),
                [],  # itemids (not used in traditional extractor)
                []   # bins (not used in traditional extractor)
            )
        else:
            # Test/Val: return only features
            return (
                X_df.reset_index(drop=True),
                y_df.reset_index(drop=True),
                pd.Series(subj_ids_out, name="subject_id"),
                pd.Series(hadm_ids_out, name="hadm_id")
            )


class TimeSeriesExtractor(BaseFeatureExtractor):
    """Time series feature extraction (original getXplus approach)."""
    
    def extract_features(self, target_cohort: str, ids: np.ndarray, 
                        feature_combination_method: str, training_data_types: List[str], 
                        fold: int, feature_threshold: bool, saved_data_path: str,
                        feat_type: str = "VMD", itemids: list = None, bins: list = None, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Extract time series features (VMD - Values, Missingness, Delta)."""
        
        all_raw = []
        all_demo = []
        hadm_ids = []
        subject_ids = []
        agg_interval = kwargs.get('agg_interval', 24)

        for subject_id, hadm_id in ids:
            raw_path = f"{saved_data_path}/processed_admission_features_csv/{target_cohort}/agg_interval_{agg_interval}h/{hadm_id}/raw_labs.csv"
            if not os.path.exists(raw_path):
                continue
            all_raw.append(pd.read_csv(raw_path, header=0))
            all_demo.append(pd.read_csv(f"{saved_data_path}/processed_admission_features_csv/{target_cohort}/agg_interval_{agg_interval}h/{hadm_id}/demo.csv", header=0))
            hadm_ids.append(hadm_id)
            subject_ids.append(subject_id)
            
        if not all_raw:
            return pd.DataFrame(), pd.DataFrame(), pd.Series([], name="subject_id"), pd.Series([], name="hadm_id")
            
        data = pd.concat(all_raw, ignore_index=True).rename(columns={"int": "bin"})
        demo = pd.concat(all_demo, keys=hadm_ids).reset_index(level=1, drop=True)
        
        labels = pd.read_csv(f'{saved_data_path}/processed_admission_features_csv/{target_cohort}/agg_interval_{agg_interval}h/labels.csv', header=0)
        y_df = labels.set_index("hadm_id").loc[hadm_ids, "label"].reset_index(drop=True)

        if feature_threshold and feat_type != "AGG":
            cc_top_features = pd.read_csv(f"{saved_data_path}/top_features/feat_imp_summary.csv")
            selected_features = cc_top_features.itemid.head(100)
            data = data[data["itemid"].isin(selected_features)]

        # Lookback window — restrict to the last N bins before discharge.
        # Applied after Stage 1 (if active) and before any feature computation
        # so all feature types (VMD, A, AGG, …) benefit automatically.
        # On train splits (itemids/bins=None) the filtered bin set is
        # discovered and returned; val/test splits reuse those same bins.
        lookback_days = kwargs.get('lookback_days', None)
        early_data = None
        # lookback window disabled — use all bins
        # if lookback_days is not None and lookback_days > 0:
        #     all_available_bins = sorted(data["bin"].unique())
        #     lookback_bins = set(all_available_bins[-lookback_days:])
        #     early_bins = set(all_available_bins) - lookback_bins
        #     if early_bins:
        #         early_data = data[data["bin"].isin(early_bins)].copy()
        #     data = data[data["bin"].isin(lookback_bins)]
        #     print(f"[lookback] Restricting to last {lookback_days} bins: {sorted(lookback_bins)}")
        #     if early_data is not None and not early_data.empty:
        #         print(f"[lookback] Early-phase aggregate over bins: {sorted(early_bins)}")

        # ------------------------------------------------------------------
        # Full M-GAM on V — quantile-binarized V matrix + missingness
        # indicators (and optionally interactions). Top-10 itemids only.
        # Trains the cutoffs on the *first* call (where itemids/bins are not
        # passed, i.e. the training fold) and reuses them on val/test calls
        # via the ``bins`` slot, which we repurpose to carry the cutoffs dict
        # without needing data_loader signature changes.
        # ------------------------------------------------------------------
        if feat_type == "V_MGAM":
            from feature_processing.mgam_pipeline import (
                fit_quantile_cutoffs,
                mgam_binarize,
            )

            # Build a NaN-preserving V matrix: per-(hadm, item, bin) mean,
            # reindexed across all admissions x items x bins, no ffill / no
            # fillna. Empty cells remain NaN so the binarizer can see them.
            all_admissions = pd.Index(hadm_ids, name="hadm_id")
            if itemids is not None and bins is not None and isinstance(bins, dict):
                all_items = list(itemids)
                all_bins = list(bins.get("__bins__", []))
                cutoffs = bins.get("__cutoffs__", {})
            else:
                all_items = sorted(data["itemid"].unique())
                all_bins = sorted(data["bin"].unique())
                cutoffs = None

            x_values = (
                data.groupby(["hadm_id", "itemid", "bin"])["value"].mean()
                .reindex(pd.MultiIndex.from_product(
                    [all_admissions, all_items, all_bins],
                    names=["hadm_id", "itemid", "bin"],
                ))
                .unstack(level="bin")          # rows: (hadm, item), cols: bin
                .unstack(level="itemid")       # rows: hadm, cols: (bin, item)
            )
            # Flatten MultiIndex columns to `<itemid>_<bin>` strings
            x_values.columns = [f"{itemid}_{b}" for (b, itemid) in x_values.columns]
            x_values = x_values.reindex(index=all_admissions)

            numerical_cols = list(x_values.columns)

            # Train fold: fit cutoffs. Val/test: reuse passed cutoffs.
            if cutoffs is None:
                cutoffs = fit_quantile_cutoffs(
                    x_values, numerical_cols, quantiles=(0.2, 0.4, 0.6, 0.8),
                )

            X_mgam = mgam_binarize(
                x_values,
                numerical_cols=numerical_cols,
                cutoffs=cutoffs,
                specific_mi_intercept=True,
                overall_mi_intercept=False,
                specific_mi_ixn=False,
                overall_mi_ixn=False,
            )

            # Concatenate demographics. NaN-preserving order matches.
            demo_aligned = demo.reindex(index=all_admissions)
            X_df = pd.concat(
                [X_mgam.reset_index(drop=True), demo_aligned.reset_index(drop=True)],
                axis=1,
            )

            if itemids is None and bins is None:
                # Training: return cutoffs piggy-backed in the bins slot.
                bins_payload = {"__bins__": all_bins, "__cutoffs__": cutoffs}
                return (
                    X_df, y_df,
                    pd.Series(subject_ids, name="subject_id"),
                    pd.Series(hadm_ids, name="hadm_id"),
                    all_items, bins_payload,
                )
            return (
                X_df, y_df,
                pd.Series(subject_ids, name="subject_id"),
                pd.Series(hadm_ids, name="hadm_id"),
            )

        # ------------------------------------------------------------------
        # Approach AM = Approach A + M-GAM-style missingness indicators on
        # ``_mean`` columns only. Layers a 300-column binary augmentation on
        # top of the standard 1500-column feat_type=A representation. The
        # M-GAM library is not imported — only the missingness-indicator
        # portion is reimplemented locally in ``feature_processing.mgam_augment``.
        # ------------------------------------------------------------------
        if feat_type == "AM":
            if itemids is None and bins is None:
                am_df, all_itemids, all_bins = self._compute_am_features(
                    data, target_cohort, return_itemids_bins=True,
                )
            else:
                am_df = self._compute_am_features(
                    data, target_cohort, itemids=itemids, bins=bins,
                )

            am_df = am_df.reindex(hadm_ids)
            X_df = pd.concat(
                [am_df.reset_index(drop=True), demo.reset_index(drop=True)],
                axis=1,
            )

            if itemids is None and bins is None:
                return (
                    X_df, y_df,
                    pd.Series(subject_ids, name="subject_id"),
                    pd.Series(hadm_ids, name="hadm_id"),
                    all_itemids, all_bins,
                )
            return (
                X_df, y_df,
                pd.Series(subject_ids, name="subject_id"),
                pd.Series(hadm_ids, name="hadm_id"),
            )

        # ------------------------------------------------------------------
        # Approach A+ — refined per-phase set (last_value, robust_trend,
        # observed_fraction, abnormal_fraction, volatility). Same three
        # B-bin partition as Approach A; ref ranges fit on train fold.
        # ------------------------------------------------------------------
        if feat_type == "A+":
            if itemids is None and bins is None:
                ap_df, all_itemids, all_bins = self._compute_a_plus_features(
                    data, target_cohort, return_itemids_bins=True,
                )
            else:
                ap_df = self._compute_a_plus_features(
                    data, target_cohort, itemids=itemids, bins=bins,
                )

            ap_df = ap_df.reindex(hadm_ids)
            X_df = pd.concat(
                [ap_df.reset_index(drop=True), demo.reset_index(drop=True)],
                axis=1,
            )

            if itemids is None and bins is None:
                return (
                    X_df, y_df,
                    pd.Series(subject_ids, name="subject_id"),
                    pd.Series(hadm_ids, name="hadm_id"),
                    all_itemids, all_bins,
                )
            return (
                X_df, y_df,
                pd.Series(subject_ids, name="subject_id"),
                pd.Series(hadm_ids, name="hadm_id"),
            )

        # ------------------------------------------------------------------
        # Approach A short-circuits the V/M/D pipeline: it operates directly
        # on the long-format ``data`` DataFrame *without any imputation* and
        # produces three-bin-aggregated features per (admission, itemid).
        # ------------------------------------------------------------------
        if feat_type == "A":
            if itemids is None and bins is None:
                a_df, all_itemids, all_bins = self._compute_a_features(
                    data, target_cohort, return_itemids_bins=True,
                )
            else:
                a_df = self._compute_a_features(
                    data, target_cohort, itemids=itemids, bins=bins,
                )

            a_df = a_df.reindex(hadm_ids)
            X_df = pd.concat(
                [a_df.reset_index(drop=True), demo.reset_index(drop=True)],
                axis=1,
            )

            if itemids is None and bins is None:
                return (
                    X_df, y_df,
                    pd.Series(subject_ids, name="subject_id"),
                    pd.Series(hadm_ids, name="hadm_id"),
                    all_itemids, all_bins,
                )
            return (
                X_df, y_df,
                pd.Series(subject_ids, name="subject_id"),
                pd.Series(hadm_ids, name="hadm_id"),
            )

        # Extract itemids and bins from training data or use provided ones
        if itemids is None and bins is None:
            # Training: extract itemids and bins
            x_df, m_df, delta_df, all_itemids, all_bins = self._compute_vmd_features(data, target_cohort, return_itemids_bins=True)
        else:
            # Test/Val: use provided itemids and bins
            x_df, m_df, delta_df = self._compute_vmd_features(data, target_cohort, itemids=itemids, bins=bins)


        # Reindex to ensure correct admission order
        x_df = x_df.reindex(hadm_ids)
        m_df = m_df.reindex(hadm_ids)
        delta_df = delta_df.reindex(hadm_ids)

        if feature_combination_method == "concatenate":
            x_df.columns = [f"X_{itemid}_{bin}" for (bin, itemid) in x_df.columns]
            m_df.columns = [f"M_{itemid}_{bin}" for (bin, itemid) in m_df.columns]
            delta_df.columns = [f"D_{itemid}_{bin}" for (bin, itemid) in delta_df.columns]

        else:
            x_df = x_df.groupby(axis=1, level="itemid").mean()
            x_df.columns = [f"X_{itemid}" for itemid in x_df.columns]
            m_df = m_df.groupby(axis=1, level="itemid").mean()
            m_df.columns = [f"M_{itemid}" for itemid in m_df.columns]
            delta_df = delta_df.groupby(axis=1, level="itemid").mean()
            delta_df.columns = [f"D_{itemid}" for itemid in delta_df.columns]

        # Combine features based on feat_type
        # For timeseries, 'standard' maps to 'V'
        if feat_type == "standard":
            feat_type = "V"
        X_df = self._combine_feature_types(x_df, m_df, delta_df, demo, feat_type)
        if itemids is None and bins is None:
            # Training: return itemids and bins
            return X_df, y_df, pd.Series(subject_ids, name="subject_id"), pd.Series(hadm_ids, name="hadm_id"), all_itemids, all_bins
        else:
            # Test/Val: return only features
            return X_df, y_df, pd.Series(subject_ids, name="subject_id"), pd.Series(hadm_ids, name="hadm_id")
    
    def _compute_vmd_features(self, data: pd.DataFrame, target_cohort, return_itemids_bins=False, itemids=None, bins=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compute Values, Missingness, and Delta features from raw lab data."""
        
        all_admissions = data["hadm_id"].unique()
        
        # Use provided itemids and bins or extract from current data
        if itemids is not None and bins is not None:
            all_items = itemids
            all_bins = bins
        else:
            all_items = data["itemid"].unique()
            all_bins = sorted(data["bin"].unique())
        # In uker dataset some splits have missing itemids add them for consistency
        '''if "uker" in target_cohort:
            expected_items = [i for i in range(1, 49) if i != 39] # based on training data
            missing_items = set(expected_items) - set(all_items)
            if missing_items:
                print(f"Adding missing item IDs: {sorted(missing_items)}")
                all_items = sorted(set(all_items) | set(expected_items))'''
    
        # X: average values within bins
        x_values = (
            data.groupby(["hadm_id", "itemid", "bin"])["value"]
            .mean()
        )
        
        # Ensure all bins are present before unstack
        x_values = x_values.reindex(
            pd.MultiIndex.from_product([all_admissions, all_items, all_bins], names=["hadm_id", "itemid", "bin"])
        )
        
        x_values = x_values.unstack(level="bin")
        
        # Impute missing values (forward/backward fill) after adding missing bins
        x_values = x_values.ffill(axis=1).bfill(axis=1)
        
        # Fill remaining missing values with 0
        x_values = x_values.fillna(0)
        x_df = x_values.unstack(level="itemid")
        
        # M: missingness
        m_values = (
            data.groupby(["hadm_id", "itemid", "bin"])["value"]
            .count()
        )
        
        # Ensure all bins are present before unstack
        m_values = m_values.reindex(
            pd.MultiIndex.from_product([all_admissions, all_items, all_bins], names=["hadm_id", "itemid", "bin"])
        )
        
        m_values = m_values.unstack(level="bin")
        
        # Fill missing values with 0 (no observations = 0 count)
        m_values = m_values.fillna(0).astype(int)
        m_df = m_values.unstack(level="itemid")
        
        # Delta: time since last observed
        delta = np.zeros_like(m_values.values, dtype=float)
        delta[:, 0] = 1 - m_values.values[:, 0]
        
        for t in range(1, m_values.shape[1]):
            prev = delta[:, t - 1]
            obs = m_values.values[:, t]
            delta[:, t] = np.where(obs == 1, 0, 1 + prev)
        
        delta = delta / m_values.shape[1]
        delta_df = pd.DataFrame(delta, index=m_values.index, columns=m_values.columns)
        delta_df = delta_df.unstack(level="itemid")
        
        if return_itemids_bins:
            extracted_itemids = sorted(list(all_items))
            extracted_bins = sorted(list(all_bins))
            return x_df, m_df, delta_df, extracted_itemids, extracted_bins
        else:
            return x_df, m_df, delta_df
    
    def _compute_a_features(self, data: pd.DataFrame, target_cohort: str,
                            return_itemids_bins: bool = False,
                            itemids: list = None, bins: list = None):
        """Approach A — Temporal Bin Aggregation with Missingness Preservation.

        For every (admission, itemid) pair, summarize *observed* values inside
        three consecutive day bins (B1/B2/B3 — see ``A_BINS`` at module
        level). Each bin emits five features: mean, delta, observed_fraction,
        last_value, observation_center.

        Missing days remain NaN throughout aggregation — there is *no*
        forward-fill, backward-fill, or zero imputation, so missingness is
        clinically informative. Every (lab, B-bin) pair therefore yields a
        column ``"<itemid>_B<bin>_<feature>"``.

        Parameters
        ----------
        data : DataFrame
            Long format with columns ``hadm_id, itemid, bin, value`` (the same
            shape consumed by ``_compute_vmd_features``).
        return_itemids_bins : bool
            If True, also return the discovered itemids / bin indices so that
            val/test folds can be aligned with the train fold.
        itemids, bins : optional
            When supplied, reuse those itemids / bin indices instead of
            re-discovering them (used by val/test splits).

        Returns
        -------
        a_df : DataFrame
            Indexed by ``hadm_id``. Columns are ``"<itemid>_B<bin>_<feature>"``.
        itemids, bins : optional
            Returned when ``return_itemids_bins=True``.
        """
        all_admissions = data["hadm_id"].unique()

        if itemids is not None and bins is not None:
            all_items = list(itemids)
            all_bins = list(bins)
        else:
            all_items = sorted(data["itemid"].unique())
            all_bins = sorted(data["bin"].unique())

        # 1) Per-day mean across (admission, itemid, bin), no imputation.
        #    Reindexing introduces NaN for combinations that were never
        #    measured — exactly the signal Approach A wants to preserve.
        daily = (
            data.groupby(["hadm_id", "itemid", "bin"])["value"]
                .mean()
                .reindex(pd.MultiIndex.from_product(
                    [all_admissions, all_items, all_bins],
                    names=["hadm_id", "itemid", "bin"],
                ))
                .unstack(level="bin")  # rows: (hadm_id, itemid); cols: bin idx
        )

        n_admissions = len(all_admissions)
        out_columns: List[str] = []
        out_arrays: List[np.ndarray] = []

        # 2) For each itemid, build a (n_admissions, bin_size) matrix per
        #    Approach-A bin and compute the five features in one vectorized
        #    call — same routine the standalone unit tests cover.
        for itemid in all_items:
            try:
                sub = daily.xs(itemid, level="itemid")
            except KeyError:
                sub = pd.DataFrame(
                    index=all_admissions, columns=daily.columns, dtype=np.float64,
                )
            sub = sub.reindex(index=all_admissions)

            for bin_name, day_idx in A_BINS:
                mat = np.full(
                    (n_admissions, len(day_idx)), np.nan, dtype=np.float64,
                )
                for j, d in enumerate(day_idx):
                    if d in sub.columns:
                        mat[:, j] = sub[d].to_numpy(
                            dtype=np.float64, na_value=np.nan,
                        )
                feats = _compute_a_bin_features(mat, len(day_idx))
                for fname in A_FEATURES:
                    out_columns.append(f"{itemid}_{bin_name}_{fname}")
                    out_arrays.append(feats[fname])

        if out_arrays:
            arr = np.column_stack(out_arrays)
        else:
            arr = np.zeros((n_admissions, 0), dtype=np.float64)

        a_df = pd.DataFrame(
            arr,
            index=pd.Index(all_admissions, name="hadm_id"),
            columns=out_columns,
        )

        # Post-aggregation NaN → 0. The five-feature spec emits NaN for empty
        # bins and missing observations; downstream models (GAM in particular)
        # cannot tolerate NaN cells, and median-imputing a column that is
        # >50 % NaN distorts the signal. Replacing with 0 preserves the
        # missingness signal in ``observed_fraction`` (always 0 for empty
        # bins) while giving solvers a clean, finite design matrix.
        a_df = a_df.fillna(0.0)

        if return_itemids_bins:
            return a_df, all_items, all_bins
        return a_df

    def _compute_am_features(self, data: pd.DataFrame, target_cohort: str,
                             return_itemids_bins: bool = False,
                             itemids: list = None, bins: list = None):
        """Approach AM = Approach A + M-GAM-style missingness indicators.

        Computes the standard ``feat_type=A`` representation (5 features per
        (itemid, B-bin)), then appends one binary missingness indicator per
        ``<itemid>_B<n>_mean`` column. The flag is 1 where the original bin
        had zero observations (i.e. ``<...>_observed_fraction == 0``) and 0
        otherwise.

        Column-count footprint
        ----------------------
        Applying missingness indicators to every A feature would emit
        ``5 * 100 itemids * 3 bins = 1500`` extra columns. Restricting the
        augmentation to ``_mean`` columns only emits ``1 * 100 * 3 = 300``
        — the 5x reduction the design calls for.

        Total dimensionality (top-100 itemids, 14-day window, 3 bins):
            A features            : 100 itemids x 3 bins x 5 = 1500
            M-GAM missing flags   : 100 itemids x 3 bins x 1 =  300
            demographics          :                            +  2
                                                              ------
                                                             ~ 1802 cols
        """
        from feature_processing.mgam_augment import (
            add_missingness_indicators,
            missingness_mask_from_observed_fraction,
        )

        # 1) Build the standard A representation.
        if return_itemids_bins:
            a_df, all_items, all_bins = self._compute_a_features(
                data,
                target_cohort,
                return_itemids_bins=True,
                fill_missing=False,
            )
        else:
            a_df = self._compute_a_features(
                data,
                target_cohort,
                itemids=itemids,
                bins=bins,
                fill_missing=False,
            )

        # 2) Derive a missingness mask from the A frame's ``_observed_fraction``
        #    columns and append one 0/1 indicator per ``_mean`` column.
        miss_mask = missingness_mask_from_observed_fraction(
            a_df,
            observed_fraction_suffix="_observed_fraction",
            target_suffix="_mean",
        )
        mean_cols = [c for c in a_df.columns if c.endswith("_mean")]
        if miss_mask.shape[1] != len(mean_cols):
            # observed_fraction columns and mean columns must be 1:1
            raise RuntimeError(
                f"Mismatch between A's _mean ({len(mean_cols)}) and "
                f"_observed_fraction ({miss_mask.shape[1]}) column counts"
            )
        am_df = add_missingness_indicators(
            a_df,
            target_cols=mean_cols,
            missing_mask=miss_mask,
            suffix="_missing",
        )
        am_df = am_df.fillna(0.0)

        if return_itemids_bins:
            return am_df, all_items, all_bins
        return am_df

    def _combine_feature_types(self, x_df: pd.DataFrame, m_df: pd.DataFrame, delta_df: pd.DataFrame,
                             demo: pd.DataFrame, feat_type: str) -> pd.DataFrame:
        """Combine different feature types based on feat_type parameter."""
        if feat_type == "V":
            return pd.concat([x_df, demo], axis=1).reset_index(drop=True)
        elif feat_type == "M":
            return pd.concat([m_df, demo], axis=1).reset_index(drop=True)
        elif feat_type == "D":
            return pd.concat([delta_df, demo], axis=1).reset_index(drop=True)
        elif feat_type == "VD":
            return pd.concat([x_df, delta_df, demo], axis=1).reset_index(drop=True)
        elif feat_type == "VM":
            return pd.concat([x_df, m_df, demo], axis=1).reset_index(drop=True)
        elif feat_type == "MD":
            return pd.concat([m_df, delta_df, demo], axis=1).reset_index(drop=True)
        elif feat_type == "VMD":
            return pd.concat([x_df, m_df, delta_df, demo], axis=1).reset_index(drop=True)
        else:
            raise ValueError(
                f"Invalid feat_type: {feat_type}. "
                "Must be one of:'V', 'M', 'D', 'VD', 'VM', 'MD', 'VMD'."
            )




class FeatureExtractorFactory:
    """Factory class to create appropriate feature extractors."""
    
    _extractors = {
        'traditional': TraditionalMLExtractor,
        'timeseries': TimeSeriesExtractor,
    }
    
    @classmethod
    def create_extractor(cls, extractor_type: str) -> BaseFeatureExtractor:
        """Create a feature extractor of the specified type."""
        if extractor_type not in cls._extractors:
            raise ValueError(f"Unknown extractor type: {extractor_type}. "
                           f"Available types: {list(cls._extractors.keys())}")
        
        return cls._extractors[extractor_type]()
    
    @classmethod
    def get_available_extractors(cls) -> List[str]:
        """Get list of available extractor types."""
        return list(cls._extractors.keys())


# Convenience functions for backward compatibility
def getXY(target_cohort, ids: np.ndarray, feature_combination_method: str, 
          training_data_types: List[str], fold: int, feature_threshold: bool, 
          saved_data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Backward compatibility wrapper for traditional ML features."""
    extractor = TraditionalMLExtractor()
    return extractor.extract_features(target_cohort, ids, feature_combination_method, 
                                     training_data_types, fold, feature_threshold, saved_data_path)


def getXplus(ids, saved_data_path, target_cohort, feature_threshold=False, 
            feature_combination_method="concatenate", feat_type="VMD") -> Tuple[pd.DataFrame, pd.DataFrame, List, List]:
    """Backward compatibility wrapper for time series features."""
    extractor = TimeSeriesExtractor()
    X_df, y_df, subj_ids, hadm_ids = extractor.extract_features(
        target_cohort, ids, feature_combination_method, ["LAB", "DEMO"],
        0, feature_threshold, saved_data_path, feat_type=feat_type
    )
    return X_df, y_df, subj_ids.tolist(), hadm_ids.tolist()
