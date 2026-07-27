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
# Approach A — refined per-phase feature set
# ---------------------------------------------------------------------------
#
# A single phase spanning the whole 15-day observation window (day indices
# 0..14). Nine features per (lab, phase), chosen via a from-scratch
# candidate-expansion + correlation-clustering experiment (2026-07-19,
# aplus_feature_expansion_experiment.py): a from-scratch tree-selected
# 17-candidate pool reached 0.8224 AUC-ROC (vs the old 4+3 = 7-feature set's
# 0.7965) on mimic_cohort_NF_30_days/VMD/top-250/no-oversampling, 5-fold —
# but pairwise |correlation| across all 17 (averaged per-lab) showed two
# tightly redundant clusters: a "level" cluster (mean/median/min/max/
# latest_value/first_value, mean pairwise |corr| 0.87-0.99) and a "spread"
# cluster (std/range/iqr/volatility, 0.83-0.97) — together spanning only
# ~6 independent dimensions out of 17 (90% variance via eigen-decomposition).
# Keeping ONE representative per cluster plus the 5 genuinely-distinct
# features (each <0.6 |corr| with everything else) matched the full-17
# pool's AUC-ROC (0.8235 vs 0.8224) at roughly half the column count —
# this decorrelated-9 set is what ships here (reduced set used in A):
#
#   1. median                 — level-cluster representative (best-connected
#                               node: 0.986 |corr| w/ mean, 0.93 w/ min,
#                               0.93 w/ latest_value, 0.91 w/ max).
#   2. std                    — spread-cluster representative (ddof=1;
#                               best-connected: 0.97 w/ range, 0.93 w/ iqr,
#                               0.90 w/ volatility).
#   3. latest_zscore           — signed z-score of the most recent value vs
#                               a per-itemid train-fold robust (median, IQR)
#                               reference (recency-aware deviation).
#   4. robust_trend            — Theil-Sen slope when n_obs >= 3; falls back
#                               to (last - first) / (t_last - t_first) when
#                               n_obs == 2; 0 when n_obs <= 1.
#   5. observed_fraction       — fraction of phase days with an observation.
#   6. mean, min, max          — simple block-level aggregates retained for
#                               compatibility / interpretability.
#
# `latest_value` remains computed internally (needed for latest_zscore) but
# is not part of A's own output. `min`/`max`/`range`/`first_value`/`mean`/
# `iqr`/`volatility` were all tested as standalone candidates and dropped
# here as cluster-redundant with `median`/`std`.

A_BINS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    # Day indices are 1-indexed in the raw per-admission data (bin column
    # values 1..15, confirmed against processed_admission_features_csv), NOT
    # 0-indexed. A hardcoded `tuple(range(15))` = (0..14) silently drops day
    # 15 (the most recent day) from every A feature entirely, since day 0
    # never exists in the data (`if d in sub.columns` just silently no-ops
    # for the nonexistent day-0 slot).
    ("P1", tuple(range(1, 16))),
)

A_FEATURES: Tuple[str, ...] = (
    "median",
    "mean",
    "min",
    "max",
    "std",
    "latest_zscore",
    "robust_trend",
    "observed_fraction",
)

_A_EPS = 1e-6


def _compute_a_bin_features(
    values: np.ndarray,
    bin_size: int,
    ref_median: float = 0.0,
    ref_iqr: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Compute the Approach-A feature superset for one (lab, B-bin) block.

    Parameters
    ----------
    values : (n_rows, bin_size) array
        Lab values in this bin. NaN means "not measured on that day".
    bin_size : int
        Days the bin spans (denominator for ``observed_fraction``).
    ref_median, ref_iqr : float
        Training-fold robust statistics for the parent itemid. Used to
        compute ``latest_zscore`` and to normalise deviations for internal
        guards (IQR-based scaling).

    Returns
    -------
    dict with keys ``latest_value, robust_trend, observed_fraction,
    latest_zscore, median, mean, min, max, std``. ``latest_value``/
    ``robust_trend``/``median``/``min``/``max``/``std`` are NaN where
    insufficient data exists; the rest are always numeric (0.0 for empty
    rows). Not every key is part of A's own output (``A_FEATURES``)
    — ``latest_value`` in particular is kept only for internal use by
    ``latest_zscore``.
    """
    n_rows = values.shape[0]
    if values.shape[1] != bin_size:
        raise ValueError(
            f"values has {values.shape[1]} cols but bin_size={bin_size}"
        )

    mask = ~np.isnan(values)
    n_obs = mask.sum(axis=1)
    has_any = n_obs > 0

    # observed_fraction
    observed_fraction = n_obs.astype(np.float64) / float(bin_size)

    # latest_value — most recent observed day in the bin
    latest_value = np.full(n_rows, np.nan, dtype=np.float64)
    if has_any.any():
        rev_mask = mask[has_any][:, ::-1]
        last_idx = bin_size - 1 - np.argmax(rev_mask, axis=1)
        rows = np.where(has_any)[0]
        latest_value[rows] = values[rows, last_idx]

    # robust_trend
    #   n_obs >= 3 → Theil-Sen median pair slope
    #   n_obs == 2 → (last - first) / (t_last - t_first)
    #   n_obs <= 1 → 0
    robust_trend = np.full(n_rows, np.nan, dtype=np.float64)
    if has_any.any():
        i_idx, j_idx = np.triu_indices(bin_size, k=1)
        if len(i_idx) > 0:
            denom = (j_idx - i_idx).astype(np.float64)
            v_diff = values[:, j_idx] - values[:, i_idx]
            with np.errstate(invalid="ignore"):
                slopes = v_diff / denom[None, :]
            valid = mask[:, i_idx] & mask[:, j_idx]
            slopes_masked = np.where(valid, slopes, np.nan)

            # n_obs >= 3: full Theil-Sen median over the valid pair slopes.
            rows_3 = np.where(n_obs >= 3)[0]
            if rows_3.size:
                with np.errstate(all="ignore"):
                    robust_trend[rows_3] = np.nanmedian(slopes_masked[rows_3], axis=1)

            # n_obs == 2: by construction there's exactly one valid pair; its
            # slope is precisely (last - first) / (t_last - t_first). Use the
            # nanmedian of one number to extract it.
            rows_2 = np.where(n_obs == 2)[0]
            if rows_2.size:
                with np.errstate(all="ignore"):
                    robust_trend[rows_2] = np.nanmedian(slopes_masked[rows_2], axis=1)

        # n_obs <= 1: zero (no direction available)
        robust_trend[n_obs <= 1] = 0.0

    # Robust per-cell deviation: signed and absolute.
    denom_iqr = float(max(ref_iqr, 0.0)) + _A_EPS
    with np.errstate(invalid="ignore"):
        d_abs = np.abs(values - ref_median) / denom_iqr  # for persistence

    # latest_zscore — continuous signed z-score of the bin's latest_value.
    # Stability guards:
    #   * clip to [-10, 10] so an IQR≈0 column can't emit absurd magnitudes
    #     that destabilise linear / neural models;
    #   * round to 2 decimals so any value with |z| < 1e-3 collapses to 0
    #     (effectively a noise floor cleaner than naive truncation).
    latest_zscore = np.zeros(n_rows, dtype=np.float64)
    if has_any.any():
        rows_any = np.where(has_any)[0]
        with np.errstate(invalid="ignore"):
            latest_zscore[rows_any] = (
                latest_value[rows_any] - ref_median
            ) / denom_iqr
        # Guard: any residual NaN (shouldn't happen since latest_value is set
        # for rows_any, but be defensive about NaN medians).
        latest_zscore = np.where(np.isnan(latest_zscore), 0.0, latest_zscore)
        latest_zscore = np.clip(latest_zscore, -10.0, 10.0)
        latest_zscore = np.round(latest_zscore, 2)

    # median / std / min / max — whole-block level and spread. std uses ddof=1 (0.0 for
    # exactly one observation, NaN — later sentinel-filled to 0.0 — for zero).
    median = np.full(n_rows, np.nan, dtype=np.float64)
    std = np.full(n_rows, np.nan, dtype=np.float64)
    mean = np.full(n_rows, np.nan, dtype=np.float64)
    min_val = np.full(n_rows, np.nan, dtype=np.float64)
    max_val = np.full(n_rows, np.nan, dtype=np.float64)
    if has_any.any():
        rows = np.where(has_any)[0]
        with np.errstate(all="ignore"):
            median[rows] = np.nanmedian(values[rows], axis=1)
            mean[rows] = np.nanmean(values[rows], axis=1)
            min_val[rows] = np.nanmin(values[rows], axis=1)
            max_val[rows] = np.nanmax(values[rows], axis=1)
    rows_2plus = np.where(n_obs >= 2)[0]
    if rows_2plus.size:
        std[rows_2plus] = np.nanstd(values[rows_2plus], axis=1, ddof=1)
    std[n_obs == 1] = 0.0

    return {
        "latest_value": latest_value,
        "robust_trend": robust_trend,
        "observed_fraction": observed_fraction,
        "latest_zscore": latest_zscore,
        "median": median,
        "mean": mean,
        "min": min_val,
        "max": max_val,
        "std": std,
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

        if feature_threshold:
            cc_top_features = pd.read_csv(f"{saved_data_path}/top_features/feat_imp_summary.csv")
            selected_features = cc_top_features.itemid.head(100)
            data = data[data["itemid"].isin(selected_features)]

        # ------------------------------------------------------------------
        # Approach A — refined single-phase set (median, mean, min, max,
        # std, latest_zscore, robust_trend, observed_fraction) covering
        # the whole 15-day window; ref ranges fit on train fold.
        # ------------------------------------------------------------------
        if feat_type == "A":
            if itemids is None and bins is None:
                ap_df, all_itemids, all_bins = self._compute_a_features(
                    data, target_cohort, return_itemids_bins=True,
                )
            else:
                ap_df = self._compute_a_features(
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
        """Approach A — refined single-phase feature set: the decorrelated-9
        features (see module-level ``A_FEATURES`` docstring) over the
        whole 15-day window, per lab.

        Per-itemid robust statistics ``(median, IQR)`` are fitted on the
        training-fold cohort and used to compute the z-score/persistence
        features. Val/test calls reuse the train-fold stats via the
        ``bins`` piggy-back slot.
        """
        all_admissions = data["hadm_id"].unique()

        # Train call: itemids/bins None → discover everything and fit stats.
        # Val/test call: ``bins`` is a dict carrying the pre-fit stats.
        pathology_stats: Dict[int, Tuple[float, float]] = {}
        if itemids is not None and bins is not None and isinstance(bins, dict):
            all_items = list(itemids)
            all_bins = list(bins.get("__bins__", []))
            pathology_stats = bins.get("__pathology_stats__", {}) or {}
        else:
            all_items = sorted(data["itemid"].unique())
            all_bins = sorted(data["bin"].unique())
            # Fit per-itemid (median, IQR) on train-fold observed values.
            for it in all_items:
                vals = (
                    data.loc[data["itemid"] == it, "value"]
                    .dropna()
                    .to_numpy(dtype=np.float64)
                )
                if vals.size >= 2:
                    med = float(np.median(vals))
                    q1 = float(np.quantile(vals, 0.25))
                    q3 = float(np.quantile(vals, 0.75))
                    iqr = q3 - q1
                else:
                    # Degenerate column: every observation maps to state 0.
                    med, iqr = 0.0, 0.0
                pathology_stats[it] = (med, iqr)

        # Per-day mean per (admission, itemid, bin) — no imputation.
        daily = (
            data.groupby(["hadm_id", "itemid", "bin"])["value"]
                .mean()
                .reindex(pd.MultiIndex.from_product(
                    [all_admissions, all_items, all_bins],
                    names=["hadm_id", "itemid", "bin"],
                ))
                .unstack(level="bin")
        )

        n_admissions = len(all_admissions)
        out_columns: List[str] = []
        out_arrays: List[np.ndarray] = []

        for itemid in all_items:
            ref_median, ref_iqr = pathology_stats.get(itemid, (0.0, 0.0))
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
                feats = _compute_a_bin_features(
                    mat, len(day_idx),
                    ref_median=ref_median, ref_iqr=ref_iqr,
                )
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
        # Sentinel-fill policy — keeps the StandardScaler-fed downstream
        # models (LR/GAM) stable.
        a_df = a_df.fillna(0.0)

        if return_itemids_bins:
            payload = {"__bins__": all_bins, "__pathology_stats__": pathology_stats}
            return a_df, all_items, payload
        return a_df

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
