"""
dgam_wrapper.py  —  Distributed-Lag Dynamic GAM (DGAM) via the mgcv R package.

Why this design
---------------
The chemo-NF task is *terminal-label classification* from a multivariate lab
*trajectory*, not time-series forecasting. A literal port of Clark & Wells'
DGAM (one series per patient, a single Bernoulli label at the final bin, a
per-patient AR(1) latent trend) collapses to "a GAM on the final bin's labs
plus an irreducible per-patient noise term" — it wastes the trajectory and
adds variance instead of signal.

This wrapper instead realises the *distributed-lag* branch of the DGAM family
that the paper explicitly cites (Gasparrini 2011, "smooth distributed lag
covariate functions"). For each lab k the contribution to the logit is a
smooth, lag-weighted functional of the whole trajectory:

    logit(p_i) = beta0
                 + sum_k  SUM_ell  w_k(ell) * value_{i,k,ell}        (linear DL)
                 [ + sum_k SUM_ell f_k(ell, value_{i,k,ell})  ]       (non-linear DL)
                 [ + sum_k SUM_ell m_k(ell) * missing_{i,k,ell} ]     (missingness DL)
                 + GAM(demographics)

where ell indexes the daily lag bin (15 = day of discharge, 1 = 14 days prior)
and w_k(.) / f_k(.) / m_k(.) are penalised smooths estimated by REML. This is
mgcv's "summation convention" / linear-functional-term machinery: a smooth of a
matrix covariate ``Lag`` with a matrix ``by`` argument evaluates to
``sum_ell smooth(Lag[,ell]) * by[,ell]`` for each row.

Properties vs. the AR(1) state-space port
------------------------------------------
  * One row per patient — no NA-response rows, the entire trajectory enters
    the likelihood.
  * w_k(ell) learns *which days before discharge matter* for each lab — an
    interpretable temporal weighting, the "dynamic" part of the DGAM.
  * Generalises to unseen patients through ``predict.gam`` (no per-series
    latent state to marginalise, no Stan ``n_series`` constraint, no batching).
  * Fit by REML with ``bam`` — seconds-to-minutes per fold, far cheaper and
    more stable than MCMC / Laplace state-space estimation.

Feature-name conventions (from ml_feature_matrix_builder.py, VMD/concatenate)
----------------------------------------------------------------------------
    X_<itemid>_<bin>   lab value at a daily bin   -> distributed-lag value term
    M_<itemid>_<bin>   1 = not measured at bin    -> distributed-lag missingness
    D_<itemid>_<bin>   rate-of-change at each bin  -> optional DL velocity term
    <demographic>      age / los / gender / ...   -> ordinary GAM terms

Columns that don't match the VMD lab pattern are treated as static covariates.
A lab needs at least ``min_lags`` distinct bins present in the (already
feature-selected) matrix to earn a distributed-lag term; sparser labs fall
back to plain linear terms so nothing is silently dropped.
"""

import os
import tempfile
import textwrap
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from .r_bridge import RE_X as _RE_X, RE_M as _RE_M, RE_D as _RE_D
from .r_bridge import san as _san, run_r as _run_r


class DGAMClassifier(ClassifierMixin, BaseEstimator):
    """Distributed-lag Dynamic GAM classifier (mgcv backend).

    Parameters
    ----------
    dl_nonlinear : bool
        If True, each lab gets a non-linear DLNM crossbasis ``te(Lag, Value)``
        (value effect varies smoothly with lag). If False (default), a linear
        signal-regression term ``s(Lag, by=Value)`` — cheaper and more stable.
    k_lag : int
        Basis dimension over the lag (time-before-discharge) axis.
    k_val : int
        Basis dimension over the value axis (only used when ``dl_nonlinear``).
    include_delta : bool
        Add a distributed-lag velocity term ``s(Lag, by=Delta)`` per lab,
        capturing the rate of change of the trajectory at each lag. This is the
        closest approximation of DGAM's latent AR state within the single-label
        classification setting: the model sees both *position* (V) and *velocity*
        (D) of each lab trajectory, weighted by a smooth lag function.
    include_missingness : bool
        Add a distributed-lag missingness term ``s(Lag, by=Missing)`` per lab,
        capturing informative sampling over time.
    max_labs : int
        Cap on the number of labs that receive a distributed-lag term (ranked
        by trajectory variance). Limits model size / fit time.
    min_lags : int
        Minimum distinct bins a lab must have to earn a DL term; sparser labs
        become plain linear terms.
    method : str
        mgcv ``bam`` smoothing-parameter method ('fREML' default).
    random_state : int
    verbose : bool
    """

    _DEMO_SPLINE = ("age", "los", "weight", "height", "bmi")
    _DEMO_CAT    = ("gender", "sex", "ethnicity", "race")

    def __init__(
        self,
        dl_nonlinear: bool = False,
        k_lag: int = 4,
        k_val: int = 5,
        include_delta: bool = False,
        include_missingness: bool = False,
        value_spline: bool = False,   # scalar s(mean value) is collinear with the
                                      # DL term and destabilises REML — keep off
        max_labs: int = 40,
        min_lags: int = 3,
        linear_fallback: bool = False,
        method: str = "fREML",
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.dl_nonlinear        = dl_nonlinear
        self.k_lag               = k_lag
        self.k_val               = k_val
        self.include_delta       = include_delta
        self.include_missingness = include_missingness
        self.value_spline        = value_spline
        self.max_labs            = max_labs
        self.min_lags            = min_lags
        self.linear_fallback     = linear_fallback
        self.method              = method
        self.random_state        = random_state
        self.verbose             = verbose

    # ── logging ─────────────────────────────────────────────────────────────────
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[DGAM] {msg}", flush=True)

    # ── parse the wide VMD matrix into per-lab lag structure ────────────────────
    def _parse(self, feature_names: list[str]):
        """Split columns into value-lag / missing-lag maps and static columns.

        Returns
        -------
        val_bins   : {itemid: {bin: col_idx}}
        miss_bins  : {itemid: {bin: col_idx}}
        delta_bins : {itemid: {bin: col_idx}}
        static     : list[(col_idx, name)]
        """
        val_bins:   Dict[str, Dict[int, int]] = {}
        miss_bins:  Dict[str, Dict[int, int]] = {}
        delta_bins: Dict[str, Dict[int, int]] = {}
        static:     list[Tuple[int, str]]     = []
        for i, fn in enumerate(feature_names):
            mx = _RE_X.match(fn)
            if mx:
                val_bins.setdefault(mx.group(1), {})[int(mx.group(2))] = i
                continue
            mm = _RE_M.match(fn)
            if mm:
                miss_bins.setdefault(mm.group(1), {})[int(mm.group(2))] = i
                continue
            md = _RE_D.match(fn)
            if md:
                delta_bins.setdefault(md.group(1), {})[int(md.group(2))] = i
                continue
            static.append((i, fn))
        return val_bins, miss_bins, delta_bins, static

    def _demo_kind(self, name: str) -> str:
        nl = name.lower()
        if any(k == nl or nl.startswith(k + "_") or k in nl for k in self._DEMO_CAT):
            return "cat"
        if any(k == nl or nl.startswith(k + "_") or k in nl for k in self._DEMO_SPLINE):
            return "spline"
        return "linear"

    # ── build the per-patient design (matrices flattened to wide CSV) ───────────
    def _build_frame(self, X: np.ndarray) -> pd.DataFrame:
        """Assemble the flat DataFrame written to CSV for R.

        For every kept lab ``g`` and lag position ``p`` (1..L) emit columns
        ``v_<g>_<p>`` (value) and ``m_<g>_<p>`` (missingness). Static covariates
        keep sanitised names. The lag grid itself is rebuilt inside R.
        """
        n = X.shape[0]
        cols: Dict[str, np.ndarray] = {}

        for g in self._kept_labs:
            bin_map   = self._val_bins[g]
            miss_map  = self._miss_bins.get(g, {})
            delta_map = self._delta_bins.get(g, {})
            med       = self._lab_median.get(g, 0.0)
            v_lags = []
            for p, b in enumerate(self._lag_grid, start=1):
                # value
                if b in bin_map:
                    v = X[:, bin_map[b]].astype(float)
                    v = np.where(np.isfinite(v), v, med)
                else:
                    v = np.full(n, med, dtype=float)
                cols[f"v_{_san(g)}_{p}"] = v
                v_lags.append(v)
                # missingness (1 = not measured)
                if b in miss_map:
                    m = X[:, miss_map[b]].astype(float)
                    m = np.where(np.isfinite(m), m, 1.0)
                elif b in bin_map:
                    m = (~np.isfinite(X[:, bin_map[b]].astype(float))).astype(float)
                else:
                    m = np.ones(n, dtype=float)
                cols[f"m_{_san(g)}_{p}"] = m
                # delta / rate-of-change (velocity)
                if self.include_delta:
                    if b in delta_map:
                        d = X[:, delta_map[b]].astype(float)
                        d = np.where(np.isfinite(d), d, 0.0)
                    else:
                        d = np.zeros(n, dtype=float)
                    cols[f"d_{_san(g)}_{p}"] = d
            # scalar per-lab mean value (non-linear level effect via s(sv_g))
            if self.value_spline:
                cols[f"sv_{_san(g)}"] = np.mean(np.column_stack(v_lags), axis=1)

        # plain linear labs (too few bins for a DL term) — averaged trajectory
        for g in self._linear_labs:
            idx = list(self._val_bins[g].values())
            vals = X[:, idx].astype(float)
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(np.where(np.isfinite(vals), vals, np.nan), axis=1)
            mean = np.where(np.isfinite(mean), mean, self._lab_median.get(g, 0.0))
            cols[f"lin_{_san(g)}"] = mean

        # static covariates
        for ci, name in self._static:
            col = X[:, ci].astype(float)
            med = self._static_median.get(name, 0.0)
            cols[f"s_{_san(name)}"] = np.where(np.isfinite(col), col, med)

        return pd.DataFrame(cols, index=np.arange(n))

    # ── R formula + data-reconstruction code ────────────────────────────────────
    def _r_data_block(self, df_var: str = "df") -> str:
        """R code that rebuilds matrices + a `dat` list from the flat CSV."""
        L = len(self._lag_grid)
        lines = [
            f"n <- nrow({df_var})",
            f"Lag <- matrix(rep(seq_len({L}), each = n), nrow = n, ncol = {L})",
            "dat <- list(Lag = Lag)",
            f"if ('y' %in% names({df_var})) dat$y <- {df_var}$y",
        ]
        for g in self._kept_labs:
            gs = _san(g)
            vcols = ", ".join(f'"v_{gs}_{p}"' for p in range(1, L + 1))
            mcols = ", ".join(f'"m_{gs}_{p}"' for p in range(1, L + 1))
            dcols = ", ".join(f'"d_{gs}_{p}"' for p in range(1, L + 1))
            lines.append(f"dat$V_{gs} <- as.matrix({df_var}[, c({vcols})])")
            if self.include_missingness:
                lines.append(f"dat$M_{gs} <- as.matrix({df_var}[, c({mcols})])")
            if self.include_delta:
                lines.append(f"dat$D_{gs} <- as.matrix({df_var}[, c({dcols})])")
            if self.value_spline:
                lines.append(f"dat$sv_{gs} <- {df_var}$sv_{gs}")
        for g in self._linear_labs:
            gs = _san(g)
            lines.append(f"dat$lin_{gs} <- {df_var}$lin_{gs}")
        for _, name in self._static:
            ns = _san(name)
            kind = self._demo_kind(name)
            if kind == "cat":
                lines.append(f"dat$s_{ns} <- as.factor({df_var}$s_{ns})")
            else:
                lines.append(f"dat$s_{ns} <- {df_var}$s_{ns}")
        return "\n".join(lines)

    def _r_formula(self) -> str:
        terms: list[str] = []
        for g in self._kept_labs:
            gs = _san(g)
            if self.dl_nonlinear:
                # Penalised DLNM crossbasis (Gasparrini 2017): the summation
                # convention over the two matched matrices yields
                #   sum_ell f(Lag[i,ell], Value[i,ell])
                # i.e. a non-linear value effect that varies smoothly by lag.
                terms.append(
                    f"te(Lag, V_{gs}, bs = c('cr','cr'), "
                    f"k = c({self.k_lag}, {self.k_val}))"
                )
            else:
                terms.append(f"s(Lag, by = V_{gs}, bs = 'cr', k = {self.k_lag})")
                if self.value_spline:
                    # non-linear effect of the lab's overall level
                    terms.append(f"s(sv_{gs}, bs = 'cr', k = {self.k_val})")
            if self.include_delta:
                terms.append(f"s(Lag, by = D_{gs}, bs = 'cr', k = {self.k_lag})")
            if self.include_missingness:
                terms.append(f"s(Lag, by = M_{gs}, bs = 'cr', k = {self.k_lag})")
        for g in self._linear_labs:
            terms.append(f"lin_{_san(g)}")
        for _, name in self._static:
            ns = _san(name)
            kind = self._demo_kind(name)
            if kind == "spline":
                terms.append(f"s(s_{ns}, k = 5)")
            else:                              # linear or factor
                terms.append(f"s_{ns}")
        rhs = " + ".join(terms) if terms else "1"
        return "y ~ " + rhs

    # ── sklearn API ─────────────────────────────────────────────────────────────
    def fit(self, X, y, feature_names=None, cat_feature_names=None):
        if feature_names is None:
            feature_names = (list(X.columns) if hasattr(X, "columns")
                             else [f"f{i}" for i in range(X.shape[1])])
        self.feature_names_in_ = list(feature_names)
        Xa = np.asarray(X.values if hasattr(X, "values") else X, dtype=float)
        ya = np.asarray(y).astype(int)
        self.classes_ = unique_labels(ya)

        val_bins, miss_bins, delta_bins, static = self._parse(self.feature_names_in_)
        if not val_bins:
            raise ValueError("[DGAM] No X_<itemid>_<bin> value columns found.")
        self._val_bins   = val_bins
        self._miss_bins  = miss_bins
        self._delta_bins = delta_bins
        self._static     = static

        # Lag grid = sorted union of every bin index seen across all labs.
        all_bins = sorted({b for m in val_bins.values() for b in m})
        self._lag_grid = all_bins

        # Per-lab median (for imputing absent cells) and a *supervised* score:
        # |point-biserial correlation| between the lab's across-bin mean
        # trajectory and the outcome. This ranks labs by predictive relevance
        # rather than raw variance, so the limited DL budget is spent on the
        # most informative trajectories.
        self._lab_median: Dict[str, float] = {}
        lab_score: Dict[str, float] = {}
        y_c = ya.astype(float) - ya.mean()
        y_sd = ya.std() + 1e-12
        for g, m in val_bins.items():
            vals = Xa[:, list(m.values())].astype(float)
            obs = vals[np.isfinite(vals)]
            self._lab_median[g] = float(np.median(obs)) if obs.size else 0.0
            with np.errstate(invalid="ignore"):
                mean_traj = np.nanmean(np.where(np.isfinite(vals), vals, np.nan), axis=1)
            mean_traj = np.where(np.isfinite(mean_traj), mean_traj, self._lab_median[g])
            sd = mean_traj.std()
            if sd < 1e-9 or not obs.size:
                lab_score[g] = 0.0
            else:
                lab_score[g] = abs(float(np.mean((mean_traj - mean_traj.mean()) * y_c)
                                          / (sd * y_sd)))

        # Top max_labs labs (with >= min_lags bins) become distributed-lag terms.
        eligible = [g for g, m in val_bins.items() if len(m) >= self.min_lags]
        eligible.sort(key=lambda g: lab_score.get(g, 0.0), reverse=True)
        self._kept_labs = eligible[: self.max_labs]
        kept_set        = set(self._kept_labs)
        # Remaining labs: optionally enter as a single penalised linear term
        # (their across-bin mean); off by default to avoid overfitting.
        self._linear_labs = (
            [g for g in val_bins if g not in kept_set] if self.linear_fallback else []
        )

        self._static_median = {
            name: (float(np.median(Xa[np.isfinite(Xa[:, ci]), ci]))
                   if np.isfinite(Xa[:, ci]).any() else 0.0)
            for ci, name in static
        }

        extras = (
            (", +delta" if self.include_delta else "")
            + (", +missingness" if self.include_missingness else "")
        )
        self._log(
            f"{len(self.feature_names_in_)} cols -> "
            f"{len(self._kept_labs)} DL labs (L={len(self._lag_grid)} lags, "
            f"{'nonlinear te()' if self.dl_nonlinear else 'linear s(Lag,by=)'}"
            f"{extras}), "
            f"{len(self._linear_labs)} linear labs, {len(static)} static"
        )

        frame = self._build_frame(Xa)
        frame.insert(0, "y", ya)

        self._tmpdir   = tempfile.mkdtemp(prefix="dgam_")
        train_csv      = os.path.join(self._tmpdir, "train.csv")
        self._model_rds = os.path.join(self._tmpdir, "model.rds")
        frame.to_csv(train_csv, index=False)

        formula = self._r_formula()
        script = textwrap.dedent(f"""
            suppressPackageStartupMessages(library(mgcv))
            set.seed({self.random_state})
            df <- read.csv("{train_csv.replace(os.sep, '/')}")
            {self._r_data_block('df')}
            form <- {formula!r}
            cat("[R] n=", n, " terms in formula\\n", sep="")
            fit <- bam(as.formula(form), data = dat, family = binomial(),
                       method = "{self.method}", discrete = TRUE)
            saveRDS(fit, "{self._model_rds.replace(os.sep, '/')}")
            cat("[R] edf=", round(sum(fit$edf), 1),
                " dev.expl=", round(summary(fit)$dev.expl, 4), "\\n", sep="")
            cat("[R] FIT OK\\n")
        """)
        out = _run_r(script)
        for ln in out.splitlines():
            if ln.strip():
                self._log(f"  {ln}")
        if "FIT OK" not in out:
            raise RuntimeError("[DGAM] mgcv fit failed — see R output above.")
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "_kept_labs")
        Xa = np.asarray(X.values if hasattr(X, "values") else X, dtype=float)
        n = Xa.shape[0]
        frame = self._build_frame(Xa)
        test_csv = os.path.join(self._tmpdir, "test.csv")
        out_csv  = os.path.join(self._tmpdir, "pred.csv")
        frame.to_csv(test_csv, index=False)

        script = textwrap.dedent(f"""
            suppressPackageStartupMessages(library(mgcv))
            fit <- readRDS("{self._model_rds.replace(os.sep, '/')}")
            df <- read.csv("{test_csv.replace(os.sep, '/')}")
            {self._r_data_block('df')}
            p <- as.numeric(predict(fit, newdata = dat, type = "response"))
            write.csv(data.frame(p = p), "{out_csv.replace(os.sep, '/')}",
                      row.names = FALSE)
            cat("[R] PREDICT OK\\n")
        """)
        out = _run_r(script)
        if not os.path.exists(out_csv) or "PREDICT OK" not in out:
            for ln in out.splitlines():
                if ln.strip():
                    self._log(f"  {ln}")
            self._log("prediction failed — returning 0.5")
            p1 = np.full(n, 0.5)
        else:
            p1 = pd.read_csv(out_csv)["p"].to_numpy()
            p1 = np.clip(np.where(np.isfinite(p1), p1, 0.5), 1e-6, 1 - 1e-6)
            if len(p1) != n:
                p1 = np.resize(p1, n)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    def get_selected_features(self) -> list[str]:
        check_is_fitted(self, "_kept_labs")
        return list(self._kept_labs)
