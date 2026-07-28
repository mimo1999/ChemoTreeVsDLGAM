"""
Distributed-Lag GAM (DL-GAM) classifier — a thin wrapper around the R
``mvgam`` package (Clark & Wells, 2022).

This wrapper does not implement its own model. It only:
  1. reshapes the wide VMD feature matrix (columns ``X_<itemid>_<bin>``) into
     the per-lab ``Value``/``Lag`` matrix pairs that ``mvgam`` expects,
  2. writes them out plus the degenerate ``series``/``time`` fields
     ``mvgam``'s data contract requires,
  3. calls ``mvgam::mvgam()`` with the same distributed-lag formula
     convention shown in the package's own case study
     (https://rpubs.com/NickClark47/mvgam3) — a tensor-product smooth over
     matrix arguments, ``te(Value, Lag, k = c(k_val, k_lag))`` — and
  4. reads back ``predict.mvgam(..., type = "response", summary = TRUE)``.

Everything else (the MCMC sampling, the smooth basis construction, the
trend model) is ``mvgam``'s, not this repo's.

Model
-----
For patient i, lab k, and lag bin ell (1..L, the daily bin index):

    logit(p_i) = beta0 + sum_k te(V_k, Lag)_i + z_i

``te(V_k, Lag)`` is mgcv's matrix-argument tensor-product convention:
mvgam builds this internally as a distributed-lag basis over the value axis
and the lag axis jointly (see the case study above). ``z_i`` is mvgam's
latent trend term, controlled by ``trend_model``. Because this is
terminal-label classification (one row per patient, not a time series),
each patient is a length-1 series and ``trend_model`` defaults to
``"None"`` — there is no within-series autocorrelation to estimate here;
the parameter is exposed rather than hidden so this restriction is explicit
and can be revisited if the data model changes.

Feature-name conventions (from ml_feature_matrix_builder.py, VMD)
------------------------------------------------------------------
    X_<itemid>_<bin>   lab value at a daily bin    -> Value/Lag matrix pair
    <demographic>      age / los / gender / ...    -> ordinary smooth/factor terms

``M_<itemid>_<bin>`` (missingness) and ``D_<itemid>_<bin>`` (delta) columns
are ignored: mvgam's distributed-lag case study has no missingness or
velocity term, and mvgam does not accept NA covariates, so missing lab
values are median-imputed before being handed to R (data preparation, not
a model term). A lab needs at least ``min_lags`` distinct bins to receive a
distributed-lag term; sparser labs are dropped rather than approximated
with a repo-specific fallback that mvgam has no equivalent for.
"""

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from .r_bridge import san as _san, run_r as _run_r

from config.constants import RANDOM_SEED, STATIC_SPLINE_FEATURES, STATIC_CATEGORICAL_FEATURES
from utils.feature_naming import parse_vmd_column
from ._base import BaseWrapperClassifier


class DLGAMClassifier(BaseWrapperClassifier):
    """Distributed-lag GAM classifier — wraps ``mvgam::mvgam()`` (R).

    Parameters
    ----------
    k_lag : int
        Basis dimension over the lag (time-before-discharge) axis, passed
        as the first element of ``te(..., k = c(k_val, k_lag))``.
    k_val : int
        Basis dimension over the value axis, the second ``te(...)`` element.
    max_labs : int
        Cap on the number of labs given a distributed-lag term, ranked by
        point-biserial correlation with the outcome. Bounds model size.
    min_lags : int
        Minimum distinct bins a lab must have to receive a term; sparser
        labs are dropped.
    trend_model : str
        mvgam's latent-trend spec ('None', 'AR1', 'AR2', 'AR3', 'GP', ...).
        Default 'None': each row is an independent patient, not a
        timepoint in a shared series, so there is no autocorrelation for a
        trend model to capture.
    family : str
        mvgam family name for a binary outcome — 'bernoulli' (default) or
        'binomial'.
    backend : str
        mvgam/brms backend, forwarded verbatim ('cmdstanr' default, 'rstan').
    chains, burnin, samples, thin : int
        MCMC sampler configuration, forwarded to ``mvgam()``.
    parallel : bool
        Run chains in parallel, forwarded to ``mvgam()``.
    random_state : int
    verbose : bool
    work_dir : str or Path or None
        Directory for R I/O files (train.csv, test.csv, model.rds, pred.csv).
        Defaults to saved_data/gam_workdir/dl_gam/ relative to the project root.
    timeout : int
        Seconds allowed for the R subprocess (MCMC fitting is much slower
        than a penalized-likelihood fit; default 3600s).
    """

    _DEMO_SPLINE = STATIC_SPLINE_FEATURES
    _DEMO_CAT    = STATIC_CATEGORICAL_FEATURES
    _DEFAULT_WORK_DIR = Path(__file__).resolve().parents[1] / "saved_data" / "gam_workdir" / "dl_gam"

    def __init__(
        self,
        k_lag: int = 4,
        k_val: int = 5,
        max_labs: int = 40,
        min_lags: int = 3,
        trend_model: str = "None",
        family: str = "bernoulli",
        backend: str = "cmdstanr",
        chains: int = 4,
        burnin: int = 500,
        samples: int = 500,
        thin: int = 1,
        parallel: bool = True,
        random_state: int = RANDOM_SEED,
        verbose: bool = True,
        work_dir: "str | Path | None" = None,
        timeout: int = 3600,
    ):
        self.k_lag         = k_lag
        self.k_val         = k_val
        self.max_labs      = max_labs
        self.min_lags      = min_lags
        self.trend_model   = trend_model
        self.family        = family
        self.backend       = backend
        self.chains        = chains
        self.burnin        = burnin
        self.samples       = samples
        self.thin          = thin
        self.parallel      = parallel
        self.random_state  = random_state
        self.verbose       = verbose
        self.work_dir      = Path(work_dir) if work_dir is not None else self._DEFAULT_WORK_DIR
        self.timeout       = timeout

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[DLGAM] {msg}", flush=True)

    def _r_path(self, filename: str) -> str:
        """Return an R-safe (forward-slash) path inside work_dir."""
        return (self.work_dir / filename).as_posix()

    def _demo_kind(self, name: str) -> str:
        """Classify a static covariate as 'cat', 'spline', or 'linear'."""
        nl = name.lower()
        if any(nl == k or nl.startswith(k + "_") for k in self._DEMO_CAT):
            return "cat"
        if any(nl == k or nl.startswith(k + "_") for k in self._DEMO_SPLINE):
            return "spline"
        return "linear"

    # ------------------------------------------------------------------
    # Column parsing
    # ------------------------------------------------------------------

    def _parse_feature_columns(self, feature_names: list[str]):
        """Split column names into per-lab value/lag maps and static covariates.

        Returns
        -------
        val_bins : {lab_id: {bin_idx: col_idx}}
        static   : [(col_idx, col_name)]
        """
        val_bins  = {}
        static    = []
        n_dropped = 0

        for col_idx, col_name in enumerate(feature_names):
            parsed = parse_vmd_column(col_name)
            if parsed is not None and parsed[2] is not None:
                channel, itemid, bin_id = parsed
                if channel == "X":
                    val_bins.setdefault(itemid, {})[bin_id] = col_idx
                    continue
                # Missingness/delta columns have no mvgam analog (see module
                # docstring) — drop rather than silently treating them as
                # generic static covariates.
                n_dropped += 1
                continue
            # Unmatched, or a channel-matched column with no bin (e.g. an
            # aggregate feature_method's X_<itemid>): no lag axis to build a
            # distributed-lag term from, so it's treated as a plain static
            # covariate instead.
            static.append((col_idx, col_name))

        if n_dropped:
            self._log(f"Dropped {n_dropped} M_/D_ columns (no mvgam distributed-lag analog).")

        return val_bins, static

    # ------------------------------------------------------------------
    # Design matrix construction
    # ------------------------------------------------------------------

    def _build_frame(self, X_arr: np.ndarray) -> pd.DataFrame:
        """Assemble the flat DataFrame written to CSV for R.

        For every kept lab and lag position p (1..L) emit column
        v_<lab>_<p> (value). Static covariates retain sanitised names.
        The Value/Lag matrices are reconstructed in R from these columns.
        """
        n_rows = X_arr.shape[0]
        cols   = {}

        for lab_id in self._kept_labs:
            bin_map = self._val_bins[lab_id]
            lab_med = self._lab_median.get(lab_id, 0.0)
            lab_s   = _san(lab_id)

            for p, bin_idx in enumerate(self._lag_grid, start=1):
                if bin_idx in bin_map:
                    v = X_arr[:, bin_map[bin_idx]].astype(float)
                    v = np.where(np.isfinite(v), v, lab_med)
                else:
                    v = np.full(n_rows, lab_med, dtype=float)
                cols[f"v_{lab_s}_{p}"] = v

        for col_idx, col_name in self._static:
            med = self._static_median.get(col_name, 0.0)
            col = X_arr[:, col_idx].astype(float)
            cols[f"s_{_san(col_name)}"] = np.where(np.isfinite(col), col, med)

        return pd.DataFrame(cols, index=np.arange(n_rows))

    # ------------------------------------------------------------------
    # R code generation
    # ------------------------------------------------------------------

    def _build_r_data_block(self, df_var: str = "df") -> str:
        """R code that reconstructs the Value/Lag matrices and mvgam's
        required series/time fields from the flat CSV."""
        L     = len(self._lag_grid)
        lines = [
            f"n <- nrow({df_var})",
            f"Lag <- matrix(rep(seq_len({L}), each = n), nrow = n, ncol = {L})",
            "dat <- list(Lag = Lag)",
            f"if ('y' %in% names({df_var})) dat$y <- {df_var}$y",
            "dat$series <- factor(rep('series1', n))",
            "dat$time <- seq_len(n)",
        ]
        for lab_id in self._kept_labs:
            lab_s = _san(lab_id)
            vcols = ", ".join(f'"v_{lab_s}_{p}"' for p in range(1, L + 1))
            lines.append(f"dat$V_{lab_s} <- as.matrix({df_var}[, c({vcols})])")
        for _, col_name in self._static:
            col_s = _san(col_name)
            if self._demo_kind(col_name) == "cat":
                lines.append(f"dat$s_{col_s} <- as.factor({df_var}$s_{col_s})")
            else:
                lines.append(f"dat$s_{col_s} <- {df_var}$s_{col_s}")
        return "\n".join(lines)

    def _build_r_formula(self) -> str:
        terms = []
        for lab_id in self._kept_labs:
            lab_s = _san(lab_id)
            terms.append(f"te(V_{lab_s}, Lag, k = c({self.k_val}, {self.k_lag}))")
        for _, col_name in self._static:
            col_s = _san(col_name)
            if self._demo_kind(col_name) == "spline":
                terms.append(f"s(s_{col_s}, k = 5)")
            else:
                terms.append(f"s_{col_s}")
        return "y ~ " + (" + ".join(terms) if terms else "1")

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X, y, feature_names=None):
        if feature_names is None:
            feature_names = list(X.columns)
        self.feature_names_in_ = list(feature_names)

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])

        val_bins, static = self._parse_feature_columns(self.feature_names_in_)
        if not val_bins:
            raise ValueError("[DLGAM] No X_<itemid>_<bin> value columns found.")
        self._val_bins  = val_bins
        self._static    = static
        self._lag_grid  = sorted({b for bins in val_bins.values() for b in bins})

        # Score each lab by point-biserial correlation between its mean
        # trajectory and the outcome — rank by predictive relevance.
        self._lab_median: dict[str, float] = {}
        lab_score: dict[str, float] = {}
        y_centered = y_arr.astype(float) - y_arr.mean()
        y_std      = y_arr.std() + 1e-12

        for lab_id, bin_map in val_bins.items():
            vals = X_arr[:, list(bin_map.values())].astype(float)
            obs  = vals[np.isfinite(vals)]
            self._lab_median[lab_id] = float(np.median(obs)) if obs.size else 0.0

            with np.errstate(invalid="ignore"):
                mean_traj = np.nanmean(np.where(np.isfinite(vals), vals, np.nan), axis=1)
            mean_traj = np.where(np.isfinite(mean_traj), mean_traj, self._lab_median[lab_id])

            traj_std = mean_traj.std()
            if traj_std < 1e-9 or not obs.size:
                lab_score[lab_id] = 0.0
            else:
                r = np.mean((mean_traj - mean_traj.mean()) * y_centered) / (traj_std * y_std)
                lab_score[lab_id] = abs(float(r))

        # Labs with >= min_lags bins earn a DL term (capped at max_labs,
        # ranked by score); sparser labs are dropped — mvgam has no
        # equivalent fallback term for them.
        eligible = [lab for lab, bins in val_bins.items() if len(bins) >= self.min_lags]
        eligible.sort(key=lambda lab: lab_score.get(lab, 0.0), reverse=True)
        self._kept_labs = eligible[: self.max_labs]

        self._static_median = {
            col_name: (float(np.median(X_arr[np.isfinite(X_arr[:, ci]), ci]))
                       if np.isfinite(X_arr[:, ci]).any() else 0.0)
            for ci, col_name in static
        }

        self._log(
            f"{len(self.feature_names_in_)} cols -> "
            f"{len(self._kept_labs)} DL labs (L={len(self._lag_grid)} lags), "
            f"{len(static)} static; backend={self.backend}, trend_model={self.trend_model}"
        )

        frame = self._build_frame(X_arr)
        frame.insert(0, "y", y_arr)

        self.work_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.work_dir / "train.csv", index=False)

        formula = self._build_r_formula()
        script  = textwrap.dedent(f"""
            suppressPackageStartupMessages(library(mvgam))
            set.seed({self.random_state})
            df <- read.csv("{self._r_path('train.csv')}")
            {self._build_r_data_block('df')}
            form <- {formula!r}
            cat("[R] n=", n, " terms in formula\\n", sep="")
            fit <- mvgam(as.formula(form), data = dat,
                          family = {self.family}(),
                          trend_model = "{self.trend_model}",
                          backend = "{self.backend}",
                          chains = {self.chains}, burnin = {self.burnin},
                          samples = {self.samples}, thin = {self.thin},
                          parallel = {"TRUE" if self.parallel else "FALSE"},
                          silent = 2)
            saveRDS(fit, "{self._r_path('model.rds')}")
            cat("[R] FIT OK\\n")
        """)
        r_out = _run_r(script, timeout=self.timeout)
        for line in r_out.splitlines():
            if line.strip():
                self._log(f"  {line}")
        if "FIT OK" not in r_out:
            raise RuntimeError("[DLGAM] mvgam fit failed — see R output above.")
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "_kept_labs")
        X_arr  = np.asarray(X, dtype=float)
        n_rows = X_arr.shape[0]
        frame  = self._build_frame(X_arr)
        frame.to_csv(self.work_dir / "test.csv", index=False)

        script = textwrap.dedent(f"""
            suppressPackageStartupMessages(library(mvgam))
            fit <- readRDS("{self._r_path('model.rds')}")
            df  <- read.csv("{self._r_path('test.csv')}")
            {self._build_r_data_block('df')}
            pr <- predict(fit, newdata = dat, type = "response", summary = TRUE)
            write.csv(data.frame(p = pr[, "Estimate"]), "{self._r_path('pred.csv')}", row.names = FALSE)
            cat("[R] PREDICT OK\\n")
        """)
        r_out    = _run_r(script, timeout=self.timeout)
        pred_csv = self.work_dir / "pred.csv"

        if "PREDICT OK" not in r_out or not pred_csv.exists():
            for line in r_out.splitlines():
                if line.strip():
                    self._log(f"  {line}")
            raise RuntimeError("[DLGAM] R predict failed — see output above.")

        p1 = pd.read_csv(pred_csv)["p"].to_numpy()
        if len(p1) != n_rows:
            raise RuntimeError(
                f"[DLGAM] Prediction length mismatch: expected {n_rows}, got {len(p1)}."
            )
        p1 = np.clip(np.where(np.isfinite(p1), p1, 0.5), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p1, p1])

    # predict() / predict_log_proba() are inherited from BaseWrapperClassifier.

    def get_selected_features(self) -> list[str]:
        check_is_fitted(self, "_kept_labs")
        return list(self._kept_labs)
