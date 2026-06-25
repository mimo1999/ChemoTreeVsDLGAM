"""
IGANN (Interpretable Generalized Additive Neural Network) wrapper for the
ChemoTreeVsDL pipeline.

Pipeline: StandardScaler -> IGANN

IGANN fits one ELM (extreme-learning-machine) shape function per feature using
boosting across per-feature ELMs (Kraus et al., https://github.com/MathiasKraus/igann).

Feature selection and NaN-filling are handled upstream by DataLoader.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from sklearn.preprocessing import StandardScaler

from igann import IGANN


class IGANNWrapperClassifier(ClassifierMixin, BaseEstimator):
    """IGANN wrapped with StandardScaler.

    Parameters
    ----------
    n_estimators : int, default 5000
        Cap on boosting iterations (early_stopping usually fires first).
    boost_rate : float, default 0.1
        Learning rate for the boosting step that combines per-feature ELMs.
    elm_scale : float, default 1.0
        Scale of the ELM hidden-layer initialisation.
    elm_alpha : float, default 1.0
        L2 regularisation strength inside each ELM.
    n_hid : int, default 10
        Hidden units per ELM shape function.
    init_reg : float, default 1.0
        Initial regularisation applied to the linear model IGANN bootstraps off of.
    act : str, default "elu"
        Activation inside each ELM.
    early_stopping : int, default 50
        IGANN patience — stop boosting after this many rounds without
        validation improvement.
    device : str, default "cpu"
        CPU is typically faster than CUDA on small clinical datasets.
    verbose : int, default 0
    random_state : int, default 1
    """

    def __init__(
        self,
        n_estimators: int = 5000,
        boost_rate: float = 0.1,
        elm_scale: float = 1.0,
        elm_alpha: float = 1.0,
        n_hid: int = 10,
        init_reg: float = 1.0,
        act: str = "elu",
        early_stopping: int = 50,
        device: str = "cpu",
        verbose: int = 0,
        random_state: int = 1,
    ):
        self.n_estimators = n_estimators
        self.boost_rate = boost_rate
        self.elm_scale = elm_scale
        self.elm_alpha = elm_alpha
        self.n_hid = n_hid
        self.init_reg = init_reg
        self.act = act
        self.early_stopping = early_stopping
        self.device = device
        self.verbose = verbose
        self.random_state = random_state

    def fit(
        self,
        X,
        y,
        feature_names: Iterable[str] | None = None,
        val_set: tuple | None = None,
    ):
        """Fit IGANN.

        Parameters
        ----------
        X, y : training data (may be oversampled).
        feature_names : column labels forwarded from the pipeline.
        val_set : optional (X_val, y_val) drawn from the *original*
            (non-oversampled) distribution. When provided, IGANN uses it
            for early stopping instead of splitting the training data
            internally. This gives more reliable early stopping against
            the real class distribution and typically improves AUC-ROC on
            imbalanced datasets.
        """
        if feature_names is None:
            feature_names = list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])]
        self.selected_features_ = list(feature_names)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        igann_val_set = None
        if val_set is not None:
            X_v, y_v = val_set
            X_v_scaled = self.scaler_.transform(X_v)
            # IGANN._preprocess_feature_matrix sorts columns alphabetically;
            # val_set[0] bypasses that path so we replicate the sort here.
            sorted_cols = sorted(self.selected_features_)
            col_order = [self.selected_features_.index(c) for c in sorted_cols]
            X_v_sorted = X_v_scaled[:, col_order]
            igann_val_set = (
                torch.tensor(X_v_sorted, dtype=torch.float32),
                torch.from_numpy(np.asarray(y_v).ravel().astype(np.float32)),
            )

        print(
            f"[IGANN] Training with {X_scaled.shape[1]} features | "
            f"n_estimators={self.n_estimators} | boost_rate={self.boost_rate} | "
            f"n_hid={self.n_hid} | early_stopping={self.early_stopping} | "
            f"val_set={'yes' if igann_val_set is not None else 'internal split'}"
        )

        self.igann_ = IGANN(
            task="classification",
            n_hid=self.n_hid,
            n_estimators=self.n_estimators,
            boost_rate=self.boost_rate,
            init_reg=self.init_reg,
            elm_scale=self.elm_scale,
            elm_alpha=self.elm_alpha,
            act=self.act,
            early_stopping=self.early_stopping,
            device=self.device,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.selected_features_)
        y_series = pd.Series(np.asarray(y).ravel())
        self.igann_.fit(X_scaled_df, y_series, val_set=igann_val_set)
        self.classes_ = np.array([0, 1])
        return self

    def _transform_X(self, X):
        check_is_fitted(self, "igann_")
        X_scaled = self.scaler_.transform(X)
        return pd.DataFrame(X_scaled, columns=self.selected_features_)

    def predict_proba(self, X):
        proba = np.asarray(self.igann_.predict_proba(self._transform_X(X)))
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        return proba

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_log_proba(self, X):
        return np.log(np.clip(self.predict_proba(X), 1e-10, 1.0))

    def get_selected_features(self):
        check_is_fitted(self, "igann_")
        return list(self.selected_features_)

    def get_igann(self):
        check_is_fitted(self, "igann_")
        return self.igann_
