"""
NAM (Neural Additive Model, Agarwal et al., 2021) wrapper for the
ChemoTreeVsDL pipeline.

Contains the batched PyTorch architecture (`_BatchedNAM`, `penalized_bce`)
and the sklearn-API adapter built on top of it (`NAMWrapperClassifier`):
preprocessing, the training loop with early stopping, and the sklearn
fit/predict_proba/predict contract.

Preprocessing:
    SimpleImputer (median) -> MinMax scaling to [0, 1]
    MinMax scaling is required because exp(w_i) with w_i ~= 4 (the ExU
    activation's weight init) can explode on unscaled inputs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from config.constants import RANDOM_SEED
from ._base import BaseWrapperClassifier


# ---------------------------------------------------------------------------
# Batched PyTorch architecture
#
# One sub-network per feature, each a small MLP with an ExU first layer:
#     h_i(x_i) = clamp(ReLU((x_i - b_i) * exp(w_i)), 0, 1)
# The model prediction is the sum of all per-feature outputs plus a scalar
# bias:
#     logit(x) = sum_i f_i(x_i) + beta
# This gives the same additive interpretability as a GAM, but with learned
# nonlinear shape functions.
#
# Batched forward pass:
#     The original NAM repo iterates over individual FeatureNN objects in
#     Python, which is O(F) sequential PyTorch calls per batch. This
#     implementation instead stacks all feature weights into 3-D tensors and
#     runs all F feature networks in a single kernel sweep:
#
#         (B, F) -> ExU -> (B, F, U) -> Linear(s) -> (B, F, 1) -> sum_F -> (B,)
#
#     This gives a ~50-100x wall-clock speedup on CPU for typical tabular
#     widths (F ~= 50-250).
# ---------------------------------------------------------------------------

class _ExU(nn.Module):
    """ExU activation applied jointly to all F features in one pass.

    Computes h_i(x_i) = clamp(ReLU((x_i − b_i) · exp(w_i)), 0, 1) for every
    feature i simultaneously.

    Input  : (B, F)
    Output : (B, F, U)   — U learned basis responses per feature
    """

    def __init__(self, n_features: int, n_units: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_features, n_units))
        self.bias    = nn.Parameter(torch.empty(n_features))
        self._init_params()

    def _init_params(self) -> None:
        # Agarwal et al.: W ~ N(4, 0.5) clamped to [3, 5]; b ~ N(0, 0.5).
        with torch.no_grad():
            self.weights.normal_(mean=4.0, std=0.5).clamp_(3.0, 5.0)
            self.bias.normal_(mean=0.0, std=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Broadcast: (B, F) → (B, F, 1); weights: (F, U) → (1, F, U)
        diff = (x - self.bias).unsqueeze(-1)
        out  = diff * torch.exp(self.weights).unsqueeze(0)
        return F.relu(out).clamp(max=1.0)


class _GroupedLinear(nn.Module):
    """Independent linear layers for each of F features, run in a single bmm.

    Applies an independent linear transformation to each feature's hidden
    representation without mixing information across features.

    Input  : (B, F, in_dim)
    Output : (B, F, out_dim)
    """

    def __init__(
        self,
        n_features: int,
        in_dim: int,
        out_dim: int,
        activation: bool = False,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.weights = nn.Parameter(torch.empty(n_features, in_dim, out_dim))
        self.bias    = nn.Parameter(torch.zeros(n_features, out_dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, F, in) → (F, B, in) @ (F, in, out) → (F, B, out) → (B, F, out)
        out = torch.bmm(x.permute(1, 0, 2), self.weights).permute(1, 0, 2)
        out = out + self.bias.unsqueeze(0)
        return F.relu(out) if self.activation else out


class _BatchedNAM(nn.Module):
    """All F feature sub-networks executed in a single forward pass.

    Returns (logit, feature_contributions):
        logit                : (B,)   — raw log-odds
        feature_contributions: (B, F) — per-feature additive output before sum
    """

    def __init__(
        self,
        n_features: int,
        n_units: int,
        hidden_sizes: list[int],
        dropout: float,
        feature_dropout: float,
    ) -> None:
        super().__init__()
        self.feature_dropout = feature_dropout

        # Build per-feature network: ExU → [hidden LinReLU layers] → Linear(1)
        layers: list[nn.Module] = [_ExU(n_features, n_units)]
        prev = n_units
        for h in hidden_sizes:
            layers.append(_GroupedLinear(n_features, prev, h, activation=True))
            prev = h
        layers.append(_GroupedLinear(n_features, prev, 1, activation=False))

        self.layers  = nn.ModuleList(layers)
        self.dropout = nn.Dropout(p=dropout)
        self.bias    = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.layers[0](x)      # ExU: (B, F) → (B, F, U)
        out = self.dropout(out)
        for layer in self.layers[1:]:
            out = self.dropout(layer(out))

        # (B, F, 1) → (B, F)
        contributions = out.squeeze(-1)

        # Feature dropout: zero entire feature sub-networks randomly per batch.
        # Applied before summing so logit is consistent with masked contributions.
        if self.training and self.feature_dropout > 0.0:
            keep = (
                torch.rand(contributions.shape[1], device=contributions.device)
                >= self.feature_dropout
            ).float()
            contributions = contributions * keep.unsqueeze(0)

        logit = contributions.sum(dim=-1) + self.bias
        return logit, contributions


def penalized_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    contributions: torch.Tensor,
    model: _BatchedNAM,
    output_regularization: float,
    l2_regularization: float,
) -> torch.Tensor:
    """BCE loss + output regularization + L2 on the output linear layer only.

    L2 is deliberately restricted to the final linear layer and NOT applied to
    ExU weights.  ExU weights are initialized to mean ≈ 4; pulling them toward
    zero with L2 collapses the activation's effective slope (exp(w) → 1).
    """
    loss = F.binary_cross_entropy_with_logits(logits.view(-1), targets.view(-1))

    if output_regularization > 0.0:
        loss = loss + output_regularization * contributions.pow(2).mean()

    if l2_regularization > 0.0:
        output_layer = model.layers[-1]
        l2 = output_layer.weights.pow(2).sum() + output_layer.bias.pow(2).sum()
        loss = loss + l2_regularization * l2 / contributions.shape[1]

    return loss


# ---------------------------------------------------------------------------
# sklearn-compatible wrapper
# ---------------------------------------------------------------------------

class NAMWrapperClassifier(BaseWrapperClassifier):
    """Neural Additive Model (Agarwal et al., 2021) with a scikit-learn API.

    Preprocessing is handled internally: median imputation followed by MinMax
    scaling to [0, 1] (required for ExU numerical stability).

    Parameters
    ----------
    num_basis_functions : int
        ExU units per feature.  The NAM paper uses 1000 for small datasets;
        64 works well for wide tabular settings (F ≈ 50–250).
    hidden_sizes : list[int]
        Extra hidden layers appended after ExU.  Empty list = ExU → Linear(1).
    dropout : float
        Dropout probability applied after every layer inside the feature nets.
    feature_dropout : float
        Probability of zeroing an entire feature sub-network per training batch.
        Encourages feature independence.
    output_regularization : float
        Penalty coefficient on the mean-squared per-feature contribution.
    l2_regularization : float
        L2 penalty coefficient on the output linear layer weights.
    lr : float
        AdamW learning rate.
    num_epochs : int
        Maximum number of training epochs.
    batch_size : int
    early_stopping_patience : int
        Stop training if validation BCE does not improve for this many epochs.
    val_fraction : float
        Fraction of training data held out for early stopping.
    device : str
        Requested device ('cuda' or 'cpu').  Falls back to CPU if CUDA is
        unavailable.
    random_state : int
    verbose : int
        0 = silent.  1 = log every 10 epochs.
    """

    def __init__(
        self,
        num_basis_functions: int = 64,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.1,
        feature_dropout: float = 0.0,
        output_regularization: float = 0.1,
        l2_regularization: float = 0.05,
        lr: float = 0.01,
        num_epochs: int = 100,
        batch_size: int = 1024,
        early_stopping_patience: int = 20,
        val_fraction: float = 0.15,
        device: str = "cuda",
        random_state: int = RANDOM_SEED,
        verbose: int = 1,
    ) -> None:
        self.num_basis_functions    = num_basis_functions
        self.hidden_sizes           = hidden_sizes
        self.dropout                = dropout
        self.feature_dropout        = feature_dropout
        self.output_regularization  = output_regularization
        self.l2_regularization      = l2_regularization
        self.lr                     = lr
        self.num_epochs             = num_epochs
        self.batch_size             = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.val_fraction           = val_fraction
        self.device                 = device
        self.random_state           = random_state
        self.verbose                = verbose

    # ------------------------------------------------------------------
    # Internal preprocessing (fit-time and transform-time)
    # ------------------------------------------------------------------

    def _fit_preprocessor(self, X: np.ndarray) -> np.ndarray:
        """Fit median imputer + MinMax scaler; drop zero-variance columns."""
        self.imputer_ = SimpleImputer(strategy="median")
        X_imp = np.asarray(self.imputer_.fit_transform(X), dtype=np.float32)

        col_min   = X_imp.min(axis=0)
        col_range = X_imp.max(axis=0) - col_min

        # Drop columns with no variation to avoid division-by-zero in scaling.
        self.valid_cols_  = col_range > 1e-8
        n_dropped = int((~self.valid_cols_).sum())
        if n_dropped:
            print(f"[NAM] Dropped {n_dropped} zero-variance column(s) after selection.")

        self.col_min_   = col_min[self.valid_cols_]
        self.col_scale_ = col_range[self.valid_cols_]

        return self._scale(X_imp[:, self.valid_cols_])

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X_imp = np.asarray(self.imputer_.transform(X), dtype=np.float32)
        return self._scale(X_imp[:, self.valid_cols_])

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.col_min_) / self.col_scale_, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _make_val_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stratified validation split; falls back to random if classes are too sparse."""
        idx = np.arange(len(y))
        try:
            tr_idx, val_idx = train_test_split(
                idx,
                test_size=self.val_fraction,
                random_state=self.random_state,
                stratify=y,
            )
        except ValueError:
            rng = np.random.RandomState(self.random_state)
            val_size = max(1, int(len(y) * self.val_fraction))
            val_idx  = rng.choice(len(y), size=val_size, replace=False)
            tr_idx   = np.setdiff1d(idx, val_idx)

        return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

    def _train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        device: torch.device,
    ) -> _BatchedNAM:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        model = _BatchedNAM(
            n_features=X_train.shape[1],
            n_units=self.num_basis_functions,
            hidden_sizes=list(self.hidden_sizes or []),
            dropout=self.dropout,
            feature_dropout=self.feature_dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs, eta_min=self.lr * 0.01
        )

        X_tr_t  = torch.as_tensor(X_train, device=device)
        y_tr_t  = torch.as_tensor(y_train, device=device)
        X_val_t = torch.as_tensor(X_val,   device=device)
        y_val_t = torch.as_tensor(y_val,   device=device)

        best_val_loss  = float("inf")
        best_state     = None
        patience_count = 0

        for epoch in range(1, self.num_epochs + 1):
            model.train()
            perm       = torch.randperm(len(X_tr_t), device=device)
            train_loss = 0.0
            n_batches  = 0

            for start in range(0, len(X_tr_t), self.batch_size):
                batch_idx = perm[start : start + self.batch_size]
                Xb, yb   = X_tr_t[batch_idx], y_tr_t[batch_idx]

                logits, contributions = model(Xb)
                loss = penalized_bce(
                    logits, yb, contributions, model,
                    self.output_regularization, self.l2_regularization,
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                train_loss += loss.item()
                n_batches  += 1

            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_logits, _ = model(X_val_t)
                val_loss = F.binary_cross_entropy_with_logits(
                    val_logits.view(-1), y_val_t.view(-1)
                ).item()

            if self.verbose and epoch % 10 == 0:
                avg_train = train_loss / max(n_batches, 1)
                print(
                    f"[NAM] epoch {epoch:>4}/{self.num_epochs} | "
                    f"train_loss={avg_train:.4f} | val_loss={val_loss:.4f}"
                )

            if val_loss < best_val_loss - 1e-5:
                best_val_loss  = val_loss
                best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.early_stopping_patience:
                    if self.verbose:
                        print(
                            f"[NAM] Early stopping at epoch {epoch} "
                            f"(best val_loss={best_val_loss:.4f})"
                        )
                    break

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        return model

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        y,
        feature_names = None,
    ):
        device = torch.device(
            self.device
            if (self.device == "cpu" or torch.cuda.is_available())
            else "cpu"
        )

        if feature_names is None:
            feature_names = list(X.columns)
        self.feature_names_ = list(feature_names)

        X_scaled = self._fit_preprocessor(np.asarray(X, dtype=np.float32))
        y_arr    = np.asarray(y, dtype=np.float32).ravel()

        print(
            f"[NAM] Training | features={X_scaled.shape[1]} | "
            f"num_basis_functions={self.num_basis_functions} | "
            f"num_epochs={self.num_epochs} | batch_size={self.batch_size} | "
            f"lr={self.lr} | device={device}"
        )

        X_tr, y_tr, X_val, y_val = self._make_val_split(X_scaled, y_arr)

        if self.verbose:
            print(
                f"[NAM] Val split | "
                f"train: {int(y_tr.sum())}/{len(y_tr)} positives | "
                f"val: {int(y_val.sum())}/{len(y_val)} positives"
            )

        self.model_   = self._train(X_tr, y_tr, X_val, y_val, device)
        self.device_  = device
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        X_t = torch.as_tensor(
            self._transform(np.asarray(X, dtype=np.float32)),
            device=self.device_,
        )
        self.model_.eval()
        with torch.no_grad():
            logits, _ = self.model_(X_t)
            proba_pos = torch.sigmoid(logits.view(-1)).cpu().numpy()
        proba_pos = np.clip(proba_pos, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - proba_pos, proba_pos])

    # predict() / predict_log_proba() are inherited from BaseWrapperClassifier.

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_selected_features(self) -> list[str]:
        """Return names of features passed to fit() (after zero-variance filtering).

        Overrides BaseWrapperClassifier's default since NAM drops
        zero-variance columns internally (see _fit_preprocessor).
        """
        check_is_fitted(self, "model_")
        return [
            name for name, keep in zip(self.feature_names_, self.valid_cols_) if keep
        ]

    def get_feature_contributions(self, X) -> np.ndarray:
        """Per-feature additive contributions for each sample.

        Returns an array of shape (n_samples, n_active_features).
        Values are the raw outputs of each feature sub-network before summing.
        """
        check_is_fitted(self, "model_")
        X_t = torch.as_tensor(
            self._transform(np.asarray(X, dtype=np.float32)),
            device=self.device_,
        )
        self.model_.eval()
        with torch.no_grad():
            _, contributions = self.model_(X_t)
        return contributions.cpu().numpy()
