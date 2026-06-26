"""
NAM (Neural Additive Model) wrapper for the ChemoTreeVsDL pipeline.

Architecture (Agarwal et al., 2021):
    One FeatureNN per selected feature, each a small MLP with an ExU first
    layer.  Predictions are the sum of all per-feature outputs + a scalar bias.
    This gives the same additive interpretability as a GAM but with learned
    nonlinear shape functions.

Pipeline: SimpleImputer → MinMax[0,1] → BatchedNAM

MinMax scaling is required because ExU's exp() can explode on raw z-scores.

The original NAM repo lives at ``ml_model_training/model_wrappers/nam/``.
The sequential ``FeatureNN`` loop from the repo (200 Python calls / batch) is
replaced here with a fully-batched 3-D tensor implementation:

    (batch, F) → ExU → (batch, F, U) → bmm → (batch, F, 1) → sum_F → (batch,)

where F = n_selected_features, U = num_basis_functions.  This eliminates the
Python-overhead bottleneck and reduces wall-clock time by ~50–100x on CPU
compared to the sequential formulation at 200 features / 1000 units.

Training loop (hand-written, no pytorch_lightning):
    • penalized BCE loss (output-L2 + weight-L2 regularisation)
    • feature dropout (randomly zero entire feature columns per batch)
    • AdamW + cosine-decay LR schedule
    • early stopping on validation BCE loss (val_fraction of train set)
"""

from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted


# ===================================================================
# Batched NAM — parallel formulation
# ===================================================================

class _BatchedExU(nn.Module):
    """ExU activation applied jointly to all F features.

    Input:  (batch, F)       — one value per feature per sample
    Output: (batch, F, U)    — U learned basis responses per feature

    ExU formula (Agarwal et al.):
        h_i(x_i) = ReLU-n( (x_i − b_i) · exp(w_i) )
    where b_i ∈ R and w_i ∈ R^U are the bias and weights for feature i.
    """

    def __init__(self, n_features: int, n_units: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_units = n_units
        self.weights = nn.Parameter(torch.empty(n_features, n_units))
        self.bias = nn.Parameter(torch.empty(n_features))
        self._init_params()

    def _init_params(self) -> None:
        with torch.no_grad():
            self.weights.normal_(mean=4.0, std=0.5).clamp_(3.0, 5.0)
            self.bias.normal_(mean=0.0, std=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = (x - self.bias).unsqueeze(-1)           # (B, F, 1)
        out = diff * torch.exp(self.weights).unsqueeze(0)  # (B, F, U)
        return F.relu(out).clamp(max=1.0)               # ReLU-1 capped at n=1


class _BatchedLinReLU(nn.Module):
    """Grouped linear + ReLU: (B, F, in) → (B, F, out)."""

    def __init__(self, n_features: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.weights = nn.Parameter(torch.empty(n_features, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(n_features, out_features))
        nn.init.xavier_uniform_(self.weights.view(n_features, in_features, out_features).contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.permute(1, 0, 2)                       # (F, B, in)
        out = torch.bmm(x_t, self.weights)              # (F, B, out)
        out = out.permute(1, 0, 2)                      # (B, F, out)
        return F.relu(out + self.bias.unsqueeze(0))


class _BatchedLinear(nn.Module):
    """Grouped linear (no activation): (B, F, in) → (B, F, out)."""

    def __init__(self, n_features: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_features, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(n_features, out_features))
        nn.init.xavier_uniform_(self.weights.view(n_features, in_features, out_features).contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.permute(1, 0, 2)                       # (F, B, in)
        out = torch.bmm(x_t, self.weights)              # (F, B, out)
        return (out + self.bias.unsqueeze(1)).permute(1, 0, 2)  # (B, F, out)


class _BatchedNAM(nn.Module):
    """Batched NAM: all F feature nets run in a single kernel sweep.

    forward(x) → (logit, fnn_out)
        logit:   (B,)     — raw log-odds
        fnn_out: (B, F)   — per-feature scalar contribution (before sum+bias)
    """

    def __init__(
        self,
        n_features: int,
        n_units: int,
        hidden_sizes: list,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.dropout = nn.Dropout(p=dropout)

        layers = [_BatchedExU(n_features, n_units)]
        prev = n_units
        for h in hidden_sizes:
            layers.append(_BatchedLinReLU(n_features, prev, h))
            prev = h
        layers.append(_BatchedLinear(n_features, prev, 1))
        self.layers = nn.ModuleList(layers)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        out = self.layers[0](x)          # ExU → (B, F, U)
        out = self.dropout(out)
        for layer in self.layers[1:]:
            out = layer(out)
            out = self.dropout(out)
        fnn_out = out.squeeze(-1)        # (B, F)
        logit = fnn_out.sum(dim=-1) + self.bias  # (B,)
        return logit, fnn_out


# ===================================================================
# Loss
# ===================================================================

def _penalized_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    fnn_out: torch.Tensor,
    model: _BatchedNAM,
    output_regularization: float,
    l2_regularization: float,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits.view(-1), targets.view(-1))
    if output_regularization > 0.0:
        loss = loss + output_regularization * torch.mean(fnn_out ** 2)
    if l2_regularization > 0.0:
        last_linear = model.layers[-1]
        l2 = (last_linear.weights ** 2).sum() + (last_linear.bias ** 2).sum()
        loss = loss + l2_regularization * l2 / model.n_features
    return loss


# ===================================================================
# sklearn wrapper
# ===================================================================

class NAMWrapperClassifier(ClassifierMixin, BaseEstimator):
    """Batched NAM wrapped with median-imputation + MinMax [0,1] scaling.

    Parameters
    ----------
    num_basis_functions : int, default 64
        ExU units per feature.
    hidden_sizes : list[int], default []
        Extra grouped linear-ReLU layers after ExU.
    dropout : float, default 0.1
    feature_dropout : float, default 0.0
        Fraction of feature columns randomly zeroed per training batch.
    output_regularization : float, default 0.1
    l2_regularization : float, default 0.05
    lr : float, default 0.01
    num_epochs : int, default 100
    batch_size : int, default 1024
    early_stopping_patience : int, default 20
    val_fraction : float, default 0.15
    device : str, default "cuda"
    random_state : int, default 42
    verbose : int, default 1
    """

    def __init__(
        self,
        num_basis_functions: int = 64,
        hidden_sizes=None,
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
        random_state: int = 42,
        verbose: int = 1,
    ):
        self.num_basis_functions = num_basis_functions
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.output_regularization = output_regularization
        self.l2_regularization = l2_regularization
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.val_fraction = val_fraction
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_fit(self, X, y, feature_names):
        if feature_names is None:
            feature_names = list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])]
        self.selected_features_ = list(feature_names)
        self.imputer_ = SimpleImputer(strategy="median")
        X_out = np.asarray(self.imputer_.fit_transform(X), dtype=np.float32)
        col_min = X_out.min(axis=0)
        col_max = X_out.max(axis=0)
        col_range = col_max - col_min

        self._nam_keep_mask_ = col_range > 1e-8
        if not self._nam_keep_mask_.all():
            n_drop = int((~self._nam_keep_mask_).sum())
            print(f"[NAM] dropping {n_drop} zero-variance columns post-selection")
            X_out = X_out[:, self._nam_keep_mask_]
            col_min = col_min[self._nam_keep_mask_]
            col_range = col_range[self._nam_keep_mask_]

        self._nam_col_min_ = col_min
        self._nam_col_range_ = col_range
        return np.clip((X_out - col_min) / col_range, 0.0, 1.0).astype(np.float32)

    def _preprocess_transform(self, X):
        check_is_fitted(self, "nam_")
        Xt = np.asarray(self.imputer_.transform(X), dtype=np.float32)
        if hasattr(self, "_nam_keep_mask_"):
            Xt = Xt[:, self._nam_keep_mask_]
        return np.clip(
            (Xt - self._nam_col_min_) / self._nam_col_range_, 0.0, 1.0
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _train(self, X_tr, y_tr, X_val, y_val, device):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        n_features = X_tr.shape[1]
        model = _BatchedNAM(
            n_features=n_features,
            n_units=self.num_basis_functions,
            hidden_sizes=list(self.hidden_sizes or []),
            dropout=self.dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs, eta_min=self.lr * 0.01
        )

        X_tr_t = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
        y_tr_t = torch.as_tensor(y_tr, dtype=torch.float32, device=device)
        X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
        y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=device)

        n_tr = X_tr_t.shape[0]
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.num_epochs + 1):
            model.train()
            perm = torch.randperm(n_tr, device=device)
            train_loss_acc = 0.0
            n_batches = 0

            for start in range(0, n_tr, self.batch_size):
                idx = perm[start: start + self.batch_size]
                Xb, yb = X_tr_t[idx], y_tr_t[idx]
                logits, fnn_out = model(Xb)

                if self.feature_dropout > 0.0:
                    mask = (
                        torch.rand(fnn_out.shape[1], device=device) > self.feature_dropout
                    ).float()
                    fnn_out = fnn_out * mask.unsqueeze(0)
                    logits = fnn_out.sum(dim=-1) + model.bias

                loss = _penalized_bce(
                    logits, yb, fnn_out, model,
                    self.output_regularization, self.l2_regularization,
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss_acc += loss.item()
                n_batches += 1

            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_logits, _ = model(X_val_t)
                val_loss = F.binary_cross_entropy_with_logits(
                    val_logits.view(-1), y_val_t.view(-1)
                ).item()

            if self.verbose and epoch % 10 == 0:
                print(
                    f"[NAM] epoch {epoch:>4}/{self.num_epochs} | "
                    f"train_loss={train_loss_acc / max(n_batches,1):.4f} | "
                    f"val_loss={val_loss:.4f}"
                )

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"[NAM] Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
                    break

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        return model

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X, y, feature_names: Iterable[str] | None = None):
        device = torch.device(
            self.device if (self.device == "cpu" or torch.cuda.is_available()) else "cpu"
        )

        X_scaled = self._preprocess_fit(X, y, feature_names)
        y_arr = np.asarray(y, dtype=np.float32).ravel()
        n_features = X_scaled.shape[1]

        print(
            f"[NAM] Training with {n_features} features | "
            f"num_basis_functions={self.num_basis_functions} | "
            f"num_epochs={self.num_epochs} | batch_size={self.batch_size} | "
            f"lr={self.lr} | device={device}"
        )

        n_total = len(y_arr)
        idx = np.arange(n_total)
        try:
            tr_idx, val_idx = train_test_split(
                idx, test_size=self.val_fraction,
                random_state=self.random_state, stratify=y_arr,
            )
        except ValueError:
            rng = np.random.RandomState(self.random_state)
            val_idx = rng.choice(n_total, size=max(1, int(n_total * self.val_fraction)), replace=False)
            tr_mask = np.ones(n_total, dtype=bool)
            tr_mask[val_idx] = False
            tr_idx = idx[tr_mask]

        if self.verbose:
            print(
                f"[NAM] val split | train positives={int(y_arr[tr_idx].sum())}/"
                f"{len(tr_idx)} | val positives={int(y_arr[val_idx].sum())}/{len(val_idx)}"
            )

        self.nam_ = self._train(X_scaled[tr_idx], y_arr[tr_idx], X_scaled[val_idx], y_arr[val_idx], device)
        self.device_ = device
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "nam_")
        Xt = torch.as_tensor(self._preprocess_transform(X), dtype=torch.float32, device=self.device_)
        self.nam_.eval()
        with torch.no_grad():
            logits, _ = self.nam_(Xt)
            proba_pos = torch.sigmoid(logits.view(-1)).cpu().numpy()
        proba_pos = np.clip(proba_pos, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - proba_pos, proba_pos])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_log_proba(self, X):
        return np.log(np.clip(self.predict_proba(X), 1e-10, 1.0))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_selected_features(self) -> list:
        check_is_fitted(self, "nam_")
        return list(self.selected_features_)

    def get_nam(self) -> _BatchedNAM:
        check_is_fitted(self, "nam_")
        return self.nam_

    def get_feature_contributions(self, X) -> np.ndarray:
        """Return per-feature contributions for each sample: shape (n_samples, n_features)."""
        check_is_fitted(self, "nam_")
        Xt = torch.as_tensor(self._preprocess_transform(X), dtype=torch.float32, device=self.device_)
        self.nam_.eval()
        with torch.no_grad():
            _, fnn_out = self.nam_(Xt)
        return fnn_out.cpu().numpy()
