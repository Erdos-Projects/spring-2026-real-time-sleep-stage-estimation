"""
sleep_model_utils.py
--------------------
Training, evaluation, and persistence helpers for sleep-stage sequence models.

These utilities are model-agnostic — they work with any PyTorch Module that:
  * accepts (batch, seq_len, features) → (batch, num_classes) logits
  * is compatible with SleepSequenceDataset from sleep_data_utils

Public API
----------
make_class_weights(y, num_classes)
run_epoch(model, loader, optimizer, criterion, device, train)
build_loaders(X_train, y_train, info_train, X_val, y_val, info_val, ...)
train_one_config(X_train, y_train, ..., model_class, model_kwargs, ...)
evaluate_saved_sequence_model(model, X_eval, y_eval, info_eval, ...)
plot_confusion_matrix(eval_result, stage_map, title)
save_result_bundle(result, save_dir, scaler, extra_metrics)
load_saved_experiment(load_dir, model_class_map, device)
"""

from __future__ import annotations

import copy
import json
import os
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Feature engineering pipeline
# ---------------------------------------------------------------------------

class SleepFeaturePipeline:
    """Serializable feature-engineering pipeline for sleep-stage models.

    Steps
    -----
    1. ``add_core_sleep_ratios``  – relative band-power + cross-band ratios per EEG channel.
    2. ``add_lag_features``       – temporal lag columns for selected signals.

    Usage
    -----
    >>> pipe = SleepFeaturePipeline(
    ...     ratio_cols=["eeg_c4", "eeg_f4"],
    ...     lag_features=["eeg_c4_bp_delta", "hr_mean"],
    ...     lags=(1, 2, 3),
    ...     original_cols=FEAT_COLS + FEAT_METADATA_COLS,
    ... )
    >>> X_tr, y_tr, info_tr = pipe.fit_transform(X_train, y_train, info_train)
    >>> X_va, y_va, info_va = pipe.transform(X_val,  y_val,  info_val)
    >>> pipe.save("models/feat_pipeline.joblib")
    >>> pipe2 = SleepFeaturePipeline.load("models/feat_pipeline.joblib")
    """

    def __init__(
        self,
        ratio_cols: list[str],
        lag_features: list[str],
        lags: tuple[int, ...] = (1, 2, 3),
        original_cols: list[str] | None = None,
        eps: float = 1e-6,
    ) -> None:
        self.ratio_cols = ratio_cols
        self.lag_features = lag_features
        self.lags = lags
        self.original_cols = original_cols
        self.eps = eps
        # set after fit_transform
        self.feature_names_: list[str] | None = None
        self._names_after_ratio: list[str] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_ratios(self, X: np.ndarray) -> tuple[np.ndarray, list[str]]:
        assert self.original_cols is not None, "original_cols must be set before transform"
        df = pd.DataFrame(X, columns=self.original_cols)
        eps = self.eps
        for ch in self.ratio_cols:
            delta = df[f"{ch}_bp_delta"]
            theta = df[f"{ch}_bp_theta"]
            alpha = df[f"{ch}_bp_alpha"]
            sigma = df[f"{ch}_bp_sigma"]
            beta  = df[f"{ch}_bp_beta"]
            total = delta + theta + alpha + sigma + beta
            df[f"{ch}_bp_delta_rel"]        = delta / (total + eps)
            df[f"{ch}_bp_theta_rel"]        = theta / (total + eps)
            df[f"{ch}_bp_alpha_rel"]        = alpha / (total + eps)
            df[f"{ch}_bp_sigma_rel"]        = sigma / (total + eps)
            df[f"{ch}_bp_beta_rel"]         = beta  / (total + eps)
            df[f"{ch}_bp_delta_over_theta"] = delta / (theta + eps)
            df[f"{ch}_bp_delta_over_alpha"] = delta / (alpha + eps)
            df[f"{ch}_bp_theta_over_alpha"] = theta / (alpha + eps)
            df[f"{ch}_bp_sigma_over_delta"] = sigma / (delta + eps)
        return df.to_numpy(dtype=np.float32), df.columns.tolist()

    def _add_lags(
        self,
        X: np.ndarray,
        feature_names: list[str],
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        df = pd.DataFrame(X, columns=feature_names)
        for col in self.lag_features:
            if col not in df.columns:
                raise ValueError(f"Lag feature '{col}' not found in feature names")
            for lag in self.lags:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
        keep_idx = np.arange(max(self.lags), len(df))
        df = df.iloc[keep_idx].reset_index(drop=True)
        return df.to_numpy(dtype=np.float32), df.columns.tolist(), keep_idx

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        info: "pd.DataFrame | None" = None,
    ) -> tuple:
        """Apply both steps and store resulting feature names for future transforms."""
        X_ratio, names_ratio = self._add_ratios(X)
        X_out, names_out, keep_idx = self._add_lags(X_ratio, names_ratio)
        self.feature_names_ = names_out
        self._names_after_ratio = names_ratio
        out: list = [X_out]
        if y is not None:
            out.append(y[keep_idx])
        if info is not None:
            out.append(info.iloc[keep_idx].reset_index(drop=True))
        return tuple(out) if len(out) > 1 else X_out

    def transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        info: "pd.DataFrame | None" = None,
    ) -> tuple:
        """Apply the pipeline using the configuration stored by ``fit_transform``."""
        if self._names_after_ratio is None:
            raise RuntimeError("Call fit_transform before transform.")
        X_ratio, _ = self._add_ratios(X)
        X_out, _, keep_idx = self._add_lags(X_ratio, self._names_after_ratio)
        out: list = [X_out]
        if y is not None:
            out.append(y[keep_idx])
        if info is not None:
            out.append(info.iloc[keep_idx].reset_index(drop=True))
        return tuple(out) if len(out) > 1 else X_out

    def save(self, path: str | os.PathLike) -> None:
        """Persist the fitted pipeline to disk with joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"Feature pipeline saved → {path}")

    @classmethod
    def load(cls, path: str | os.PathLike) -> "SleepFeaturePipeline":
        """Load a previously saved pipeline."""
        pipe = joblib.load(path)
        if not isinstance(pipe, cls):
            raise TypeError(f"Loaded object is {type(pipe)}, expected {cls.__name__}")
        print(f"Feature pipeline loaded ← {path}")
        return pipe

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SleepFeaturePipeline("
            f"ratio_cols={self.ratio_cols}, "
            f"lag_features={self.lag_features}, "
            f"lags={self.lags})"
        )


# ---------------------------------------------------------------------------
# Class-weight helper
# ---------------------------------------------------------------------------

def make_class_weights(y: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    """Return a balanced class-weight tensor for CrossEntropyLoss."""
    classes = np.arange(num_classes)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return torch.tensor(w, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    train: bool = True,
) -> tuple[float, float]:
    """Run one train or validation epoch.

    Returns
    -------
    avg_loss : float
    macro_f1 : float
    """
    model.train() if train else model.eval()

    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())

    all_preds_arr = np.concatenate(all_preds)
    all_targets_arr = np.concatenate(all_targets)

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_targets_arr, all_preds_arr, average="macro", zero_division=0)

    return avg_loss, macro_f1


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    info_train: pd.DataFrame,
    X_val: np.ndarray,
    y_val: np.ndarray,
    info_val: pd.DataFrame,
    seq_len: int = 30,
    batch_size: int = 64,
    subject_col: str = "subject_id",
    block_col: str = "block_id",
    *,
    sleep_sequence_dataset_class=None,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders.

    Parameters
    ----------
    sleep_sequence_dataset_class :
        Pass ``SleepSequenceDataset`` from ``sleep_data_utils``.
        Falls back to a late import if omitted (requires sleep_data_utils on sys.path).
    """
    if sleep_sequence_dataset_class is None:
        from sleep_data_utils import SleepSequenceDataset  # type: ignore
        sleep_sequence_dataset_class = SleepSequenceDataset

    train_ds = sleep_sequence_dataset_class(
        X_train, y_train, info_train,
        seq_len=seq_len,
        subject_col=subject_col,
        block_col=block_col,
    )
    val_ds = sleep_sequence_dataset_class(
        X_val, y_val, info_val,
        seq_len=seq_len,
        subject_col=subject_col,
        block_col=block_col,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train_one_config(
    X_train: np.ndarray,
    y_train: np.ndarray,
    info_train: pd.DataFrame,
    X_val: np.ndarray,
    y_val: np.ndarray,
    info_val: pd.DataFrame,
    *,
    model_class,
    model_kwargs: dict,
    seq_len: int = 30,
    batch_size: int = 64,
    num_classes: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 40,
    patience: int = 6,
    subject_col: str = "subject_id",
    block_col: str = "block_id",
    device: torch.device | None = None,
    sleep_sequence_dataset_class=None,
) -> dict:
    """Train a model for one hyperparameter configuration.

    Parameters
    ----------
    model_class : type
        A PyTorch Module class (e.g. ``BiLSTMSleepClassifier``).
    model_kwargs : dict
        Instantiation kwargs for ``model_class``.
    sleep_sequence_dataset_class :
        Passed through to :func:`build_loaders`.

    Returns
    -------
    dict with keys: model_class, model_kwargs, seq_len, batch_size, lr,
    weight_decay, best_epoch, best_val_f1, history (DataFrame),
    model_state_dict, model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_loaders(
        X_train, y_train, info_train,
        X_val, y_val, info_val,
        seq_len=seq_len,
        batch_size=batch_size,
        subject_col=subject_col,
        block_col=block_col,
        sleep_sequence_dataset_class=sleep_sequence_dataset_class,
    )

    model = model_class(**model_kwargs).to(device)

    class_weights = make_class_weights(y_train, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    best_val_f1 = -1.0
    best_epoch = -1
    best_state = None
    wait = 0
    history: list[dict] = []

    for epoch in range(max_epochs):
        train_loss, train_f1 = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True
        )
        val_loss, val_f1 = run_epoch(
            model, val_loader, optimizer, criterion, device, train=False
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_f1": val_f1,
                "lr_now": current_lr,
            }
        )

        print(
            f"Epoch {epoch+1:02d} | "
            f"train loss {train_loss:.4f} | train F1 {train_f1:.4f} | "
            f"val loss {val_loss:.4f} | val F1 {val_f1:.4f} | "
            f"lr {current_lr:.2e}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history)

    return {
        "model_class": model_class.__name__,
        "model_kwargs": model_kwargs,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "history": history_df,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "model": model,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_saved_sequence_model(
    model: nn.Module,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    info_eval: pd.DataFrame,
    *,
    seq_len: int,
    batch_size: int = 128,
    device: torch.device | None = None,
    target_names: tuple[str, ...] = ("W", "N1", "N2", "N3", "R"),
    sleep_sequence_dataset_class=None,
) -> dict:
    """Evaluate a trained model and print/return metrics.

    Returns
    -------
    dict with keys: y_true, y_pred, confusion_matrix,
    classification_report_df.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if sleep_sequence_dataset_class is None:
        from sleep_data_utils import SleepSequenceDataset  # type: ignore
        sleep_sequence_dataset_class = SleepSequenceDataset

    eval_ds = sleep_sequence_dataset_class(
        X_eval,
        y_eval,
        info_eval,
        seq_len=seq_len,
    )
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for x_batch, y_batch in eval_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=list(target_names),
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    cm = confusion_matrix(y_true, y_pred)

    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=list(target_names),
            digits=4,
            zero_division=0,
        )
    )

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "classification_report_df": report_df,
    }


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def mode_filter_1d(y: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply a mode (majority-vote) smoothing filter along a 1-D label sequence.

    Parameters
    ----------
    y : array-like of int
        Predicted label sequence.
    kernel_size : int, optional
        Window width (must be odd).  Default is 5.

    Returns
    -------
    np.ndarray
        Smoothed label sequence of the same length as *y*.
        Tie-breaking rule: keep the centre label when it appears in the
        majority set, otherwise take the first majority candidate.
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    pad = kernel_size // 2
    y = np.asarray(y)
    y_pad = np.pad(y, (pad, pad), mode="edge")
    out = np.empty_like(y)

    for i in range(len(y)):
        window = y_pad[i : i + kernel_size]
        cnt = Counter(window)
        max_count = max(cnt.values())
        candidates = [k for k, v in cnt.items() if v == max_count]
        center = y[i]
        out[i] = center if center in candidates else candidates[0]

    return out


def plot_confusion_matrix(
    eval_result: dict,
    stage_map: dict,
    title: str = "",
) -> None:
    """Plot column- and row-normalised confusion matrices side by side.

    Parameters
    ----------
    eval_result :
        Dict returned by :func:`evaluate_saved_sequence_model`
        (must contain key ``"confusion_matrix"`).
    stage_map :
        ``{stage_name: int_label}`` dict used to derive axis tick labels.
    title :
        Optional super-title for the figure.
    """
    cm = eval_result["confusion_matrix"]
    labels = [k for k, _ in sorted(stage_map.items(), key=lambda kv: kv[1])]

    cm_f = cm.astype(float)

    col_sum = cm_f.sum(axis=0, keepdims=True)
    col_sum[col_sum == 0] = 1.0
    cm_col_norm = cm_f / col_sum

    row_sum = cm_f.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    cm_row_norm = cm_f / row_sum

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    if title:
        fig.suptitle(title)

    for ax, mat, subplot_title, xlabel, ylabel in [
        (axes[0], cm_col_norm, "Column-normalised (diag: precision)", "Predicted label", "True label"),
        (axes[1], cm_row_norm, "Row-normalised (diag: recall)",       "Predicted label", "True label"),
    ]:
        im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(subplot_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=12)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.show()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_result_bundle(
    result: dict,
    save_dir: str | os.PathLike,
    scaler=None,
    extra_metrics: dict | None = None,
) -> None:
    """Save model weights, config, metrics, history, and optionally scaler."""
    save_dir = str(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    torch.save(result["model_state_dict"], os.path.join(save_dir, "model.pt"))

    config = {
        "model_class": result["model_class"],
        "model_kwargs": result["model_kwargs"],
        "seq_len": result["seq_len"],
        "batch_size": result["batch_size"],
        "lr": result["lr"],
        "weight_decay": result["weight_decay"],
    }
    with open(os.path.join(save_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=4)

    metrics: dict = {
        "best_epoch": result["best_epoch"],
        "best_val_f1": result["best_val_f1"],
    }
    if extra_metrics is not None:
        metrics.update(extra_metrics)
    with open(os.path.join(save_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=4)

    result["history"].to_csv(os.path.join(save_dir, "history.csv"), index=False)

    if scaler is not None:
        joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

    print(f"Saved to {save_dir}")


def load_saved_experiment(
    load_dir: str | os.PathLike,
    model_class_map: dict,
    device: torch.device | None = None,
) -> tuple:
    """Load a saved experiment from *load_dir*.

    Parameters
    ----------
    model_class_map : dict[str, type]
        Maps ``model_class`` name strings to their Python classes.

    Returns
    -------
    (model, scaler, config, metrics)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_dir = str(load_dir)

    with open(os.path.join(load_dir, "config.json")) as fh:
        config = json.load(fh)

    model_class = model_class_map[config["model_class"]]
    model = model_class(**config["model_kwargs"]).to(device)

    state_dict = torch.load(
        os.path.join(load_dir, "model.pt"), map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()

    with open(os.path.join(load_dir, "metrics.json")) as fh:
        metrics = json.load(fh)

    scaler_path = os.path.join(load_dir, "scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    return model, scaler, config, metrics
