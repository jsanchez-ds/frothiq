"""Sliding-window LSTM regressor for flotation quality prediction.

The literature consensus on continuous-process datasets is that LSTMs benefit
from being fed **raw normalized sensor sequences** (not the engineered tabular
features the gradient boosters consume) — that way the comparison between the
two model families is fair and the LSTM gets to learn its own temporal
aggregations.

Pipeline:
  1. Z-score each sensor based on **train** statistics only.
  2. Slide a window of length `W` over the time series; the target at every
     window is the lab measurement at the last cycle of the window.
  3. Train one model per target (or a single multi-output head).
  4. Log everything to MLflow with a model signature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

# ----- normalization ----------------------------------------------------------


@dataclass
class SensorScaler:
    """Per-column z-score using train statistics."""

    mean: np.ndarray
    std: np.ndarray
    columns: list[str]

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list[str]) -> SensorScaler:
        sub = df[columns].to_numpy(dtype=float)
        mean = sub.mean(axis=0)
        std = sub.std(axis=0)
        std = np.where(std < 1e-9, 1.0, std)
        return cls(mean=mean, std=std, columns=list(columns))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        sub = df[self.columns].to_numpy(dtype=float)
        return (sub - self.mean) / self.std


# ----- windowing --------------------------------------------------------------


def make_windows(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int,
    target_cols: list[str],
    scaler: SensorScaler | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding windows of length ``window`` over the (sorted) time series.

    Returns
    -------
    (X, y)
        X: (N, window, n_features) float32 — z-scored sensor sequences.
        y: (N, n_targets) float32 — lab measurement at the last cycle of each window.
    """
    if scaler is not None:
        scaled = scaler.transform(df).astype(np.float32)
    else:
        scaled = df[sensor_cols].to_numpy(dtype=np.float32)
    targets = df[target_cols].to_numpy(dtype=np.float32)

    n_rows = scaled.shape[0]
    if n_rows < window:
        raise ValueError(f"Need at least {window} rows for window={window}, got {n_rows}")

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for end in range(window, n_rows + 1, stride):
        X_list.append(scaled[end - window : end])
        y_list.append(targets[end - 1])
    return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


# ----- model ------------------------------------------------------------------


class LSTMRegressor(nn.Module):
    """Stacked-LSTM with a small MLP head emitting one prediction per target."""

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class _ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ----- training ---------------------------------------------------------------


@dataclass
class LSTMResult:
    model: LSTMRegressor
    scaler: SensorScaler
    val_metrics: dict[str, float]
    test_metrics: dict[str, float] | None = None
    best_epoch: int = -1


def _eval_pred(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {f"{prefix}_rmse": rmse, f"{prefix}_mae": mae, f"{prefix}_r2": r2}


def train_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    sensor_cols: list[str],
    target_cols: list[str],
    *,
    window: int = 180,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.25,
    lr: float = 1e-3,
    batch_size: int = 128,
    max_epochs: int = 30,
    patience: int = 6,
    test_df: pd.DataFrame | None = None,
    device: str | None = None,
    run_name: str = "lstm",
    seed: int = 0,
) -> LSTMResult:
    """End-to-end LSTM training for FrothIQ.

    Parameters
    ----------
    train_df, val_df : DataFrame
        Sorted by timestamp ascending. Sensor and target columns must exist.
    sensor_cols, target_cols : list of str
    window : int
        Length of sliding windows in rows (default 180 = 1 hour at 20-second sampling).
    test_df : DataFrame, optional
        If provided, also computes test metrics at the end.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    scaler = SensorScaler.fit(train_df, sensor_cols)
    X_tr, y_tr = make_windows(train_df, sensor_cols, window, target_cols, scaler=scaler)
    X_va, y_va = make_windows(val_df, sensor_cols, window, target_cols, scaler=scaler)

    train_loader = DataLoader(_ArrayDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_ArrayDataset(X_va, y_va), batch_size=batch_size, shuffle=False)

    model = LSTMRegressor(
        n_features=len(sensor_cols),
        n_targets=len(target_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val_rmse = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_family": "lstm",
                "n_features": len(sensor_cols),
                "n_targets": len(target_cols),
                "window": window,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "lr": lr,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "patience": patience,
                "device": device,
                "seed": seed,
            }
        )

        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optim.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optim.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_preds: list[np.ndarray] = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    xb = xb.to(device)
                    val_preds.append(model(xb).cpu().numpy())
            y_val_pred = np.concatenate(val_preds)
            val_metrics = _eval_pred(y_va, y_val_pred, "val")
            mlflow.log_metrics({"train_mse": train_loss, **val_metrics}, step=epoch)

            if val_metrics["val_rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["val_rmse"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Final evaluation on val.
        model.eval()
        with torch.no_grad():
            val_preds = []
            for xb, _ in val_loader:
                val_preds.append(model(xb.to(device)).cpu().numpy())
            y_val_pred = np.concatenate(val_preds)
        val_metrics_final = _eval_pred(y_va, y_val_pred, "val")
        mlflow.log_metrics({f"final_{k}": v for k, v in val_metrics_final.items()})
        mlflow.log_metric("best_epoch", best_epoch)

        # Optional test set.
        test_metrics = None
        if test_df is not None:
            X_test, y_test = make_windows(
                test_df, sensor_cols, window, target_cols, scaler=scaler
            )
            with torch.no_grad():
                y_test_pred = model(torch.from_numpy(X_test).to(device)).cpu().numpy()
            test_metrics = _eval_pred(y_test, y_test_pred, "test")
            mlflow.log_metrics(test_metrics)

        # Log model with signature.
        example_in = X_tr[:2]
        with torch.no_grad():
            example_out = model(torch.from_numpy(example_in).to(device)).cpu().numpy()
        signature = infer_signature(example_in, example_out)
        mlflow.pytorch.log_model(model, name="model", signature=signature)

    return LSTMResult(
        model=model,
        scaler=scaler,
        val_metrics=val_metrics_final,
        test_metrics=test_metrics,
        best_epoch=best_epoch,
    )
