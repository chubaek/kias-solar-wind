"""First-pass 72-hour solar-wind speed model.

This script intentionally depends only on the Python standard library, NumPy,
and PyTorch so it can run in the current project environment without pandas or
scikit-learn.

Task:
    predict Speed(t + 72h) from observations available at forecast origin t.

Split:
    train:       public data, target timestamp in 2011-2019
    validation:  public data, target timestamp in 2020-2021
    public test: public data, target timestamp in 2022-2023
    private:     target timestamp in 2024-2025, evaluated once after training
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
FULL_CSV = HERE / "solar_wind_data.csv"
OUT_DIR = HERE / "outputs" / "first_try_72h"

DATETIME = "datetime"
TARGET = "Speed (km/s)"
FEATURE_COLUMNS = [
    "Speed (km/s)",
    "Density (1/cm^3)",
    "Temperature (K)",
    "B (nT)",
    "Sunspot Number",
    "Coronal Hole Area",
]

HORIZON_HOURS = 72

# Generic recent-history lags plus the two physics-relevant lags for h=72:
# recurrence: Speed(t + 72 - 648) = Speed(t - 576)
# CH delay:   CH(t + 72 - 96)     = CH(t - 24)
LAGS_HOURS = sorted(
    {
        0,
        1,
        2,
        3,
        6,
        12,
        24,
        48,
        72,
        96,
        168,
        336,
        576,
        648,
    }
)


def parse_float(value: str) -> float:
    if value in {"", "NaN", "nan", "NA", "<NA>"}:
        return math.nan
    return float(value)


def parse_dt(value: str) -> np.datetime64:
    # CSV timestamps are naive UTC strings such as "2011-01-01 00:30:00".
    return np.datetime64(datetime.fromisoformat(value), "h")


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[np.datetime64] = []
    rows: list[list[float]] = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(parse_dt(row[DATETIME]))
            rows.append([parse_float(row[col]) for col in FEATURE_COLUMNS])

    return np.asarray(timestamps), np.asarray(rows, dtype=np.float32)


def year_mask(timestamps: np.ndarray, start_year: int, end_year: int) -> np.ndarray:
    start = np.datetime64(f"{start_year:04d}-01-01T00", "h")
    end = np.datetime64(f"{end_year + 1:04d}-01-01T00", "h")
    return (timestamps >= start) & (timestamps < end)


@dataclass
class Samples:
    x: np.ndarray
    y: np.ndarray
    origin_times: np.ndarray
    target_times: np.ndarray


def build_samples(
    timestamps: np.ndarray,
    data: np.ndarray,
    target_start_year: int,
    target_end_year: int,
    daily_origins: bool,
) -> Samples:
    """Build supervised examples, splitting by target timestamp.

    The row index is hourly and continuous in this dataset, so lag k hours maps
    to index i-k. We still verify target timestamps by year to avoid leakage
    across chronological split boundaries.
    """

    max_lag = max(LAGS_HOURS)
    target_mask = year_mask(timestamps, target_start_year, target_end_year)

    features: list[list[float]] = []
    targets: list[float] = []
    origins: list[np.datetime64] = []
    target_times: list[np.datetime64] = []

    for origin_idx in range(max_lag, len(timestamps) - HORIZON_HOURS):
        target_idx = origin_idx + HORIZON_HOURS
        if not target_mask[target_idx]:
            continue

        # Optional operational setup from plan.md: one forecast per day at 23:30.
        if daily_origins:
            origin_text = np.datetime_as_string(timestamps[origin_idx], unit="m")
            if not origin_text.endswith("23:00"):
                # datetime64[h] loses the original :30 minute, so 23:30 becomes 23.
                continue

        target_speed = data[target_idx, FEATURE_COLUMNS.index(TARGET)]
        if math.isnan(float(target_speed)):
            continue

        row_features: list[float] = []
        for lag in LAGS_HOURS:
            row_features.extend(data[origin_idx - lag].tolist())

        # Calendar features known at forecast origin.
        origin_py = datetime.fromisoformat(
            np.datetime_as_string(timestamps[origin_idx], unit="h")
        )
        day_angle = 2.0 * math.pi * (origin_py.timetuple().tm_yday - 1) / 365.25
        hour_angle = 2.0 * math.pi * origin_py.hour / 24.0
        row_features.extend(
            [
                math.sin(day_angle),
                math.cos(day_angle),
                math.sin(hour_angle),
                math.cos(hour_angle),
            ]
        )

        features.append(row_features)
        targets.append(float(target_speed))
        origins.append(timestamps[origin_idx])
        target_times.append(timestamps[target_idx])

    return Samples(
        x=np.asarray(features, dtype=np.float32),
        y=np.asarray(targets, dtype=np.float32),
        origin_times=np.asarray(origins),
        target_times=np.asarray(target_times),
    )


@dataclass
class Preprocessor:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float


def fit_preprocessor(train: Samples) -> Preprocessor:
    feature_mean = np.nanmean(train.x, axis=0)
    feature_mean = np.where(np.isfinite(feature_mean), feature_mean, 0.0)
    x_filled = np.where(np.isnan(train.x), feature_mean, train.x)
    feature_std = x_filled.std(axis=0)
    feature_std = np.where(feature_std > 1e-6, feature_std, 1.0)

    target_mean = float(train.y.mean())
    target_std = float(train.y.std())
    if target_std <= 1e-6:
        target_std = 1.0

    return Preprocessor(feature_mean, feature_std, target_mean, target_std)


def transform_x(samples: Samples, prep: Preprocessor) -> np.ndarray:
    x = np.where(np.isnan(samples.x), prep.feature_mean, samples.x)
    return ((x - prep.feature_mean) / prep.feature_std).astype(np.float32)


def transform_y(samples: Samples, prep: Preprocessor) -> np.ndarray:
    return ((samples.y - prep.target_mean) / prep.target_std).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict(model: nn.Module, x: np.ndarray, prep: Preprocessor, device: torch.device) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            batch = torch.from_numpy(x[start : start + 4096]).to(device)
            pred = model(batch).cpu().numpy()
            preds.append(pred)
    pred_scaled = np.concatenate(preds)
    return pred_scaled * prep.target_std + prep.target_mean


def metrics(y_true: np.ndarray, y_pred: np.ndarray, persistence: np.ndarray | None = None) -> dict[str, float]:
    err = y_pred - y_true
    out = {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
    }
    if np.std(y_pred) > 1e-8 and np.std(y_true) > 1e-8:
        out["cc"] = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        out["cc"] = float("nan")
    if persistence is not None:
        p_mae = float(np.mean(np.abs(persistence - y_true)))
        out["mae_skill_vs_27day"] = float(1.0 - out["mae"] / p_mae)
    return out


def persistence_predictions(samples: Samples) -> np.ndarray:
    # Lag 576 is exactly Speed(t + 72 - 648) for a 72h forecast.
    lag_pos = LAGS_HOURS.index(576) * len(FEATURE_COLUMNS)
    pred = samples.x[:, lag_pos]
    fallback = np.nanmean(samples.y)
    return np.where(np.isnan(pred), fallback, pred)


def write_predictions(path: Path, samples: Samples, y_pred: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["origin_datetime", "target_datetime", "predicted_speed", "observed_speed"])
        for origin, target, pred, obs in zip(
            samples.origin_times, samples.target_times, y_pred, samples.y
        ):
            writer.writerow(
                [
                    np.datetime_as_string(origin, unit="h").replace("T", " ") + ":30:00",
                    np.datetime_as_string(target, unit="h").replace("T", " ") + ":30:00",
                    f"{pred:.3f}",
                    f"{obs:.3f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--daily-origins",
        action="store_true",
        help="Use only daily 23:30 forecast origins, matching plan.md.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamps, data = load_csv(FULL_CSV)
    train = build_samples(timestamps, data, 2011, 2019, args.daily_origins)
    val = build_samples(timestamps, data, 2020, 2021, args.daily_origins)
    public_test = build_samples(timestamps, data, 2022, 2023, args.daily_origins)
    private = build_samples(timestamps, data, 2024, 2025, args.daily_origins)

    prep = fit_preprocessor(train)
    x_train, y_train = transform_x(train, prep), transform_y(train, prep)
    x_val, y_val = transform_x(val, prep), transform_y(val, prep)
    x_public, x_private = transform_x(public_test, prep), transform_x(private, prep)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(x_train.shape[1], args.hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_x_t = torch.from_numpy(x_val).to(device)
    val_y_t = torch.from_numpy(y_val).to(device)

    best_val = float("inf")
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x_t), val_y_t).item())

        train_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_pred = predict(model, x_val, prep, device)
    public_pred = predict(model, x_public, prep, device)
    private_pred = predict(model, x_private, prep, device)

    train_mean = float(train.y.mean())
    result = {
        "task": "predict Speed(t + 72h)",
        "horizon_hours": HORIZON_HOURS,
        "lags_hours": LAGS_HOURS,
        "daily_origins": args.daily_origins,
        "feature_columns": FEATURE_COLUMNS,
        "sample_counts": {
            "train_2011_2019": int(len(train.y)),
            "val_2020_2021": int(len(val.y)),
            "public_test_2022_2023": int(len(public_test.y)),
            "private_2024_2025": int(len(private.y)),
        },
        "model": {
            "type": "MLP",
            "hidden": args.hidden,
            "dropout": args.dropout,
            "epochs_run": len(history),
            "best_val_loss_scaled": best_val,
            "device": str(device),
        },
        "metrics": {
            "val_mlp": metrics(val.y, val_pred, persistence_predictions(val)),
            "public_test_mlp": metrics(
                public_test.y, public_pred, persistence_predictions(public_test)
            ),
            "private_mlp": metrics(private.y, private_pred, persistence_predictions(private)),
            "private_public_mean_baseline": metrics(
                private.y, np.full_like(private.y, train_mean), persistence_predictions(private)
            ),
            "private_27day_persistence": metrics(
                private.y,
                persistence_predictions(private),
                persistence_predictions(private),
            ),
        },
        "history": history,
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "preprocessor": {
                "feature_mean": prep.feature_mean,
                "feature_std": prep.feature_std,
                "target_mean": prep.target_mean,
                "target_std": prep.target_std,
            },
            "args": vars(args),
            "feature_columns": FEATURE_COLUMNS,
            "lags_hours": LAGS_HOURS,
        },
        OUT_DIR / "model.pt",
    )
    (OUT_DIR / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=True))
    write_predictions(OUT_DIR / "private_predictions.csv", private, private_pred)

    print("\nPrivate metrics")
    for name, value in result["metrics"]["private_mlp"].items():
        print(f"  {name}: {value}")
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
