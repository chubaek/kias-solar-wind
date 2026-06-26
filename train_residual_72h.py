"""Residual-learning MLP for 72-hour solar-wind speed forecasting.

The model predicts the correction to target-aligned 27-day persistence:

    residual = Speed(t + 72h) - Speed(t + 72h - 648h)
    final    = Speed(t + 72h - 648h) + alpha * predicted_residual

For a 72-hour forecast, Speed(t + 72h - 648h) is Speed(t - 576h).
Samples without this persistence value are ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train_first_try_72h as base


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "residual_72h"
PERSISTENCE_LAG = 648 - base.HORIZON_HOURS
ALPHAS = [round(i / 10.0, 1) for i in range(11)]


def persistence_raw(samples: base.Samples) -> np.ndarray:
    lag_pos = base.LAGS_HOURS.index(PERSISTENCE_LAG) * len(base.FEATURE_COLUMNS)
    return samples.x[:, lag_pos].astype(np.float32)


def keep_finite_persistence(samples: base.Samples) -> base.Samples:
    p = persistence_raw(samples)
    mask = np.isfinite(p)
    return base.Samples(
        x=samples.x[mask],
        y=samples.y[mask],
        origin_times=samples.origin_times[mask],
        target_times=samples.target_times[mask],
    )


class ResidualPreprocessor:
    def __init__(self, feature_mean, feature_std, residual_mean, residual_std):
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.target_mean = residual_mean
        self.target_std = residual_std


def fit_residual_preprocessor(samples: base.Samples) -> ResidualPreprocessor:
    feature_mean = np.nanmean(samples.x, axis=0)
    feature_mean = np.where(np.isfinite(feature_mean), feature_mean, 0.0)
    x_filled = np.where(np.isnan(samples.x), feature_mean, samples.x)
    feature_std = x_filled.std(axis=0)
    feature_std = np.where(feature_std > 1e-6, feature_std, 1.0)

    residual = samples.y - persistence_raw(samples)
    residual_mean = float(residual.mean())
    residual_std = float(residual.std())
    if residual_std <= 1e-6:
        residual_std = 1.0
    return ResidualPreprocessor(feature_mean, feature_std, residual_mean, residual_std)


def transform_residual_y(samples: base.Samples, prep: ResidualPreprocessor) -> np.ndarray:
    residual = samples.y - persistence_raw(samples)
    return ((residual - prep.target_mean) / prep.target_std).astype(np.float32)


def train_one(
    train: base.Samples,
    val: base.Samples,
    epochs: int,
    hidden: int,
    dropout: float,
    lr: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[
    base.MLP,
    ResidualPreprocessor,
    dict[str, float],
    dict[str, float],
    list[dict[str, float]],
]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    prep = fit_residual_preprocessor(train)
    x_train = base.transform_x(train, prep)
    y_train = transform_residual_y(train, prep)
    x_val = base.transform_x(val, prep)
    y_val = transform_residual_y(val, prep)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    model = base.MLP(x_train.shape[1], hidden, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    val_x_t = torch.from_numpy(x_val).to(device)
    val_y_t = torch.from_numpy(y_val).to(device)
    best_loss = float("inf")
    best_loss_state = None
    best_cc = -float("inf")
    best_cc_state = None
    best_cc_metrics: dict[str, float] | None = None
    best_cc_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_scaled_residual = model(val_x_t).cpu().numpy()
            val_loss = float(loss_fn(torch.from_numpy(val_scaled_residual).to(device), val_y_t).item())

        val_residual = val_scaled_residual * prep.target_std + prep.target_mean
        val_final = persistence_raw(val) + val_residual
        val_metrics = base.metrics(val.y, val_final, persistence_raw(val))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_residual_loss": val_loss,
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_cc": val_metrics["cc"],
            "val_mae_skill_vs_27day": val_metrics["mae_skill_vs_27day"],
        }
        history.append(row)

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_loss_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if np.isfinite(val_metrics["cc"]) and val_metrics["cc"] > best_cc:
            best_cc = val_metrics["cc"]
            best_cc_epoch = epoch
            best_cc_metrics = dict(val_metrics)
            best_cc_metrics["epoch"] = epoch
            best_cc_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Do not stop too early: the order asks to report best final-prediction
        # epoch by validation CC, which may differ from residual-loss epoch.
        if stale >= 12 and epoch >= 15:
            break

    if best_cc_state is not None:
        model.load_state_dict(best_cc_state)
    elif best_loss_state is not None:
        model.load_state_dict(best_loss_state)

    val_pred = predict_final(model, val, prep, device, alpha=1.0)
    val_metrics = base.metrics(val.y, val_pred, persistence_raw(val))
    val_metrics["best_residual_loss"] = best_loss
    val_metrics["best_final_cc_epoch"] = best_cc_epoch
    if best_cc_metrics is not None:
        val_metrics["best_final_cc"] = best_cc_metrics["cc"]
    best_loss_info = {"best_residual_loss": best_loss}
    if best_cc_metrics is None:
        best_cc_metrics = val_metrics
    return model, prep, val_metrics, best_cc_metrics, history


def predict_residual(
    model: nn.Module, samples: base.Samples, prep: ResidualPreprocessor, device: torch.device
) -> np.ndarray:
    x = base.transform_x(samples, prep)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            batch = torch.from_numpy(x[start : start + 4096]).to(device)
            preds.append(model(batch).cpu().numpy())
    pred_scaled = np.concatenate(preds)
    return pred_scaled * prep.target_std + prep.target_mean


def predict_final(
    model: nn.Module,
    samples: base.Samples,
    prep: ResidualPreprocessor,
    device: torch.device,
    alpha: float = 1.0,
) -> np.ndarray:
    return persistence_raw(samples) + alpha * predict_residual(model, samples, prep, device)


def write_prediction_csv(path: Path, samples: base.Samples, pred: np.ndarray) -> None:
    p = persistence_raw(samples)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "origin_datetime",
                "target_datetime",
                "persistence_speed",
                "predicted_residual",
                "predicted_speed",
                "observed_speed",
            ]
        )
        for origin, target, baseline, yhat, obs in zip(
            samples.origin_times, samples.target_times, p, pred, samples.y
        ):
            writer.writerow(
                [
                    np.datetime_as_string(origin, unit="h").replace("T", " ") + ":30:00",
                    np.datetime_as_string(target, unit="h").replace("T", " ") + ":30:00",
                    f"{baseline:.3f}",
                    f"{(yhat - baseline):.3f}",
                    f"{yhat:.3f}",
                    f"{obs:.3f}",
                ]
            )


def best_worst(csv_path: Path) -> dict[str, dict[str, object]]:
    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            pred = float(row["predicted_speed"])
            obs = float(row["observed_speed"])
            row["error"] = pred - obs
            row["abs_error"] = abs(pred - obs)
            rows.append(row)
    return {
        "best": min(rows, key=lambda row: row["abs_error"]),
        "worst": max(rows, key=lambda row: row["abs_error"]),
    }


def tune_alpha(samples: base.Samples, predicted_residual: np.ndarray) -> tuple[float, dict[str, dict[str, float]]]:
    alpha_metrics = {}
    persistence = persistence_raw(samples)
    for alpha in ALPHAS:
        final = persistence + alpha * predicted_residual
        alpha_metrics[f"{alpha:.1f}"] = base.metrics(samples.y, final, persistence)
    best_alpha_text = max(
        alpha_metrics,
        key=lambda key: (
            -float("inf")
            if not np.isfinite(alpha_metrics[key]["cc"])
            else alpha_metrics[key]["cc"],
            -alpha_metrics[key]["mae"],
        ),
    )
    return float(best_alpha_text), alpha_metrics


def format_metric_row(name: str, metric: dict[str, float] | None) -> str:
    if metric is None:
        return f"{name:36s} {'NA':>9s} {'NA':>9s} {'NA':>8s} {'NA':>10s}"
    skill = metric.get("mae_skill_vs_27day", float("nan"))
    return (
        f"{name:36s} "
        f"{metric['mae']:9.2f} {metric['rmse']:9.2f} {metric['cc']:8.3f} {skill:10.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamps, data = base.load_csv(base.FULL_CSV)
    train = keep_finite_persistence(base.build_samples(timestamps, data, 2011, 2021, False))
    val = keep_finite_persistence(base.build_samples(timestamps, data, 2022, 2023, False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print("samples", {"train_2011_2021": len(train.y), "val_2022_2023": len(val.y)})
    print(f"persistence_lag_hours={PERSISTENCE_LAG}")

    val_residual_preds = []
    histories = {}
    seed_metrics = {}
    seed_best_cc_metrics = {}
    states = []
    preps = []
    for seed in args.seeds:
        model, prep, val_metrics, best_cc_metrics, history = train_one(
            train,
            val,
            args.epochs,
            args.hidden,
            args.dropout,
            args.lr,
            args.batch_size,
            seed,
            device,
        )
        residual_pred = predict_residual(model, val, prep, device)
        val_residual_preds.append(residual_pred)
        seed_metrics[str(seed)] = val_metrics
        seed_best_cc_metrics[str(seed)] = best_cc_metrics
        histories[str(seed)] = history
        states.append(model.state_dict())
        preps.append(prep.__dict__)
        print(
            f"seed={seed} val_MAE={val_metrics['mae']:.2f} "
            f"val_RMSE={val_metrics['rmse']:.2f} val_CC={val_metrics['cc']:.3f} "
            f"best_final_CC_epoch={best_cc_metrics.get('epoch', 'NA')}"
        )

    ensemble_residual = np.mean(np.stack(val_residual_preds, axis=0), axis=0)
    alpha, alpha_metrics = tune_alpha(val, ensemble_residual)
    ensemble_pred_alpha1 = persistence_raw(val) + ensemble_residual
    ensemble_pred_tuned = persistence_raw(val) + alpha * ensemble_residual
    val_csv = OUT_DIR / "validation_predictions.csv"
    write_prediction_csv(val_csv, val, ensemble_pred_tuned)

    residual_metrics = base.metrics(val.y, ensemble_pred_alpha1, persistence_raw(val))
    tuned_metrics = base.metrics(val.y, ensemble_pred_tuned, persistence_raw(val))
    persistence_metrics = base.metrics(val.y, persistence_raw(val), persistence_raw(val))

    previous_metrics = None
    previous_path = Path(__file__).resolve().parent / "outputs" / "public_ensemble_72h" / "metrics.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        previous_seed_metrics = list(previous.get("selection_metrics", {}).values())
        if previous_seed_metrics:
            previous_metrics = {
                "mae": float(np.mean([m["mae"] for m in previous_seed_metrics])),
                "rmse": float(np.mean([m["rmse"] for m in previous_seed_metrics])),
                "cc": float(np.mean([m["cc"] for m in previous_seed_metrics])),
                "source": "mean of previous direct-speed validation seeds",
            }

    result = {
        "task": "predict residual then add target-aligned 27-day persistence",
        "target": "Speed(t+72h) - Speed(t+72h-648h)",
        "final_prediction": "Speed(t+72h-648h) + predicted_residual",
        "persistence_lag_hours_from_origin": PERSISTENCE_LAG,
        "feature_columns": base.FEATURE_COLUMNS,
        "lags_hours": base.LAGS_HOURS,
        "sample_counts": {"train_2011_2021": int(len(train.y)), "val_2022_2023": int(len(val.y))},
        "alpha_grid": ALPHAS,
        "selected_alpha_by_validation_cc": alpha,
        "validation_metrics": {
            "target_aligned_27day_persistence": persistence_metrics,
            "residual_alpha_1": residual_metrics,
            "residual_tuned_alpha": tuned_metrics,
            "target_aligned_27day_persistence": persistence_metrics,
            "previous_direct_mlp": previous_metrics,
        },
        "alpha_metrics": alpha_metrics,
        "seed_metrics": seed_metrics,
        "seed_best_final_cc_metrics": seed_best_cc_metrics,
        "examples": best_worst(val_csv),
        "histories": histories,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=True))
    torch.save(
        {
            "model_state_dicts": states,
            "preprocessors": preps,
            "args": vars(args),
            "feature_columns": base.FEATURE_COLUMNS,
            "lags_hours": base.LAGS_HOURS,
            "persistence_lag": PERSISTENCE_LAG,
            "selected_alpha": alpha,
        },
        OUT_DIR / "model_ensemble.pt",
    )

    print("\nValidation comparison table")
    print(f"{'Model':36s} {'MAE':>9s} {'RMSE':>9s} {'CC':>8s} {'MAE Skill':>10s}")
    print(format_metric_row("27-day persistence only", persistence_metrics))
    print(format_metric_row("residual model alpha=1.0", residual_metrics))
    print(format_metric_row(f"residual tuned alpha={alpha:.1f}", tuned_metrics))
    print(format_metric_row("previous direct MLP", previous_metrics))
    print("\nAlpha search by validation CC")
    for alpha_text, metric in alpha_metrics.items():
        print(format_metric_row(f"alpha={alpha_text}", metric))
    print("\nBest example", result["examples"]["best"])
    print("Worst example", result["examples"]["worst"])
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
