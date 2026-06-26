"""Private prediction run for the residual 72-hour model.

Train on all public data, 2011-2023, then predict private 2024-2025.
By default this uses the selected epoch and alpha from public validation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train_first_try_72h as base
from train_residual_72h import (
    OUT_DIR as VALIDATION_OUT_DIR,
    PERSISTENCE_LAG,
    best_worst,
    fit_residual_preprocessor,
    keep_finite_persistence,
    persistence_raw,
    predict_final,
    transform_residual_y,
    write_prediction_csv,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "residual_private_72h"


def load_validation_selection() -> tuple[int | None, float | None]:
    metrics_path = VALIDATION_OUT_DIR / "metrics.json"
    if not metrics_path.exists():
        return None, None

    metrics = json.loads(metrics_path.read_text())
    alpha = metrics.get("selected_alpha_by_validation_cc")

    epochs = []
    for seed_metric in metrics.get("seed_best_final_cc_metrics", {}).values():
        epoch = seed_metric.get("epoch")
        if epoch is not None:
            epochs.append(int(epoch))
    selected_epoch = int(round(float(np.median(epochs)))) if epochs else None
    return selected_epoch, alpha


def train_private_model(
    train: base.Samples,
    epochs: int,
    hidden: int,
    dropout: float,
    lr: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[base.MLP, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    prep = fit_residual_preprocessor(train)
    x_train = base.transform_x(train, prep)
    y_train = transform_residual_y(train, prep)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    model = base.MLP(x_train.shape[1], hidden, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

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
        print(f"seed={seed} epoch={epoch} train_loss={np.mean(losses):.5f}")

    return model, prep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs. Default: selected validation epoch, or 50 if unavailable.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Residual blending alpha. Default: selected validation alpha, or 1.0 if unavailable.",
    )
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()

    selected_epoch, selected_alpha = load_validation_selection()
    epochs = args.epochs if args.epochs is not None else max(selected_epoch if selected_epoch else 50, 50)
    alpha = args.alpha if args.alpha is not None else (float(selected_alpha) if selected_alpha is not None else 1.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamps, data = base.load_csv(base.FULL_CSV)
    public_all = keep_finite_persistence(base.build_samples(timestamps, data, 2011, 2023, False))
    private = keep_finite_persistence(base.build_samples(timestamps, data, 2024, 2025, False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print("samples", {"public_all_2011_2023": len(public_all.y), "private_2024_2025": len(private.y)})
    print(f"persistence_lag_hours={PERSISTENCE_LAG}")
    print(f"selected_epochs={epochs}")
    print(f"selected_alpha={alpha}")

    preds = []
    states = []
    preps = []
    for seed in args.seeds:
        model, prep = train_private_model(
            public_all,
            epochs,
            args.hidden,
            args.dropout,
            args.lr,
            args.batch_size,
            seed,
            device,
        )
        pred = predict_final(model, private, prep, device, alpha=alpha)
        preds.append(pred)
        states.append(model.state_dict())
        preps.append(prep.__dict__)

    private_pred = np.mean(np.stack(preds, axis=0), axis=0)
    private_csv = OUT_DIR / "private_predictions.csv"
    write_prediction_csv(private_csv, private, private_pred)

    result = {
        "task": "private prediction for residual 72h model",
        "target": "Speed(t+72h) - Speed(t+72h-648h)",
        "final_prediction": "Speed(t+72h-648h) + predicted_residual",
        "persistence_lag_hours_from_origin": PERSISTENCE_LAG,
        "epochs": epochs,
        "alpha": alpha,
        "seeds": args.seeds,
        "sample_counts": {
            "public_all_2011_2023": int(len(public_all.y)),
            "private_2024_2025": int(len(private.y)),
        },
        "metrics": {
            "private_residual_ensemble": base.metrics(
                private.y, private_pred, persistence_raw(private)
            ),
            "private_target_aligned_27day_persistence": base.metrics(
                private.y, persistence_raw(private), persistence_raw(private)
            ),
        },
        "examples": best_worst(private_csv),
        "validation_output_reference": str(VALIDATION_OUT_DIR),
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
        },
        OUT_DIR / "model_ensemble.pt",
    )

    print("\nPrivate metrics")
    for key, value in result["metrics"]["private_residual_ensemble"].items():
        print(f"  {key}: {value}")
    print("\nBest example", result["examples"]["best"])
    print("Worst example", result["examples"]["worst"])
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
