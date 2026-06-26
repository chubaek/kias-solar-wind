"""Generate full private-period final prediction files without dropping NaN-feature rows."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone

import freeze_final_ensemble_72h as freeze
import train_first_try_72h as direct
import train_tabular_models_72h as tab


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
DIAG_DIR = OUT_DIR / "final_72h_full_correction"

ENSEMBLE_WEIGHTS = {
    "direct_mlp": 0.70,
    "extratrees": 0.30,
    "persistence_27day": 0.00,
}
DIRECT_SEEDS = [11, 42, 77, 101, 123]
DIRECT_EPOCH = 1


def build_prediction_samples(
    timestamps: np.ndarray,
    data: np.ndarray,
    target_start_year: int,
    target_end_year: int,
) -> direct.Samples:
    max_lag = max(direct.LAGS_HOURS)
    target_mask = direct.year_mask(timestamps, target_start_year, target_end_year)
    features: list[list[float]] = []
    targets: list[float] = []
    origins: list[np.datetime64] = []
    target_times: list[np.datetime64] = []

    for target_idx in np.flatnonzero(target_mask):
        origin_idx = int(target_idx) - direct.HORIZON_HOURS
        if origin_idx < max_lag:
            continue

        row_features: list[float] = []
        for lag in direct.LAGS_HOURS:
            row_features.extend(data[origin_idx - lag].tolist())

        origin_py = pd.Timestamp(str(timestamps[origin_idx])).to_pydatetime()
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
        targets.append(float(data[target_idx, direct.FEATURE_COLUMNS.index(direct.TARGET)]))
        origins.append(timestamps[origin_idx])
        target_times.append(timestamps[target_idx])

    return direct.Samples(
        x=np.asarray(features, dtype=np.float32),
        y=np.asarray(targets, dtype=np.float32),
        origin_times=np.asarray(origins),
        target_times=np.asarray(target_times),
    )


def train_direct_full_private(
    train: direct.Samples,
    pred_samples: direct.Samples,
    device: torch.device,
) -> np.ndarray:
    preds: list[np.ndarray] = []
    for seed in DIRECT_SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        prep = direct.fit_preprocessor(train)
        x_train = direct.transform_x(train, prep)
        y_train = direct.transform_y(train, prep)
        x_pred = direct.transform_x(pred_samples, prep)

        model = direct.MLP(x_train.shape[1], hidden=128, dropout=0.1).to(device)
        loader = direct.make_loader(x_train, y_train, batch_size=512, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = torch.nn.SmoothL1Loss()

        for _ in range(DIRECT_EPOCH):
            model.train()
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

        preds.append(direct.predict(model, x_pred, prep, device))
        print(f"direct_full seed={seed} predicted_rows={len(preds[-1])}")
    return np.mean(preds, axis=0)


def datetime_strings_from_samples(samples: direct.Samples) -> list[str]:
    return [
        np.datetime_as_string(t, unit="h").replace("T", " ") + ":30:00"
        for t in samples.target_times
    ]


def evaluate_submission(path: Path, private: pd.DataFrame) -> dict[str, float]:
    pred = pd.read_csv(path, parse_dates=["datetime"])
    merged = private[["datetime", "Speed (km/s)"]].merge(pred, on="datetime", how="left")
    finite = merged["Speed (km/s)"].notna()
    scored = merged[finite & merged["predicted_speed"].notna()]
    y = scored["Speed (km/s)"].to_numpy(dtype=float)
    p = scored["predicted_speed"].to_numpy(dtype=float)
    err = p - y
    return {
        "prediction_rows": int(len(pred)),
        "finite_target_rows": int(finite.sum()),
        "scored_rows": int(len(scored)),
        "missing_finite_target_predictions": int(finite.sum() - len(scored)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "cc": float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 1e-8 and np.std(p) > 1e-8 else float("nan"),
    }


def main() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    private = pd.read_csv("solar_wind-private.csv", parse_dates=["datetime"])
    old_ensemble = pd.read_csv(OUT_DIR / "final_private_72h_ensemble.csv", parse_dates=["datetime"])
    finite_private = private[private["Speed (km/s)"].notna()]
    missing_old = finite_private.loc[~finite_private["datetime"].isin(old_ensemble["datetime"])]
    missing_old.to_csv(DIAG_DIR / "missing_from_old_ensemble.csv", index=False)

    config = json.loads((tab.OUT_DIR / "selected_config.json").read_text())
    selected_cfg = freeze.diag.find_selected_estimator(config)
    features = config["features"]
    target_type = config["selected"]["target_type"]
    weighted = config["selected"]["sample_weighting"] == "yes"

    table_all = tab.build_feature_table(
        tab.FULL_CSV,
        min_target_year=2011,
        max_target_year=2025,
        require_finite_target=False,
        require_finite_persistence=False,
    )
    train_idx = np.flatnonzero(
        (
            (table_all["target_year"] >= 2011)
            & (table_all["target_year"] <= 2023)
            & table_all["target_speed"].notna()
        ).to_numpy()
    )
    private_idx = np.flatnonzero(((table_all["target_year"] >= 2024) & (table_all["target_year"] <= 2025)).to_numpy())

    model = clone(selected_cfg["estimator"])
    X_train = table_all.iloc[train_idx][features]
    y_train = tab.adjusted_target(table_all, train_idx, target_type)
    sample_weight = (
        tab.target_weights(table_all.iloc[train_idx]["target_speed"].to_numpy(dtype=np.float32))
        if weighted
        else None
    )
    if sample_weight is None:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, model__sample_weight=sample_weight)
    extra_pred = tab.restore_target(table_all, private_idx, model.predict(table_all.iloc[private_idx][features]), target_type)

    timestamps, data = direct.load_csv(direct.FULL_CSV)
    direct_train = direct.build_samples(timestamps, data, 2011, 2023, daily_origins=False)
    direct_private = build_prediction_samples(timestamps, data, 2024, 2025)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    direct_pred = train_direct_full_private(direct_train, direct_private, device)

    direct_frame = pd.DataFrame({"datetime": pd.to_datetime(datetime_strings_from_samples(direct_private)), "direct_mlp": direct_pred})
    tab_private = table_all.iloc[private_idx][["target_datetime", "persistence_27day_target_aligned"]].copy()
    tab_private["datetime"] = pd.to_datetime(tab_private["target_datetime"])
    tab_private["extratrees"] = extra_pred
    full = private[["datetime"]].merge(tab_private[["datetime", "extratrees", "persistence_27day_target_aligned"]], on="datetime", how="left")
    full = full.merge(direct_frame, on="datetime", how="left")

    public_train_mean = float(pd.read_csv("solar_wind-public.csv")["Speed (km/s)"].dropna().mean())
    component_available = {
        "direct_mlp": full["direct_mlp"].notna(),
        "extratrees": full["extratrees"].notna(),
        "persistence_27day": full["persistence_27day_target_aligned"].notna(),
    }

    weighted = (
        ENSEMBLE_WEIGHTS["direct_mlp"] * full["direct_mlp"]
        + ENSEMBLE_WEIGHTS["extratrees"] * full["extratrees"]
        + ENSEMBLE_WEIGHTS["persistence_27day"] * full["persistence_27day_target_aligned"]
    )
    fallback = full["direct_mlp"].copy()
    fallback = fallback.fillna(full["extratrees"])
    fallback = fallback.fillna(full["persistence_27day_target_aligned"])
    fallback = fallback.fillna(public_train_mean)
    full["ensemble_pred"] = weighted.where(weighted.notna(), fallback)
    full["direct_full_pred"] = full["direct_mlp"].fillna(full["extratrees"]).fillna(full["persistence_27day_target_aligned"]).fillna(public_train_mean)

    freeze.write_prediction(OUT_DIR / "final_private_72h_ensemble_full.csv", full["datetime"], full["ensemble_pred"].to_numpy(dtype=float))
    freeze.write_prediction(OUT_DIR / "final_private_72h_direct_mlp_full.csv", full["datetime"], full["direct_full_pred"].to_numpy(dtype=float))

    corrected = pd.read_csv(OUT_DIR / "final_private_72h_ensemble_full.csv", parse_dates=["datetime"])
    missing_corrected = finite_private.loc[~finite_private["datetime"].isin(corrected["datetime"])]
    diagnostic = pd.DataFrame(
        [
            {
                "private_total_rows": len(private),
                "private_finite_target_rows": len(finite_private),
                "old_ensemble_prediction_rows": len(old_ensemble),
                "corrected_ensemble_prediction_rows": len(corrected),
                "missing_finite_target_rows_after_correction": len(missing_corrected),
                "direct_component_rows": int(component_available["direct_mlp"].sum()),
                "extratrees_component_rows": int(component_available["extratrees"].sum()),
                "persistence_component_rows": int(component_available["persistence_27day"].sum()),
            }
        ]
    )
    diagnostic.to_csv(DIAG_DIR / "full_private_row_diagnostic.csv", index=False)

    metrics = {
        "old_ensemble": evaluate_submission(OUT_DIR / "final_private_72h_ensemble.csv", private),
        "corrected_ensemble_full": evaluate_submission(OUT_DIR / "final_private_72h_ensemble_full.csv", private),
        "corrected_direct_mlp_full": evaluate_submission(OUT_DIR / "final_private_72h_direct_mlp_full.csv", private),
    }
    (DIAG_DIR / "evaluation_comparison.json").write_text(json.dumps(metrics, indent=2, allow_nan=True))

    print("\nRow diagnostic")
    print(diagnostic.to_string(index=False))
    print("\nEvaluation comparison")
    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
