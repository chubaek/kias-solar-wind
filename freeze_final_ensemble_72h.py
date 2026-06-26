"""Freeze the current public-selected 72-hour ensemble and sanity-check it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import inspect_public_and_private_diagnostic as diag
import train_first_try_72h as direct
import train_tabular_models_72h as tab


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
FREEZE_DIR = OUT_DIR / "final_72h_freeze"

ENSEMBLE_WEIGHTS = {
    "direct_mlp": 0.70,
    "extratrees": 0.30,
    "persistence_27day": 0.00,
}
DIRECT_SEEDS = [11, 42, 77, 101, 123]
DIRECT_EPOCH = 1


def format_dt(value: np.datetime64 | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def direct_frame(samples: direct.Samples, pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_datetime": pd.to_datetime(
                [np.datetime_as_string(x, unit="h").replace("T", " ") + ":30:00" for x in samples.target_times]
            ),
            "direct_pred": pred,
        }
    )


def align_direct_pred(table: pd.DataFrame, idx: np.ndarray, samples: direct.Samples, pred: np.ndarray) -> np.ndarray:
    frame = table.iloc[idx][["target_datetime"]].copy()
    frame["target_datetime"] = pd.to_datetime(frame["target_datetime"])
    merged = frame.merge(direct_frame(samples, pred), on="target_datetime", how="left")
    return merged["direct_pred"].to_numpy(dtype=np.float32)


def write_prediction(path: Path, datetimes: pd.Series, pred: np.ndarray) -> None:
    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(datetimes).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_speed": np.round(pred.astype(float), 6),
        }
    )
    out.to_csv(path, index=False)


def add_row(
    rows: list[dict[str, float | str | None]],
    model: str,
    fixed: dict[str, float],
    cv: dict[str, float] | None,
    private: dict[str, float],
) -> None:
    rows.append(
        {
            "model": model,
            "public_fixed_mae": fixed.get("mae"),
            "public_fixed_rmse": fixed.get("rmse"),
            "public_fixed_cc": fixed.get("cc"),
            "public_fixed_mae_skill": fixed.get("mae_skill_vs_27day"),
            "public_cv_mean_mae": None if cv is None else cv.get("mean_cv_mae"),
            "public_cv_mean_rmse": None if cv is None else cv.get("mean_cv_rmse"),
            "public_cv_mean_cc": None if cv is None else cv.get("mean_cv_cc"),
            "public_cv_mean_mae_skill": None if cv is None else cv.get("mean_cv_mae_skill"),
            "private_mae": private.get("mae"),
            "private_rmse": private.get("rmse"),
            "private_cc": private.get("cc"),
            "private_mae_skill": private.get("mae_skill_vs_27day"),
        }
    )


def cv_mean_for(cv: pd.DataFrame, model_id: str | None = None, model_name: str | None = None) -> dict[str, float] | None:
    if model_id is not None:
        rows = cv[cv["model_id"] == model_id]
    else:
        rows = cv[cv["model_name"] == model_name]
    if rows.empty:
        return None
    return {
        "mean_cv_mae": float(rows["mae"].mean()),
        "mean_cv_rmse": float(rows["rmse"].mean()),
        "mean_cv_cc": float(rows["cc"].mean()),
        "mean_cv_mae_skill": float(rows["mae_skill_vs_27day"].mean()),
    }


def make_sanity_rows(raw: pd.DataFrame, table_all: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260624)
    private_rows = table_all[(table_all["target_year"] >= 2024) & (table_all["target_year"] <= 2025)]
    sample = private_rows.sample(n=10, random_state=20260624).sort_values("origin_datetime")

    raw_by_time = raw.set_index("datetime")
    rows = []
    for _, row in sample.iterrows():
        origin = pd.Timestamp(row["origin_datetime"])
        target = pd.Timestamp(row["target_datetime"])
        persistence_time = origin - pd.Timedelta(hours=576)
        recurrence_center = origin - pd.Timedelta(hours=648)
        rows.append(
            {
                "origin_t": format_dt(origin),
                "target_t_plus_72h": format_dt(target),
                "target_minus_origin_hours": (target - origin) / pd.Timedelta(hours=1),
                "persistence_timestamp_t_minus_576h": format_dt(persistence_time),
                "recurrence_timestamp_t_minus_648h": format_dt(recurrence_center),
                "target_speed": row["target_speed"],
                "speed_at_origin_t": row["speed_current"],
                "speed_at_t_minus_576h_feature": row["persistence_27day_target_aligned"],
                "raw_speed_at_t_minus_576h": raw_by_time.loc[persistence_time, tab.TARGET],
                "speed_at_t_minus_648h_feature": row["speed_recurrence_source_surface_648h"],
                "raw_speed_at_t_minus_648h": raw_by_time.loc[recurrence_center, tab.TARGET],
                "roll_mean_24h_at_origin": row["Speed_km_s_roll_mean_24h"],
                "coronal_hole_area_current": row["coronal_hole_area_current"],
                "sunspot_current": row["sunspot_current"],
                "max_feature_timestamp": format_dt(origin),
                "leakage_check": "all listed timestamps <= origin except target",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)

    config = json.loads((tab.OUT_DIR / "selected_config.json").read_text())
    fixed = pd.read_csv(tab.OUT_DIR / "fixed_results.csv")
    cv = pd.read_csv(tab.OUT_DIR / "cv_results.csv")
    selected_id = config["selected"]["model_id"]
    selected_cfg = diag.find_selected_estimator(config)
    features = config["features"]
    target_type = config["selected"]["target_type"]
    weighted = config["selected"]["sample_weighting"] == "yes"

    raw = pd.read_csv(tab.FULL_CSV, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    table_public = tab.build_feature_table(tab.FULL_CSV)
    table_all = tab.build_feature_table_including_private(tab.FULL_CSV)

    sanity_rows = make_sanity_rows(raw, table_all)
    print("\nTimestamp alignment sanity sample")
    print(sanity_rows.to_string(index=False))
    sanity_rows.to_csv(FREEZE_DIR / "timestamp_alignment_sample.csv", index=False)

    checks = {
        "target_definition": "target is Speed(t + 72h)",
        "target_aligned_27day_persistence": "Speed(t + 72h - 648h) = Speed(t - 576h)",
        "feature_timestamp_rule": "all generated features use shift/rolling values at or before forecast origin t",
        "rolling_windows": "pandas rolling windows are trailing/causal, not centered",
        "tabular_scaler_imputer_rule": "scikit-learn pipelines are fit on train split only, then used to transform validation/private",
        "direct_mlp_preprocessor_rule": "direct MLP preprocessor is fit on train samples only",
        "private_selection_rule": "private diagnostics are not used for feature selection, model selection, hyperparameter tuning, or ensemble weight selection",
        "frozen_ensemble_weights": ENSEMBLE_WEIGHTS,
    }
    (FREEZE_DIR / "sanity_checks.json").write_text(json.dumps(checks, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\ndevice={device}")

    timestamps, data = direct.load_csv(direct.FULL_CSV)
    direct_train_fixed = direct.build_samples(timestamps, data, 2011, 2021, daily_origins=False)
    direct_val_fixed = direct.build_samples(timestamps, data, 2022, 2023, daily_origins=False)
    direct_train_public = direct.build_samples(timestamps, data, 2011, 2023, daily_origins=False)
    direct_private = direct.build_samples(timestamps, data, 2024, 2025, daily_origins=False)

    direct_fixed_pred, _, _ = diag.train_direct_ensemble(
        direct_train_fixed,
        direct_val_fixed,
        DIRECT_EPOCH,
        selected_epoch=DIRECT_EPOCH,
        seeds=DIRECT_SEEDS,
        hidden=128,
        dropout=0.1,
        lr=1e-3,
        batch_size=512,
        patience=7,
        device=device,
    )
    direct_private_pred, _, _ = diag.train_direct_ensemble(
        direct_train_public,
        direct_private,
        DIRECT_EPOCH,
        selected_epoch=DIRECT_EPOCH,
        seeds=DIRECT_SEEDS,
        hidden=128,
        dropout=0.1,
        lr=1e-3,
        batch_size=512,
        patience=7,
        device=device,
    )

    train_idx, val_idx = tab.split_rows(table_public, tab.fixed_split())
    public_idx = np.flatnonzero(((table_all["target_year"] >= 2011) & (table_all["target_year"] <= 2023)).to_numpy())
    private_idx = np.flatnonzero(((table_all["target_year"] >= 2024) & (table_all["target_year"] <= 2025)).to_numpy())

    extra_fixed_pred = diag.train_tabular_predict(selected_cfg, table_public, features, train_idx, val_idx, target_type, weighted)
    extra_private_pred = diag.train_tabular_predict(selected_cfg, table_all, features, public_idx, private_idx, target_type, weighted)

    y_fixed = table_public.iloc[val_idx]["target_speed"].to_numpy(dtype=np.float32)
    persistence_fixed = table_public.iloc[val_idx]["persistence_27day_target_aligned"].to_numpy(dtype=np.float32)
    aligned_direct_fixed = align_direct_pred(table_public, val_idx, direct_val_fixed, direct_fixed_pred)
    ensemble_fixed_pred = (
        ENSEMBLE_WEIGHTS["direct_mlp"] * aligned_direct_fixed
        + ENSEMBLE_WEIGHTS["extratrees"] * extra_fixed_pred
        + ENSEMBLE_WEIGHTS["persistence_27day"] * persistence_fixed
    )

    y_private = table_all.iloc[private_idx]["target_speed"].to_numpy(dtype=np.float32)
    persistence_private = table_all.iloc[private_idx]["persistence_27day_target_aligned"].to_numpy(dtype=np.float32)
    aligned_direct_private = align_direct_pred(table_all, private_idx, direct_private, direct_private_pred)
    ensemble_private_pred = (
        ENSEMBLE_WEIGHTS["direct_mlp"] * aligned_direct_private
        + ENSEMBLE_WEIGHTS["extratrees"] * extra_private_pred
        + ENSEMBLE_WEIGHTS["persistence_27day"] * persistence_private
    )

    direct_fixed_metrics = tab.metrics(y_fixed, aligned_direct_fixed, persistence_fixed)
    extra_fixed_metrics = tab.metrics(y_fixed, extra_fixed_pred, persistence_fixed)
    ensemble_fixed_metrics = tab.metrics(y_fixed, ensemble_fixed_pred, persistence_fixed)
    persistence_fixed_metrics = tab.metrics(y_fixed, persistence_fixed, persistence_fixed)

    direct_private_metrics = tab.metrics(y_private, aligned_direct_private, persistence_private)
    extra_private_metrics = tab.metrics(y_private, extra_private_pred, persistence_private)
    ensemble_private_metrics = tab.metrics(y_private, ensemble_private_pred, persistence_private)
    persistence_private_metrics = tab.metrics(y_private, persistence_private, persistence_private)

    selected_fixed_row = fixed[fixed["model_id"] == selected_id].iloc[0].to_dict()
    selected_cv = cv_mean_for(cv, model_id=selected_id)
    persistence_cv = cv_mean_for(cv, model_name="persistence_27day")

    rows: list[dict[str, float | str | None]] = []
    add_row(rows, "27-day persistence", persistence_fixed_metrics, persistence_cv, persistence_private_metrics)
    add_row(rows, "direct MLP", direct_fixed_metrics, None, direct_private_metrics)
    add_row(rows, "selected ExtraTrees", selected_fixed_row, selected_cv, extra_private_metrics)
    add_row(rows, "ensemble 0.7 MLP + 0.3 ExtraTrees", ensemble_fixed_metrics, None, ensemble_private_metrics)
    final = pd.DataFrame(rows)

    final_path = FREEZE_DIR / "final_comparison_table.csv"
    final.to_csv(final_path, index=False)
    print("\nFinal frozen comparison")
    print(final.to_string(index=False))

    private_datetimes = table_all.iloc[private_idx]["target_datetime"]
    write_prediction(OUT_DIR / "final_private_72h_ensemble.csv", private_datetimes, ensemble_private_pred)
    write_prediction(OUT_DIR / "final_private_72h_direct_mlp.csv", private_datetimes, aligned_direct_private)

    print(f"\nSaved {final_path}")
    print(f"Saved {OUT_DIR / 'final_private_72h_ensemble.csv'}")
    print(f"Saved {OUT_DIR / 'final_private_72h_direct_mlp.csv'}")


if __name__ == "__main__":
    main()
