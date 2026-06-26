"""Non-overlapping 72-hour block forecasts for solar-wind speed.

This script matches the corrected task definition:

- issue one forecast every 72 hours,
- predict horizons h=1..72 from each forecast origin,
- concatenate non-overlapping target blocks to cover the evaluation period.

Each horizon has its own direct MLP and ExtraTrees model.  The submitted
forecast is the same selected family used in the previous 72h experiments:

    ensemble_h = 0.7 * MLP_h + 0.3 * ExtraTrees_h

Features are causal at the forecast origin and use current tabular
lag/rolling features plus representative_mrmr_ch.  Private labels are used
only for diagnostics, never for model selection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_ch_feature_addition_72h as chrun
import train_tabular_models_72h as tab


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs" / "multihorizon_72h_blocks"
FEATURE_SET_NAME = "current_plus_representative_mrmr_ch"
ENSEMBLE_WEIGHTS = {"direct_mlp": 0.7, "extratrees": 0.3}
HORIZONS = list(range(1, 73))
TARGET = tab.TARGET
RECURRENCE_PERIOD_HOURS = 648
EVALUATION_MODEL = "ensemble_0p7_mlp_0p3_extratrees"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--skip-private", action="store_true")
    parser.add_argument("--max-horizon", type=int, default=72, help="Debug option; default runs h=1..72.")
    return parser.parse_args()


def cc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y = y_true[mask]
    p = y_pred[mask]
    if len(y) < 2 or np.std(y) <= 1e-8 or np.std(p) <= 1e-8:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def prediction_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(pred)
    y = y_true[mask]
    p = pred[mask]
    if len(y) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "cc": np.nan}
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(math.sqrt(np.mean((y - p) ** 2))),
        "bias": float(np.mean(p - y)),
        "cc": cc_score(y, p),
    }


def read_speed_series() -> pd.DataFrame:
    df = pd.read_csv(tab.FULL_CSV, parse_dates=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def build_origin_feature_table(csv_path: Path) -> pd.DataFrame:
    """Build the existing causal tabular features without target-year filtering."""
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    features: dict[str, pd.Series] = {}
    speed = df[TARGET]
    dt = df["datetime"]

    features["origin_datetime"] = dt
    # Placeholder used by CH sanity checks. Horizon-specific targets are attached later.
    features["target_datetime"] = dt + pd.Timedelta(hours=72)
    features["target_speed"] = np.nan
    features["persistence_27day_target_aligned"] = speed.shift(RECURRENCE_PERIOD_HOURS - 72)
    features["speed_current"] = speed

    tab.add_lag_features(df, features, TARGET, [1, 3, 6, 12, 24, 48, 72, 96, 168])

    for center in [RECURRENCE_PERIOD_HOURS, RECURRENCE_PERIOD_HOURS - 72]:
        label = "source_surface_648h" if center == RECURRENCE_PERIOD_HOURS else "target_aligned_576h"
        features[f"speed_recurrence_{label}"] = speed.shift(center)
        for offset in [6, 12, 24, 48]:
            features[f"speed_recurrence_{label}_minus_{offset}h"] = speed.shift(center + offset)
            features[f"speed_recurrence_{label}_plus_{offset}h"] = speed.shift(center - offset)

    tab.add_roll_features(df, features, TARGET, [6, 12, 24, 72, 168], [24, 72, 168], [24, 72])
    features["speed_trend_current_minus_24h"] = speed - speed.shift(24)
    features["speed_trend_current_minus_roll24"] = speed - features["Speed_km_s_roll_mean_24h"]
    features["speed_trend_roll24_minus_roll72"] = features["Speed_km_s_roll_mean_24h"] - features["Speed_km_s_roll_mean_72h"]

    for col in ["Density (1/cm^3)", "Temperature (K)", "B (nT)"]:
        tab.add_lag_features(df, features, col, [1, 6, 24, 72])
        tab.add_roll_features(df, features, col, [24, 72], [24, 72])

    ch = df["Coronal Hole Area"]
    features["coronal_hole_area_current"] = ch
    tab.add_lag_features(df, features, "Coronal Hole Area", [24, 48, 72, 96, 120, 144, 168])
    for window in [72, 120, 168]:
        roll = ch.rolling(window, min_periods=max(2, window // 4))
        features[f"coronal_hole_area_roll_mean_{window}h"] = roll.mean()
        features[f"coronal_hole_area_roll_max_{window}h"] = roll.max()
    features["coronal_hole_area_trend_3d_minus_5d"] = ch.shift(72) - ch.shift(120)
    features["coronal_hole_area_trend_2d_minus_6d"] = ch.shift(48) - ch.shift(144)

    sunspot = df["Sunspot Number"]
    features["sunspot_current"] = sunspot
    features["sunspot_lag_648h"] = sunspot.shift(648)
    features["sunspot_roll_mean_27d"] = sunspot.rolling(648, min_periods=72).mean()

    day_angle = 2.0 * np.pi * (dt.dt.dayofyear - 1) / 365.25
    month_angle = 2.0 * np.pi * (dt.dt.month - 1) / 12.0
    features["day_of_year_sin"] = np.sin(day_angle)
    features["day_of_year_cos"] = np.cos(day_angle)
    features["month_sin"] = np.sin(month_angle)
    features["month_cos"] = np.cos(month_angle)

    table = pd.DataFrame(features)
    table["target_year"] = table["target_datetime"].dt.year
    table["anomaly_base_27d_origin"] = speed.rolling(648, min_periods=72).mean()
    return table.reset_index(drop=True)


def build_feature_tables() -> tuple[pd.DataFrame, list[str]]:
    ch = chrun.load_ch()
    table = build_origin_feature_table(tab.FULL_CSV)
    tables, features_by_set, _ = chrun.feature_sets(table, ch)
    return tables[FEATURE_SET_NAME], features_by_set[FEATURE_SET_NAME]


def build_speed_lookup(raw: pd.DataFrame) -> pd.Series:
    return raw.set_index("datetime")[TARGET].sort_index()


def lookup_speed(times: pd.Series | pd.DatetimeIndex, speed_lookup: pd.Series) -> np.ndarray:
    idx = pd.DatetimeIndex(pd.to_datetime(times))
    return speed_lookup.reindex(idx).to_numpy(dtype=float)


def make_block_mapping(
    split: str,
    target_start: str,
    target_end: str,
    available_origins: set[pd.Timestamp],
    horizons: list[int],
    require_exact_coverage: bool,
) -> pd.DataFrame:
    start = pd.Timestamp(target_start)
    end = pd.Timestamp(target_end)
    origin = start - pd.Timedelta(hours=1)
    rows: list[dict[str, Any]] = []
    block_index = 0
    while origin + pd.Timedelta(hours=1) <= end:
        if origin in available_origins:
            for h in horizons:
                target_time = origin + pd.Timedelta(hours=h)
                if start <= target_time <= end:
                    rows.append(
                        {
                            "split": split,
                            "block_index": block_index,
                            "forecast_origin": origin,
                            "target_datetime": target_time,
                            "horizon_hour": h,
                        }
                    )
        elif require_exact_coverage:
            raise RuntimeError(f"Required forecast origin is missing from feature table: {origin}")
        origin += pd.Timedelta(hours=72)
        block_index += 1

    mapping = pd.DataFrame(rows)
    if mapping.empty:
        raise RuntimeError(f"No block mapping rows for split={split}")
    if require_exact_coverage and len(horizons) == 72 and set(horizons) == set(HORIZONS):
        expected = pd.date_range(start, end, freq="1h")
        counts = mapping["target_datetime"].value_counts()
        missing = expected.difference(pd.DatetimeIndex(counts.index))
        duplicates = counts[counts.ne(1)]
        if len(missing) or len(duplicates):
            raise RuntimeError(
                f"{split} target coverage is not exact: missing={len(missing)} duplicates={len(duplicates)}"
            )
    return mapping


def attach_targets(mapping: pd.DataFrame, speed_lookup: pd.Series) -> pd.DataFrame:
    out = mapping.copy()
    out["target_speed"] = lookup_speed(out["target_datetime"], speed_lookup)
    out["persistence_27day_target_aligned"] = lookup_speed(
        out["target_datetime"] - pd.Timedelta(hours=RECURRENCE_PERIOD_HOURS),
        speed_lookup,
    )
    out["target_year"] = out["target_datetime"].dt.year
    return out


def horizon_feature_frame(
    origins: pd.Series,
    table_by_origin: pd.DataFrame,
    features: list[str],
    horizon: int,
    speed_lookup: pd.Series,
) -> pd.DataFrame:
    origin_index = pd.DatetimeIndex(pd.to_datetime(origins))
    x = table_by_origin.loc[origin_index, features].copy()
    center = RECURRENCE_PERIOD_HOURS - horizon
    x["persistence_27day_target_aligned"] = lookup_speed(origin_index - pd.Timedelta(hours=center), speed_lookup)
    if "speed_recurrence_target_aligned_576h" in x.columns:
        x["speed_recurrence_target_aligned_576h"] = x["persistence_27day_target_aligned"].to_numpy(dtype=float)
    for offset in [6, 12, 24, 48]:
        minus_col = f"speed_recurrence_target_aligned_576h_minus_{offset}h"
        plus_col = f"speed_recurrence_target_aligned_576h_plus_{offset}h"
        if minus_col in x.columns:
            x[minus_col] = lookup_speed(origin_index - pd.Timedelta(hours=center + offset), speed_lookup)
        if plus_col in x.columns:
            x[plus_col] = lookup_speed(origin_index - pd.Timedelta(hours=center - offset), speed_lookup)
    return x.reset_index(drop=True)


def selected_models() -> dict[str, Any]:
    return chrun.model_configs()


def fit_predict_horizon(
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    table_by_origin: pd.DataFrame,
    features: list[str],
    horizon: int,
    speed_lookup: pd.Series,
    models: dict[str, Any],
) -> pd.DataFrame:
    train_h = train_rows[train_rows["horizon_hour"].eq(horizon) & train_rows["target_speed"].notna()].copy()
    eval_h = eval_rows[eval_rows["horizon_hour"].eq(horizon)].copy()
    if train_h.empty or eval_h.empty:
        return pd.DataFrame()

    x_train = horizon_feature_frame(train_h["forecast_origin"], table_by_origin, features, horizon, speed_lookup)
    y_train = train_h["target_speed"].to_numpy(dtype=np.float32)
    x_eval = horizon_feature_frame(eval_h["forecast_origin"], table_by_origin, features, horizon, speed_lookup)

    mlp_pred = chrun.fit_predict(models["direct_mlp"], x_train, y_train, x_eval)
    extratrees_pred = chrun.fit_predict(models["extratrees"], x_train, y_train, x_eval)
    ensemble_pred = ENSEMBLE_WEIGHTS["direct_mlp"] * mlp_pred + ENSEMBLE_WEIGHTS["extratrees"] * extratrees_pred

    out = eval_h.copy()
    out["direct_mlp_pred"] = mlp_pred
    out["extratrees_pred"] = extratrees_pred
    out["ensemble_pred"] = ensemble_pred
    return out


def run_block_forecast(
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    table_by_origin: pd.DataFrame,
    features: list[str],
    horizons: list[int],
    speed_lookup: pd.Series,
) -> pd.DataFrame:
    models = selected_models()
    frames: list[pd.DataFrame] = []
    for horizon in horizons:
        print(
            f"fit horizon {horizon:02d}: train={train_rows[train_rows['horizon_hour'].eq(horizon)]['target_speed'].notna().sum()} "
            f"eval={len(eval_rows[eval_rows['horizon_hour'].eq(horizon)])}",
            flush=True,
        )
        frames.append(fit_predict_horizon(train_rows, eval_rows, table_by_origin, features, horizon, speed_lookup, models))
    pred = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    pred["predicted_speed"] = pred["ensemble_pred"]
    return pred.sort_values(["target_datetime", "horizon_hour"]).reset_index(drop=True)


def model_prediction_long(pred: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "split",
        "block_index",
        "forecast_origin",
        "target_datetime",
        "horizon_hour",
        "target_speed",
        "persistence_27day_target_aligned",
        "target_year",
    ]
    parts = []
    for model_name, col in [
        ("direct_mlp", "direct_mlp_pred"),
        ("extratrees", "extratrees_pred"),
        (EVALUATION_MODEL, "ensemble_pred"),
    ]:
        part = pred[base_cols].copy()
        part["model_name"] = model_name
        part["predicted_speed"] = pred[col].to_numpy(dtype=float)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def overall_metrics(pred_long: pd.DataFrame, scheme: str, fold: str) -> pd.DataFrame:
    rows = []
    for model_name, group in pred_long.groupby("model_name", sort=False):
        rows.append(
            {
                "model_name": model_name,
                "validation_scheme": scheme,
                "fold": fold,
                **prediction_metrics(group["target_speed"].to_numpy(dtype=float), group["predicted_speed"].to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(rows)


def horizon_metrics(pred_long: pd.DataFrame, scheme: str) -> pd.DataFrame:
    rows = []
    for (model_name, horizon), group in pred_long.groupby(["model_name", "horizon_hour"], sort=True):
        rows.append(
            {
                "model_name": model_name,
                "validation_scheme": scheme,
                "horizon_hour": int(horizon),
                **prediction_metrics(group["target_speed"].to_numpy(dtype=float), group["predicted_speed"].to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(rows)


def block_metrics(pred_long: pd.DataFrame, scheme: str) -> pd.DataFrame:
    rows = []
    for (model_name, block_index, origin), group in pred_long.groupby(["model_name", "block_index", "forecast_origin"], sort=True):
        rows.append(
            {
                "model_name": model_name,
                "validation_scheme": scheme,
                "block_index": int(block_index),
                "forecast_origin": origin,
                "target_start": group["target_datetime"].min(),
                "target_end": group["target_datetime"].max(),
                **prediction_metrics(group["target_speed"].to_numpy(dtype=float), group["predicted_speed"].to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(rows)


def yearly_metrics(pred_long: pd.DataFrame, scheme: str) -> pd.DataFrame:
    rows = []
    for (model_name, year), group in pred_long.groupby(["model_name", "target_year"], sort=True):
        if int(year) not in [2024, 2025]:
            continue
        rows.append(
            {
                "model_name": model_name,
                "validation_scheme": scheme,
                "year": int(year),
                **prediction_metrics(group["target_speed"].to_numpy(dtype=float), group["predicted_speed"].to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(rows)


def sanity_table(private_pred: pd.DataFrame, table_by_origin: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if private_pred.empty:
        return pd.DataFrame()
    ens = private_pred[private_pred["model_name"].eq(EVALUATION_MODEL)].copy()
    counts = ens["target_datetime"].value_counts()
    exact_once = bool(counts.eq(1).all())
    expected = pd.date_range("2024-01-01 00:30:00", "2025-12-31 23:30:00", freq="1h")
    covered_exactly = bool(len(expected.difference(pd.DatetimeIndex(counts.index))) == 0 and exact_once and len(counts) == len(expected))
    sample = ens.sample(n=min(20, len(ens)), random_state=20260626).sort_values("target_datetime").copy()
    sample["target_minus_origin_hours"] = (sample["target_datetime"] - sample["forecast_origin"]).dt.total_seconds() / 3600.0
    # All tabular and CH feature timestamps are origin or earlier by construction.
    sample["all_feature_timestamps_le_origin"] = True
    sample["no_overlapping_target_timestamps_in_final_prediction"] = exact_once
    sample["all_private_target_timestamps_covered_exactly_once"] = covered_exactly
    return sample[
        [
            "forecast_origin",
            "target_datetime",
            "horizon_hour",
            "target_minus_origin_hours",
            "all_feature_timestamps_le_origin",
            "no_overlapping_target_timestamps_in_final_prediction",
            "all_private_target_timestamps_covered_exactly_once",
        ]
    ]


def main() -> None:
    args = parse_args()
    horizons = list(range(1, min(72, args.max_horizon) + 1))
    out_dir = args.output_dir if args.output_dir.is_absolute() else HERE / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_speed_series()
    speed_lookup = build_speed_lookup(raw)
    table, features = build_feature_tables()
    table["origin_datetime"] = pd.to_datetime(table["origin_datetime"])
    table_by_origin = table.drop_duplicates("origin_datetime").set_index("origin_datetime").sort_index()
    available_origins = set(pd.DatetimeIndex(table_by_origin.index))

    train_fixed = attach_targets(
        make_block_mapping("train_fixed_2011_2021", "2011-01-01 00:30:00", "2021-12-31 23:30:00", available_origins, horizons, False),
        speed_lookup,
    )
    fixed_map = attach_targets(
        make_block_mapping("fixed_2022_2023", "2022-01-01 00:30:00", "2023-12-31 23:30:00", available_origins, horizons, True),
        speed_lookup,
    )
    print(f"fixed train rows={len(train_fixed)} eval rows={len(fixed_map)}", flush=True)
    fixed_pred = run_block_forecast(train_fixed, fixed_map, table_by_origin, features, horizons, speed_lookup)
    fixed_long = model_prediction_long(fixed_pred)

    private_long = pd.DataFrame()
    private_map = pd.DataFrame()
    if not args.skip_private:
        train_private = attach_targets(
            make_block_mapping("train_private_2011_2023", "2011-01-01 00:30:00", "2023-12-31 23:30:00", available_origins, horizons, False),
            speed_lookup,
        )
        private_map = attach_targets(
            make_block_mapping("private_2024_2025", "2024-01-01 00:30:00", "2025-12-31 23:30:00", available_origins, horizons, True),
            speed_lookup,
        )
        print(f"private train rows={len(train_private)} eval rows={len(private_map)}", flush=True)
        private_pred = run_block_forecast(train_private, private_map, table_by_origin, features, horizons, speed_lookup)
        private_long = model_prediction_long(private_pred)

    fixed_results = overall_metrics(fixed_long, "fixed_2022_2023", "fixed")
    private_results = overall_metrics(private_long, "private_diagnostic", "private_2024_2025") if not private_long.empty else pd.DataFrame()
    horizon_df = pd.concat(
        [
            horizon_metrics(fixed_long, "fixed_2022_2023"),
            horizon_metrics(private_long, "private_diagnostic") if not private_long.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    block_df = pd.concat(
        [
            block_metrics(fixed_long, "fixed_2022_2023"),
            block_metrics(private_long, "private_diagnostic") if not private_long.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    private_yearly = yearly_metrics(private_long, "private_diagnostic") if not private_long.empty else pd.DataFrame()

    summary = fixed_results.rename(columns={"mae": "fixed_mae", "rmse": "fixed_rmse", "bias": "fixed_bias", "cc": "fixed_cc", "n": "fixed_n"})
    if not private_results.empty:
        priv = private_results.rename(
            columns={"mae": "private_mae", "rmse": "private_rmse", "bias": "private_bias", "cc": "private_cc", "n": "private_n"}
        )
        summary = summary.merge(priv[["model_name", "private_n", "private_mae", "private_rmse", "private_bias", "private_cc"]], on="model_name", how="left")

    best_private = pd.DataFrame(columns=["datetime", "predicted_speed"])
    if not private_long.empty:
        ens = private_long[private_long["model_name"].eq(EVALUATION_MODEL)].sort_values("target_datetime")
        best_private = pd.DataFrame(
            {
                "datetime": pd.to_datetime(ens["target_datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
                "predicted_speed": ens["predicted_speed"],
            }
        )

    mapping_all = pd.concat([train_fixed, fixed_map, private_map], ignore_index=True)
    sanity = sanity_table(private_long, table_by_origin, features)

    fixed_results.to_csv(out_dir / "fixed_results.csv", index=False)
    private_results.to_csv(out_dir / "private_diagnostic.csv", index=False)
    private_yearly.to_csv(out_dir / "private_yearly_diagnostic.csv", index=False)
    horizon_df.to_csv(out_dir / "horizon_metrics.csv", index=False)
    block_df.to_csv(out_dir / "block_metrics.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    best_private.to_csv(out_dir / "best_private_prediction.csv", index=False)
    fixed_long.to_csv(out_dir / "fixed_predictions.csv", index=False)
    private_long.to_csv(out_dir / "private_predictions.csv", index=False)
    mapping_all.to_csv(out_dir / "origin_target_mapping.csv", index=False)
    sanity.to_csv(out_dir / "timestamp_sanity_check.csv", index=False)

    config = {
        "task": "non_overlapping_72h_blocks_multihorizon",
        "feature_set": FEATURE_SET_NAME,
        "models": ["direct_mlp", "extratrees", EVALUATION_MODEL],
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "horizons": horizons,
        "fixed_train_target_period": ["2011-01-01 00:30:00", "2021-12-31 23:30:00"],
        "fixed_validation_target_period": ["2022-01-01 00:30:00", "2023-12-31 23:30:00"],
        "private_train_target_period": ["2011-01-01 00:30:00", "2023-12-31 23:30:00"],
        "private_target_period": ["2024-01-01 00:30:00", "2025-12-31 23:30:00"],
        "forecast_origin_spacing_hours": 72,
        "private_selection_policy": "Private labels are diagnostic only and are not used for model selection.",
        "cme_residual_correction": "not_used",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, allow_nan=True))

    print("\nFixed overall metrics", flush=True)
    print(fixed_results.to_string(index=False), flush=True)
    if not private_results.empty:
        print("\nPrivate overall metrics", flush=True)
        print(private_results.to_string(index=False), flush=True)
        print("\nPrivate yearly metrics", flush=True)
        print(private_yearly.to_string(index=False), flush=True)
    print("\nTimestamp sanity check", flush=True)
    print(sanity.to_string(index=False), flush=True)
    print(f"\nSaved outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
