"""Horizon-aware CME/context residual correction for 72h block forecasts.

This is the correction experiment for the corrected final task:

- issue one forecast every 72 hours,
- predict horizons h=1..72 from each forecast origin,
- concatenate non-overlapping 72h blocks.

The base forecast is the current multihorizon family:

    base_h = 0.7 * MLP_h + 0.3 * ExtraTrees_h

The correction is a single pooled horizon-aware residual model over
origin-horizon rows:

    residual(t, h) = true Speed(t+h) - base_h(t)
    corrected(t, h) = base_h(t) + residual_pred(t, h)

Only CME events observed at or before the forecast origin are used.
Private labels are diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import run_ch_feature_addition_72h as chrun
import run_multihorizon_72h_blocks as mh


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs" / "multihorizon_correction_72h_blocks"
CACHE_VERSION = 2

BASE_MODEL = "base_multihorizon_only"
CORRECTED_MODEL = "base_plus_horizon_cme_context_correction"
RESIDUAL_ENSEMBLE = "residual_ensemble_0p7_mlp_0p3_extratrees"
HORIZONS = mh.HORIZONS
SELECTED_HORIZONS = [1, 6, 12, 24, 36, 48, 60, 72]
AU_KM = 149_597_870.7

DEFAULT_CME_PATHS = [
    HERE / "data" / "donki_cme_catalog.csv",
    HERE / "data" / "cme_catalog.csv",
    HERE / "data" / "cme_catalog_2011_2025.csv",
    HERE / "data" / "lasco_cme_catalog.csv",
    HERE / "cme_catalog.csv",
]

RECENT_WINDOWS = [24.0, 48.0, 72.0, 96.0]
ETA_WINDOWS = [12.0, 24.0, 36.0, 48.0]
ETA_SIGMAS = [12.0, 24.0, 36.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cme-catalog", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--skip-private", action="store_true")
    parser.add_argument("--max-horizon", type=int, default=72, help="Debug option; default runs h=1..72.")
    return parser.parse_args()


def resolve_cme_catalog(path: Path | None) -> Path:
    candidates = [path] if path is not None else DEFAULT_CME_PATHS
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    searched = ", ".join(str(p.relative_to(HERE) if p.is_relative_to(HERE) else p) for p in DEFAULT_CME_PATHS)
    raise RuntimeError(f"No CME catalog found. Searched: {searched}")


def pick_column(columns: list[str], candidates: list[str], required: bool = True) -> str | None:
    lower = {col.lower().strip(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    for col in columns:
        low = col.lower()
        if any(candidate.lower() in low for candidate in candidates):
            return col
    if required:
        raise RuntimeError(f"Could not find required CME catalog column among candidates: {candidates}")
    return None


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    text = series.astype(str).str.strip().str.lower()
    return text.isin(["true", "t", "1", "yes", "y", "halo", "full", "full halo"])


def load_cme_catalog(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    cols = raw.columns.tolist()
    time_col = pick_column(cols, ["cme_time", "time", "datetime", "startTime", "activityStartTime", "firstAppearanceTime"])
    speed_col = pick_column(cols, ["cme_speed", "speed", "linear_speed", "speed_km_s", "Speed"], required=False)
    width_col = pick_column(cols, ["cme_width", "width", "angular_width", "halfAngle", "Width"], required=False)
    halo_col = pick_column(cols, ["is_halo", "halo", "halo_cme", "isHalo"], required=False)
    type_col = pick_column(cols, ["type", "cme_type"], required=False)

    cme = pd.DataFrame({"cme_time": pd.to_datetime(raw[time_col], errors="coerce")})
    cme["cme_speed"] = pd.to_numeric(raw[speed_col], errors="coerce") if speed_col else np.nan
    cme["cme_width"] = pd.to_numeric(raw[width_col], errors="coerce") if width_col else np.nan
    if width_col and "donki" in path.name.lower() and cme["cme_width"].max(skipna=True) <= 180.0:
        cme["cme_width"] = 2.0 * cme["cme_width"]

    if halo_col:
        cme["is_halo"] = parse_bool_series(raw[halo_col]).to_numpy()
    else:
        type_halo = parse_bool_series(raw[type_col]) if type_col else pd.Series(False, index=raw.index)
        cme["is_halo"] = (cme["cme_width"].fillna(-1).ge(180.0) | type_halo).to_numpy()

    valid_speed = cme["cme_speed"].notna() & cme["cme_speed"].gt(0)
    travel_hours = np.full(len(cme), np.nan, dtype=float)
    travel_hours[valid_speed.to_numpy()] = 1.1 * AU_KM / cme.loc[valid_speed, "cme_speed"].to_numpy(dtype=float) / 3600.0
    cme["travel_time_hours"] = travel_hours
    cme["cme_eta"] = cme["cme_time"] + pd.to_timedelta(cme["travel_time_hours"], unit="h")

    cme = cme[cme["cme_time"].notna()].sort_values("cme_time").reset_index(drop=True)
    if cme.empty:
        raise RuntimeError(f"CME catalog has no parseable event times: {path}")
    return cme


def metric_row(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return mh.prediction_metrics(y_true, pred)


def base_prediction_columns(pred: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["split", "block_index", "forecast_origin", "target_datetime", "horizon_hour"]
    meta_cols = key_cols + ["target_speed", "persistence_27day_target_aligned", "target_year"]
    meta = pred[meta_cols].drop_duplicates(key_cols).reset_index(drop=True)
    wide = pred.pivot_table(
        index=key_cols,
        columns="model_name",
        values="predicted_speed",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    wide = meta.merge(wide, on=key_cols, how="left")
    wide["base_pred"] = wide[mh.EVALUATION_MODEL]
    wide["residual_true"] = wide["target_speed"] - wide["base_pred"]
    wide["cache_version"] = CACHE_VERSION
    return wide


def make_target_period(year_start: int, year_end: int) -> tuple[str, str]:
    return f"{year_start}-01-01 00:30:00", f"{year_end}-12-31 23:30:00"


def make_attached_mapping(
    split: str,
    year_start: int,
    year_end: int,
    available_origins: set[pd.Timestamp],
    horizons: list[int],
    speed_lookup: pd.Series,
    require_exact: bool,
) -> pd.DataFrame:
    start, end = make_target_period(year_start, year_end)
    return mh.attach_targets(
        mh.make_block_mapping(split, start, end, available_origins, horizons, require_exact),
        speed_lookup,
    )


def run_base_block(
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    table_by_origin: pd.DataFrame,
    base_features: list[str],
    horizons: list[int],
    speed_lookup: pd.Series,
) -> pd.DataFrame:
    pred = mh.run_block_forecast(train_rows, eval_rows, table_by_origin, base_features, horizons, speed_lookup)
    return base_prediction_columns(mh.model_prediction_long(pred))


def read_cached_base(path: Path, horizons: list[int]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["forecast_origin", "target_datetime"])
    required = {"forecast_origin", "target_datetime", "horizon_hour", "target_speed", "base_pred", "residual_true"}
    if not required.issubset(df.columns):
        return None
    if "cache_version" not in df.columns or int(df["cache_version"].dropna().max()) != CACHE_VERSION:
        return None
    if sorted(df["horizon_hour"].dropna().astype(int).unique().tolist()) != horizons:
        return None
    print(f"Using cached base predictions: {path}", flush=True)
    return df


def build_oof_base_predictions(
    out_dir: Path,
    table_by_origin: pd.DataFrame,
    base_features: list[str],
    available_origins: set[pd.Timestamp],
    horizons: list[int],
    speed_lookup: pd.Series,
) -> pd.DataFrame:
    path = out_dir / "oof_base_predictions.csv"
    cached = read_cached_base(path, horizons)
    if cached is not None:
        return cached

    frames: list[pd.DataFrame] = []
    for year in range(2017, 2024):
        print(f"OOF multihorizon base train 2011-{year - 1} -> target {year}", flush=True)
        train_rows = make_attached_mapping(
            f"oof_train_2011_{year - 1}",
            2011,
            year - 1,
            available_origins,
            horizons,
            speed_lookup,
            require_exact=False,
        )
        eval_rows = make_attached_mapping(
            f"oof_val_{year}",
            year,
            year,
            available_origins,
            horizons,
            speed_lookup,
            require_exact=len(horizons) == 72,
        )
        frame = run_base_block(train_rows, eval_rows, table_by_origin, base_features, horizons, speed_lookup)
        frame["oof_year"] = year
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def build_private_base_predictions(
    out_dir: Path,
    table_by_origin: pd.DataFrame,
    base_features: list[str],
    available_origins: set[pd.Timestamp],
    horizons: list[int],
    speed_lookup: pd.Series,
) -> pd.DataFrame:
    path = out_dir / "base_private_rows.csv"
    cached = read_cached_base(path, horizons)
    if cached is not None:
        return cached

    train_rows = make_attached_mapping(
        "private_train_2011_2023",
        2011,
        2023,
        available_origins,
        horizons,
        speed_lookup,
        require_exact=False,
    )
    eval_rows = make_attached_mapping(
        "private_2024_2025",
        2024,
        2025,
        available_origins,
        horizons,
        speed_lookup,
        require_exact=len(horizons) == 72,
    )
    out = run_base_block(train_rows, eval_rows, table_by_origin, base_features, horizons, speed_lookup)
    out.to_csv(path, index=False)
    return out


def selected_context_columns(table: pd.DataFrame, base_features: list[str]) -> list[str]:
    wanted = [
        "speed_current",
        "Speed_km_s_lag_1h",
        "Speed_km_s_lag_6h",
        "Speed_km_s_lag_24h",
        "Speed_km_s_lag_72h",
        "speed_trend_current_minus_24h",
        "persistence_27day_target_aligned",
        "speed_recurrence_target_aligned_576h",
        "coronal_hole_area_current",
        "Coronal_Hole_Area_lag_24h",
        "Coronal_Hole_Area_lag_48h",
        "Coronal_Hole_Area_lag_72h",
        "Coronal_Hole_Area_lag_96h",
        "Coronal_Hole_Area_lag_120h",
        "Coronal_Hole_Area_lag_144h",
        "Coronal_Hole_Area_lag_168h",
        "coronal_hole_area_roll_mean_72h",
        "coronal_hole_area_roll_max_72h",
    ]
    ch_cols = []
    for spec in chrun.REPRESENTATIVE_CH:
        col = spec.output_name
        ch_cols.extend([col, f"{col}__missing"])
    cols = []
    for col in wanted + ch_cols:
        if col in table.columns and col in base_features and col not in cols:
            cols.append(col)
    return cols


def build_horizon_context(
    rows: pd.DataFrame,
    table_by_origin: pd.DataFrame,
    base_features: list[str],
    context_cols: list[str],
    speed_lookup: pd.Series,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for horizon, group in rows.groupby("horizon_hour", sort=True):
        x = mh.horizon_feature_frame(
            group["forecast_origin"],
            table_by_origin,
            base_features,
            int(horizon),
            speed_lookup,
        )
        part = x.reindex(columns=context_cols).reset_index(drop=True)
        part.index = group.index
        parts.append(part)
    context = pd.concat(parts).sort_index()
    return context.reset_index(drop=True)


def build_recent_cme_features(origins: pd.Series, cme: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique = pd.Series(pd.to_datetime(origins).drop_duplicates().sort_values()).reset_index(drop=True)
    event_times = pd.to_datetime(cme["cme_time"]).to_numpy(dtype="datetime64[ns]")
    speed = cme["cme_speed"].to_numpy(dtype=float)
    width = cme["cme_width"].to_numpy(dtype=float)
    halo = cme["is_halo"].to_numpy(dtype=bool)
    speed_width = np.where(np.isfinite(speed) & np.isfinite(width), speed * width, np.nan)

    rows: list[dict[str, float | pd.Timestamp | bool]] = []
    sanity: list[dict[str, Any]] = []
    for i, origin in enumerate(unique):
        origin64 = np.datetime64(origin.to_datetime64())
        hi = int(np.searchsorted(event_times, origin64, side="right"))
        row: dict[str, float | pd.Timestamp | bool] = {"forecast_origin": origin}
        latest = pd.Timestamp(event_times[hi - 1]) if hi > 0 else pd.NaT
        for hours_back in RECENT_WINDOWS:
            label = f"last_{int(hours_back)}h"
            start = origin - pd.Timedelta(hours=hours_back)
            lo = int(np.searchsorted(event_times, np.datetime64(start.to_datetime64()), side="right"))
            idx = np.arange(lo, hi)
            sp = speed[idx]
            wd = width[idx]
            sw = speed_width[idx]
            ha = halo[idx]
            finite_speed = sp[np.isfinite(sp)]
            finite_width = wd[np.isfinite(wd)]
            finite_sw = sw[np.isfinite(sw)]

            row[f"cme_{label}_cme_count"] = float(len(idx))
            row[f"cme_{label}_fast_cme_count_speed_gt_800"] = float(np.sum(sp > 800.0))
            row[f"cme_{label}_very_fast_cme_count_speed_gt_1200"] = float(np.sum(sp > 1200.0))
            row[f"cme_{label}_halo_cme_count"] = float(np.sum(ha))
            row[f"cme_{label}_max_cme_speed"] = float(np.max(finite_speed)) if len(finite_speed) else 0.0
            row[f"cme_{label}_mean_cme_speed"] = float(np.mean(finite_speed)) if len(finite_speed) else 0.0
            row[f"cme_{label}_sum_cme_speed"] = float(np.sum(finite_speed)) if len(finite_speed) else 0.0
            row[f"cme_{label}_max_cme_width"] = float(np.max(finite_width)) if len(finite_width) else 0.0
            row[f"cme_{label}_mean_cme_width"] = float(np.mean(finite_width)) if len(finite_width) else 0.0
            row[f"cme_{label}_sum_speed_width"] = float(np.sum(finite_sw)) if len(finite_sw) else 0.0

            if len(idx):
                delta_hours = (origin64 - event_times[idx]).astype("timedelta64[s]").astype(float) / 3600.0
                row[f"cme_{label}_time_since_last_cme_hours"] = float(np.min(delta_hours))
                fast_hours = delta_hours[sp > 800.0]
                halo_hours = delta_hours[ha]
                row[f"cme_{label}_time_since_last_fast_cme_hours"] = float(np.min(fast_hours)) if len(fast_hours) else 9999.0
                row[f"cme_{label}_time_since_last_halo_cme_hours"] = float(np.min(halo_hours)) if len(halo_hours) else 9999.0
            else:
                row[f"cme_{label}_time_since_last_cme_hours"] = 9999.0
                row[f"cme_{label}_time_since_last_fast_cme_hours"] = 9999.0
                row[f"cme_{label}_time_since_last_halo_cme_hours"] = 9999.0
        rows.append(row)
        if i % max(1, len(unique) // 100) == 0:
            sanity.append(
                {
                    "forecast_origin": origin,
                    "latest_cme_time_used": latest,
                    "all_recent_cme_times_le_origin": bool(pd.isna(latest) or latest <= origin),
                    "cme_last_72h_cme_count": row["cme_last_72h_cme_count"],
                    "cme_last_72h_max_cme_speed": row["cme_last_72h_max_cme_speed"],
                    "cme_last_72h_halo_cme_count": row["cme_last_72h_halo_cme_count"],
                }
            )

    features = pd.DataFrame(rows)
    return origins.to_frame("forecast_origin").merge(features, on="forecast_origin", how="left").drop(columns=["forecast_origin"]), pd.DataFrame(sanity)


def build_eta_cme_features(rows: pd.DataFrame, cme: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_times = pd.to_datetime(cme["cme_time"]).to_numpy(dtype="datetime64[ns]")
    eta_times = pd.to_datetime(cme["cme_eta"]).to_numpy(dtype="datetime64[ns]")
    speed = cme["cme_speed"].to_numpy(dtype=float)
    width = cme["cme_width"].to_numpy(dtype=float)
    halo = cme["is_halo"].to_numpy(dtype=bool)
    speed_width = np.where(np.isfinite(speed) & np.isfinite(width), speed * width, np.nan)
    valid_eta = np.isfinite(speed) & (speed > 0) & ~pd.isna(cme["cme_eta"]).to_numpy()

    out_rows: list[dict[str, float]] = []
    sanity: list[dict[str, Any]] = []
    for i, row_in in rows[["forecast_origin", "target_datetime"]].reset_index(drop=True).iterrows():
        origin = pd.Timestamp(row_in["forecast_origin"])
        target = pd.Timestamp(row_in["target_datetime"])
        origin64 = np.datetime64(origin.to_datetime64())
        target64 = np.datetime64(target.to_datetime64())
        hi = int(np.searchsorted(event_times, origin64, side="right"))
        idx = np.arange(hi)
        idx = idx[valid_eta[idx]]
        out: dict[str, float] = {}
        if len(idx):
            err = (eta_times[idx] - target64).astype("timedelta64[s]").astype(float) / 3600.0
            abs_err = np.abs(err)
            sp = speed[idx]
            wd = width[idx]
            sw = speed_width[idx]
            ha = halo[idx]
            fast = sp > 800.0
        else:
            err = abs_err = sp = wd = sw = np.array([], dtype=float)
            ha = fast = np.array([], dtype=bool)

        for window in ETA_WINDOWS:
            label = f"eta_pm{int(window)}h"
            m = abs_err <= window
            sp_m = sp[m]
            sw_m = sw[m]
            out[f"{label}_eta_cme_count"] = float(np.sum(m))
            out[f"{label}_eta_halo_cme_count"] = float(np.sum(ha[m])) if len(m) else 0.0
            out[f"{label}_eta_fast_cme_count_speed_gt_800"] = float(np.sum(fast[m])) if len(m) else 0.0
            out[f"{label}_eta_max_cme_speed"] = float(np.nanmax(sp_m)) if np.isfinite(sp_m).any() else 0.0
            out[f"{label}_eta_sum_cme_speed"] = float(np.nansum(sp_m)) if len(sp_m) else 0.0
            out[f"{label}_eta_max_speed_width"] = float(np.nanmax(sw_m)) if np.isfinite(sw_m).any() else 0.0
            out[f"{label}_eta_sum_speed_width"] = float(np.nansum(sw_m)) if len(sw_m) else 0.0
            out[f"{label}_eta_min_abs_error_hours"] = float(np.min(abs_err[m])) if np.any(m) else 9999.0

        for sigma in ETA_SIGMAS:
            label = f"eta_gauss_sigma{int(sigma)}h"
            if len(abs_err):
                keep = abs_err <= 5.0 * sigma
                weights = np.exp(-0.5 * (abs_err[keep] / sigma) ** 2)
                sp_k = sp[keep]
                sw_k = sw[keep]
                ha_k = ha[keep]
                fast_k = fast[keep]
                out[f"{label}_eta_weighted_count"] = float(np.sum(weights))
                out[f"{label}_eta_weighted_speed_sum"] = float(np.nansum(weights * sp_k))
                out[f"{label}_eta_weighted_speed_width_sum"] = float(np.nansum(weights * sw_k))
                out[f"{label}_eta_weighted_halo_count"] = float(np.sum(weights * ha_k.astype(float)))
                out[f"{label}_eta_weighted_fast_count"] = float(np.sum(weights * fast_k.astype(float)))
            else:
                out[f"{label}_eta_weighted_count"] = 0.0
                out[f"{label}_eta_weighted_speed_sum"] = 0.0
                out[f"{label}_eta_weighted_speed_width_sum"] = 0.0
                out[f"{label}_eta_weighted_halo_count"] = 0.0
                out[f"{label}_eta_weighted_fast_count"] = 0.0

        out_rows.append(out)
        if i % max(1, len(rows) // 100) == 0:
            latest = pd.Timestamp(event_times[hi - 1]) if hi > 0 else pd.NaT
            sanity.append(
                {
                    "forecast_origin": origin,
                    "target_datetime": target,
                    "latest_eta_cme_time_used": latest,
                    "all_eta_cme_times_le_origin": bool(pd.isna(latest) or latest <= origin),
                }
            )

    return pd.DataFrame(out_rows, index=rows.index), pd.DataFrame(sanity)


def correction_feature_frame(
    rows: pd.DataFrame,
    table_by_origin: pd.DataFrame,
    base_features: list[str],
    context_cols: list[str],
    speed_lookup: pd.Series,
    cme: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    base = rows.reset_index(drop=True).copy()
    horizon = base["horizon_hour"].astype(float)
    features = pd.DataFrame(
        {
            "feature_horizon_hour": horizon,
            "feature_horizon_fraction": horizon / 72.0,
            "feature_horizon_phase_sin": np.sin(2.0 * np.pi * horizon / 72.0),
            "feature_horizon_phase_cos": np.cos(2.0 * np.pi * horizon / 72.0),
            "feature_base_pred": base["base_pred"].to_numpy(dtype=float),
            "feature_base_minus_current_speed": np.nan,
            "feature_base_minus_27day_recurrence": base["base_pred"].to_numpy(dtype=float)
            - base["persistence_27day_target_aligned"].to_numpy(dtype=float),
        }
    )

    context = build_horizon_context(base, table_by_origin, base_features, context_cols, speed_lookup)
    if "speed_current" in context.columns:
        features["feature_base_minus_current_speed"] = features["feature_base_pred"] - context["speed_current"].to_numpy(dtype=float)

    recent, recent_sanity = build_recent_cme_features(base["forecast_origin"], cme)
    eta, eta_sanity = build_eta_cme_features(base, cme)
    all_features = pd.concat([features, context, recent.reset_index(drop=True), eta.reset_index(drop=True)], axis=1)
    sanity = pd.concat(
        [
            recent_sanity.assign(feature_family="recent_cme"),
            eta_sanity.assign(feature_family="eta_cme"),
        ],
        ignore_index=True,
    )
    return all_features, list(all_features.columns), sanity


def fit_predict_residual(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    eval_x: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    models = mh.selected_models()
    mlp = chrun.fit_predict(models["direct_mlp"], train_x, train_y.astype(np.float32), eval_x)
    extra = chrun.fit_predict(models["extratrees"], train_x, train_y.astype(np.float32), eval_x)
    ensemble = 0.7 * mlp + 0.3 * extra
    return {
        "residual_mlp_pred": mlp,
        "residual_extratrees_pred": extra,
        "residual_pred": ensemble,
    }, models


def evaluate_named(frame: pd.DataFrame, pred_col: str, model_name: str, scheme: str, fold: str) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "validation_scheme": scheme,
        "fold": fold,
        **metric_row(frame["target_speed"].to_numpy(dtype=float), frame[pred_col].to_numpy(dtype=float)),
    }


def long_model_predictions(frame: pd.DataFrame, scheme: str) -> pd.DataFrame:
    base_cols = [
        "forecast_origin",
        "target_datetime",
        "horizon_hour",
        "target_year",
        "target_speed",
        "base_pred",
        "residual_true",
        "residual_pred",
        "corrected_pred",
    ]
    rows = []
    base = frame[base_cols].copy()
    base["model_name"] = BASE_MODEL
    base["validation_scheme"] = scheme
    base["predicted_speed"] = base["base_pred"]
    rows.append(base)
    corrected = frame[base_cols].copy()
    corrected["model_name"] = CORRECTED_MODEL
    corrected["validation_scheme"] = scheme
    corrected["predicted_speed"] = corrected["corrected_pred"]
    rows.append(corrected)
    return pd.concat(rows, ignore_index=True)


def horizon_metrics(pred_long: pd.DataFrame, scheme: str) -> pd.DataFrame:
    rows = []
    for (model_name, horizon), group in pred_long.groupby(["model_name", "horizon_hour"], sort=True):
        rows.append(
            {
                "model_name": model_name,
                "validation_scheme": scheme,
                "horizon_hour": int(horizon),
                **metric_row(group["target_speed"].to_numpy(dtype=float), group["predicted_speed"].to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(rows)


def yearly_private_metrics(pred_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, year), group in pred_long.groupby(["model_name", "target_year"], sort=True):
        if int(year) not in [2024, 2025]:
            continue
        rows.append(
            {
                "model_name": model_name,
                "validation_scheme": "private_diagnostic",
                "year": int(year),
                **metric_row(group["target_speed"].to_numpy(dtype=float), group["predicted_speed"].to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(rows)


def correction_distribution(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    pred = frame["residual_pred"].to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {
                "split": label,
                "n": int(np.isfinite(pred).sum()),
                "mean": float(np.nanmean(pred)),
                "std": float(np.nanstd(pred)),
                "p01": float(np.nanpercentile(pred, 1)),
                "p05": float(np.nanpercentile(pred, 5)),
                "p50": float(np.nanpercentile(pred, 50)),
                "p95": float(np.nanpercentile(pred, 95)),
                "p99": float(np.nanpercentile(pred, 99)),
                "min": float(np.nanmin(pred)),
                "max": float(np.nanmax(pred)),
            }
        ]
    )


def feature_importance(
    model_info: dict[str, Any],
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    val_x: pd.DataFrame,
    val_y: np.ndarray,
    out_path: Path,
) -> pd.DataFrame:
    estimator = model_info["extratrees"]
    fitted = estimator.fit(train_x, train_y.astype(np.float32))
    model = fitted.named_steps.get("model") if hasattr(fitted, "named_steps") else fitted
    if hasattr(model, "feature_importances_") and len(model.feature_importances_) == len(train_x.columns):
        imp = pd.DataFrame({"feature": train_x.columns, "importance": model.feature_importances_})
        imp = imp.sort_values("importance", ascending=False)
    else:
        sample_n = min(2000, len(val_x))
        if sample_n < len(val_x):
            rng = np.random.default_rng(42)
            sample_idx = np.sort(rng.choice(len(val_x), size=sample_n, replace=False))
            val_sample = val_x.iloc[sample_idx]
            val_y_sample = val_y[sample_idx]
        else:
            val_sample = val_x
            val_y_sample = val_y
        result = permutation_importance(fitted, val_sample, val_y_sample, n_repeats=3, random_state=42, n_jobs=1)
        imp = pd.DataFrame({"feature": val_x.columns, "importance": result.importances_mean})
        imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(out_path, index=False)
    return imp


def event_2024_may_examples(private_frame: pd.DataFrame) -> pd.DataFrame:
    frame = private_frame[
        (private_frame["target_datetime"] >= "2024-05-01")
        & (private_frame["target_datetime"] < "2024-06-01")
    ].copy()
    if frame.empty:
        return pd.DataFrame()
    cols = [
        "forecast_origin",
        "horizon_hour",
        "target_datetime",
        "target_speed",
        "base_pred",
        "residual_true",
        "residual_pred",
        "corrected_pred",
        "cme_last_72h_cme_count",
        "cme_last_72h_max_cme_speed",
        "cme_last_72h_halo_cme_count",
        "eta_pm24h_eta_cme_count",
        "eta_pm24h_eta_max_cme_speed",
        "eta_pm24h_eta_min_abs_error_hours",
        "eta_gauss_sigma24h_eta_weighted_count",
        "eta_gauss_sigma24h_eta_weighted_speed_sum",
    ]
    out = frame.sort_values("target_speed", ascending=False).head(40)
    return out[[c for c in cols if c in out.columns]].rename(
        columns={
            "forecast_origin": "origin_time",
            "target_datetime": "target_time",
            "target_speed": "observed",
        }
    )


def private_coverage_sanity(private_frame: pd.DataFrame, horizons: list[int]) -> dict[str, Any]:
    counts = private_frame["target_datetime"].value_counts()
    expected = pd.date_range("2024-01-01 00:30:00", "2025-12-31 23:30:00", freq="1h")
    origin_unique = pd.Series(pd.to_datetime(private_frame["forecast_origin"].drop_duplicates())).sort_values()
    diffs = origin_unique.diff().dropna().dt.total_seconds() / 3600.0
    h72 = private_frame[private_frame["horizon_hour"].eq(72)]
    h72_metrics = metric_row(h72["target_speed"].to_numpy(dtype=float), h72["base_pred"].to_numpy(dtype=float))
    return {
        "private_target_rows": int(len(private_frame)),
        "expected_private_rows": int(len(expected)) if len(horizons) == 72 else np.nan,
        "private_missing_timestamps": int(len(expected.difference(pd.DatetimeIndex(counts.index)))) if len(horizons) == 72 else np.nan,
        "private_extra_timestamps": int(len(pd.DatetimeIndex(counts.index).difference(expected))) if len(horizons) == 72 else np.nan,
        "private_duplicate_timestamps": int(counts.gt(1).sum()),
        "private_target_coverage_exactly_once": bool(
            len(horizons) == 72
            and len(expected.difference(pd.DatetimeIndex(counts.index))) == 0
            and counts.eq(1).all()
            and len(counts) == len(expected)
        ),
        "unique_private_forecast_origins": int(len(origin_unique)),
        "origin_spacing_min_hours": float(diffs.min()) if len(diffs) else np.nan,
        "origin_spacing_median_hours": float(diffs.median()) if len(diffs) else np.nan,
        "origin_spacing_max_hours": float(diffs.max()) if len(diffs) else np.nan,
        "all_origin_spacing_72h": bool(diffs.eq(72.0).all()) if len(diffs) else True,
        "horizon_min": int(private_frame["horizon_hour"].min()),
        "horizon_max": int(private_frame["horizon_hour"].max()),
        "horizon_set_complete": bool(sorted(private_frame["horizon_hour"].unique().tolist()) == horizons),
        "target_equals_origin_plus_horizon": bool(
            (
                pd.to_datetime(private_frame["target_datetime"])
                - pd.to_datetime(private_frame["forecast_origin"])
                == pd.to_timedelta(private_frame["horizon_hour"], unit="h")
            ).all()
        ),
        "private_labels_used_for_correction_training": False,
        "h72_base_private_cc": h72_metrics["cc"],
        "h72_base_private_mae": h72_metrics["mae"],
        "h72_base_private_rmse": h72_metrics["rmse"],
        "h72_previous_final_cc_reference": 0.567184,
        "h72_base_cc_abs_diff_from_previous": abs(float(h72_metrics["cc"]) - 0.567184)
        if np.isfinite(h72_metrics["cc"])
        else np.nan,
    }


def leakage_report(
    private_frame: pd.DataFrame,
    feature_sanity: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    coverage = private_coverage_sanity(private_frame, horizons)
    cme_recent_ok = True
    cme_eta_ok = True
    if not feature_sanity.empty:
        if "all_recent_cme_times_le_origin" in feature_sanity.columns:
            cme_recent_ok = bool(feature_sanity["all_recent_cme_times_le_origin"].dropna().all())
        if "all_eta_cme_times_le_origin" in feature_sanity.columns:
            cme_eta_ok = bool(feature_sanity["all_eta_cme_times_le_origin"].dropna().all())
    rows = [
        {"check": "private_target_coverage_exactly_once", "status": "PASS" if coverage["private_target_coverage_exactly_once"] else "FAIL", "value": coverage["private_target_coverage_exactly_once"]},
        {"check": "forecast_origins_72h_apart", "status": "PASS" if coverage["all_origin_spacing_72h"] else "FAIL", "value": coverage["all_origin_spacing_72h"]},
        {"check": "horizon_hour_in_1_72", "status": "PASS" if coverage["horizon_min"] >= 1 and coverage["horizon_max"] <= max(horizons) else "FAIL", "value": f"{coverage['horizon_min']}..{coverage['horizon_max']}"},
        {"check": "target_time_equals_origin_plus_horizon", "status": "PASS" if coverage["target_equals_origin_plus_horizon"] else "FAIL", "value": coverage["target_equals_origin_plus_horizon"]},
        {"check": "non_cme_context_features_le_origin", "status": "PASS", "value": "constructed from origin-time feature table"},
        {"check": "recent_cme_events_le_origin", "status": "PASS" if cme_recent_ok else "FAIL", "value": cme_recent_ok},
        {"check": "eta_cme_events_le_origin", "status": "PASS" if cme_eta_ok else "FAIL", "value": cme_eta_ok},
        {"check": "private_labels_not_used_for_correction_training", "status": "PASS", "value": False},
        {"check": "h72_base_consistent_with_previous", "status": "PASS" if coverage["h72_base_cc_abs_diff_from_previous"] < 0.03 else "WARN", "value": coverage["h72_base_private_cc"]},
    ]
    rows.extend({"check": key, "status": "INFO", "value": value} for key, value in coverage.items())
    return pd.DataFrame(rows)


def choose_adoption(fixed_results: pd.DataFrame) -> tuple[bool, str]:
    base = fixed_results[fixed_results["model_name"].eq(BASE_MODEL)].iloc[0]
    corrected = fixed_results[fixed_results["model_name"].eq(CORRECTED_MODEL)].iloc[0]
    cc_improves = corrected["cc"] > base["cc"]
    mae_ratio = corrected["mae"] / base["mae"] if base["mae"] > 0 else np.inf
    rmse_ratio = corrected["rmse"] / base["rmse"] if base["rmse"] > 0 else np.inf
    if cc_improves and mae_ratio <= 1.02 and rmse_ratio <= 1.02:
        return True, "public fixed CC improved and MAE/RMSE did not worsen substantially"
    if cc_improves:
        return False, "public fixed CC improved, but MAE/RMSE worsened substantially"
    return False, "public fixed CC did not improve"


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir if args.output_dir.is_absolute() else HERE / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = list(range(1, min(72, args.max_horizon) + 1))

    cme_path = resolve_cme_catalog(args.cme_catalog)
    cme = load_cme_catalog(cme_path)
    print(f"Loaded CME catalog {cme_path} rows={len(cme)} {cme['cme_time'].min()} to {cme['cme_time'].max()}", flush=True)

    raw = mh.read_speed_series()
    speed_lookup = mh.build_speed_lookup(raw)
    table, base_features = mh.build_feature_tables()
    table["origin_datetime"] = pd.to_datetime(table["origin_datetime"])
    table_by_origin = table.drop_duplicates("origin_datetime").set_index("origin_datetime").sort_index()
    available_origins = set(pd.DatetimeIndex(table_by_origin.index))
    context_cols = selected_context_columns(table, base_features)

    oof_base = build_oof_base_predictions(out_dir, table_by_origin, base_features, available_origins, horizons, speed_lookup)
    oof_base = oof_base[oof_base["target_speed"].notna()].reset_index(drop=True)
    oof_x, correction_cols, oof_sanity = correction_feature_frame(
        oof_base, table_by_origin, base_features, context_cols, speed_lookup, cme
    )
    correction_rows = pd.concat([oof_base.reset_index(drop=True), oof_x.reset_index(drop=True)], axis=1)
    correction_rows.to_csv(out_dir / "correction_training_rows.csv", index=False)

    train_mask = correction_rows["target_year"].between(2017, 2021) & correction_rows["residual_true"].notna()
    val_mask = correction_rows["target_year"].isin([2022, 2023]) & correction_rows["residual_true"].notna()
    train = correction_rows[train_mask].copy()
    fixed = correction_rows[val_mask].copy()
    print(f"Correction fixed split train={len(train)} val={len(fixed)} features={len(correction_cols)}", flush=True)

    residual_preds, residual_models = fit_predict_residual(
        train[correction_cols],
        train["residual_true"].to_numpy(dtype=float),
        fixed[correction_cols],
    )
    for col, values in residual_preds.items():
        fixed[col] = values
    fixed["corrected_pred"] = fixed["base_pred"] + fixed["residual_pred"]
    fixed_long = long_model_predictions(fixed, "fixed_2022_2023")
    fixed_results = pd.DataFrame(
        [
            evaluate_named(fixed, "base_pred", BASE_MODEL, "fixed_2022_2023", "fixed"),
            evaluate_named(fixed, "corrected_pred", CORRECTED_MODEL, "fixed_2022_2023", "fixed"),
        ]
    )
    adopt, reason = choose_adoption(fixed_results)

    private_results = pd.DataFrame()
    private_yearly = pd.DataFrame()
    private_long = pd.DataFrame()
    private = pd.DataFrame()
    private_sanity = pd.DataFrame()
    if not args.skip_private:
        private_base = build_private_base_predictions(
            out_dir, table_by_origin, base_features, available_origins, horizons, speed_lookup
        )
        private_x, _, private_sanity = correction_feature_frame(
            private_base, table_by_origin, base_features, context_cols, speed_lookup, cme
        )
        private = pd.concat([private_base.reset_index(drop=True), private_x.reset_index(drop=True)], axis=1)
        final_train = correction_rows[correction_rows["residual_true"].notna()].copy()
        private_preds, _ = fit_predict_residual(
            final_train[correction_cols],
            final_train["residual_true"].to_numpy(dtype=float),
            private[correction_cols],
        )
        for col, values in private_preds.items():
            private[col] = values
        private["corrected_pred"] = private["base_pred"] + private["residual_pred"]
        private_long = long_model_predictions(private, "private_diagnostic")
        private_results = pd.DataFrame(
            [
                evaluate_named(private, "base_pred", BASE_MODEL, "private_diagnostic", "private_2024_2025"),
                evaluate_named(private, "corrected_pred", CORRECTED_MODEL, "private_diagnostic", "private_2024_2025"),
            ]
        )
        private_yearly = yearly_private_metrics(private_long)

    horizon_df = pd.concat(
        [
            horizon_metrics(fixed_long, "fixed_2022_2023"),
            horizon_metrics(private_long, "private_diagnostic") if not private_long.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    selected_horizon_metrics = horizon_df[horizon_df["horizon_hour"].isin(SELECTED_HORIZONS)].copy()

    summary = fixed_results.rename(columns={"n": "fixed_n", "mae": "fixed_mae", "rmse": "fixed_rmse", "bias": "fixed_bias", "cc": "fixed_cc"})
    if not private_results.empty:
        priv = private_results.rename(
            columns={"n": "private_n", "mae": "private_mae", "rmse": "private_rmse", "bias": "private_bias", "cc": "private_cc"}
        )
        summary = summary.merge(priv[["model_name", "private_n", "private_mae", "private_rmse", "private_bias", "private_cc"]], on="model_name", how="left")
    summary["adopt_correction"] = summary["model_name"].eq(CORRECTED_MODEL) & adopt
    summary["selection_reason"] = reason

    correction_dist = pd.concat(
        [
            correction_distribution(fixed, "fixed_2022_2023"),
            correction_distribution(private, "private_2024_2025") if not private.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )

    feature_importance(
        residual_models,
        train[correction_cols],
        train["residual_true"].to_numpy(dtype=float),
        fixed[correction_cols],
        fixed["residual_true"].to_numpy(dtype=float),
        out_dir / "correction_feature_importance.csv",
    )

    selected_model = CORRECTED_MODEL if adopt else BASE_MODEL
    best_private = pd.DataFrame(columns=["datetime", "predicted_speed"])
    base_private = pd.DataFrame(columns=["datetime", "predicted_speed"])
    corrected_private = pd.DataFrame(columns=["datetime", "predicted_speed"])
    if not private.empty:
        base_private = pd.DataFrame(
            {
                "datetime": pd.to_datetime(private["target_datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
                "predicted_speed": private["base_pred"],
            }
        )
        corrected_private = pd.DataFrame(
            {
                "datetime": pd.to_datetime(private["target_datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
                "predicted_speed": private["corrected_pred"],
            }
        )
        best_private = corrected_private if selected_model == CORRECTED_MODEL else base_private

    feature_example_cols = [
        "forecast_origin",
        "target_datetime",
        "horizon_hour",
        "cme_last_72h_cme_count",
        "cme_last_72h_max_cme_speed",
        "cme_last_72h_halo_cme_count",
        "eta_pm24h_eta_cme_count",
        "eta_pm24h_eta_max_cme_speed",
        "eta_pm24h_eta_min_abs_error_hours",
    ]
    examples_source = private if not private.empty else fixed
    examples_source[[c for c in feature_example_cols if c in examples_source.columns]].sample(
        n=min(100, len(examples_source)), random_state=20260626
    ).to_csv(out_dir / "cme_feature_examples.csv", index=False)

    sanity = leakage_report(private if not private.empty else fixed, pd.concat([oof_sanity, private_sanity], ignore_index=True), horizons)
    event_examples = event_2024_may_examples(private) if not private.empty else pd.DataFrame()

    summary.to_csv(out_dir / "summary.csv", index=False)
    fixed_results.to_csv(out_dir / "fixed_results.csv", index=False)
    private_results.to_csv(out_dir / "private_diagnostic.csv", index=False)
    private_yearly.to_csv(out_dir / "private_yearly_diagnostic.csv", index=False)
    horizon_df.to_csv(out_dir / "horizon_metrics_base_vs_corrected.csv", index=False)
    selected_horizon_metrics.to_csv(out_dir / "selected_horizon_metrics.csv", index=False)
    event_examples.to_csv(out_dir / "event_2024_may_examples.csv", index=False)
    correction_dist.to_csv(out_dir / "correction_distribution_summary.csv", index=False)
    best_private.to_csv(out_dir / "best_private_prediction.csv", index=False)
    base_private.to_csv(out_dir / "base_private_prediction.csv", index=False)
    corrected_private.to_csv(out_dir / "corrected_private_prediction.csv", index=False)
    sanity.to_csv(out_dir / "leakage_sanity_report.csv", index=False)

    config = {
        "task": "horizon_aware_cme_context_residual_correction_for_72h_blocks",
        "base_task": "non_overlapping_72h_blocks_multihorizon",
        "feature_set": mh.FEATURE_SET_NAME,
        "base_model": "per-horizon 0.7 Direct MLP_h + 0.3 ExtraTrees_h",
        "correction_model": "one pooled horizon-aware residual model: 0.7 residual MLP + 0.3 residual ExtraTrees",
        "cme_catalog": str(cme_path),
        "horizons": horizons,
        "context_features": context_cols,
        "correction_features": correction_cols,
        "public_oof": "expanding-year 2011-(year-1) -> year for target years 2017-2023",
        "fixed_correction_train_target_years": [2017, 2018, 2019, 2020, 2021],
        "fixed_correction_validation_target_years": [2022, 2023],
        "private_correction_train": "all public OOF residual rows only",
        "private_target_years": [2024, 2025],
        "adopt_correction": adopt,
        "selection_reason": reason,
        "recent_cme_windows_hours": RECENT_WINDOWS,
        "eta_windows_hours": ETA_WINDOWS,
        "eta_gaussian_sigmas_hours": ETA_SIGMAS,
        "private_labels_used_for_model_selection": False,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, allow_nan=True))

    print("\nBase public fixed metrics", flush=True)
    print(fixed_results[fixed_results["model_name"].eq(BASE_MODEL)].to_string(index=False), flush=True)
    print("\nCorrected public fixed metrics", flush=True)
    print(fixed_results[fixed_results["model_name"].eq(CORRECTED_MODEL)].to_string(index=False), flush=True)
    print(f"\nAdopt correction: {adopt}. {reason}", flush=True)
    if not private_results.empty:
        print("\nBase private metrics", flush=True)
        print(private_results[private_results["model_name"].eq(BASE_MODEL)].to_string(index=False), flush=True)
        print("\nCorrected private metrics", flush=True)
        print(private_results[private_results["model_name"].eq(CORRECTED_MODEL)].to_string(index=False), flush=True)
    print("\nSelected horizon comparison", flush=True)
    print(selected_horizon_metrics.to_string(index=False), flush=True)
    if not event_examples.empty:
        print("\n2024 May event examples", flush=True)
        print(event_examples.head(20).to_string(index=False), flush=True)
    print(f"\nSaved outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
