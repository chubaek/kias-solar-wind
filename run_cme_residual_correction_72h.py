"""Simple CME residual correction experiment for 72h speed forecasts.

The base forecast is the current official best:

    Speed(t + 72h) = 0.7 * Direct MLP + 0.3 * ExtraTrees

using current tabular features plus representative_mrmr_ch.  The correction
forecast uses causal recent-window CME features only:

    residual(t) = true Speed(t + 72h) - base_pred(t)
    residual_pred = 0.7 * residual_MLP + 0.3 * residual_ExtraTrees
    final_pred = base_pred + residual_pred
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import run_ch_feature_addition_72h as chrun
import train_tabular_models_72h as tab


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs" / "cme_residual_correction_72h"
BASE_FEATURE_SET = "current_plus_representative_mrmr_ch"
BASE_MODEL = "base_model_only"
CORRECTED_MODEL = "base_model_plus_cme_residual_correction"
ENSEMBLE_WEIGHTS = {"direct_mlp": 0.7, "extratrees": 0.3}

DEFAULT_CME_PATHS = [
    HERE / "data" / "cme_catalog.csv",
    HERE / "data" / "cme_catalog_2011_2025.csv",
    HERE / "data" / "donki_cme_catalog.csv",
    HERE / "data" / "lasco_cme_catalog.csv",
    HERE / "cme_catalog.csv",
]

WINDOWS = [
    ("last_24h", 24.0),
    ("last_48h", 48.0),
    ("last_72h", 72.0),
    ("last_96h", 96.0),
]

CME_FEATURE_BASES = [
    "cme_count",
    "fast_cme_count_speed_gt_800",
    "very_fast_cme_count_speed_gt_1200",
    "halo_cme_count",
    "max_cme_speed",
    "mean_cme_speed",
    "sum_cme_speed",
    "max_cme_width",
    "mean_cme_width",
    "sum_speed_width",
    "time_since_last_cme_hours",
    "time_since_last_fast_cme_hours",
    "time_since_last_halo_cme_hours",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cme-catalog", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--skip-private", action="store_true")
    return parser.parse_args()


def cc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2 or np.std(y_true) <= 1e-8 or np.std(y_pred) <= 1e-8:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def prediction_metrics(y_true: np.ndarray, pred: np.ndarray, persistence: np.ndarray | None = None) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(pred)
    y = y_true[mask]
    p = pred[mask]
    if len(y) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "cc": np.nan, "mae_skill_vs_27day": np.nan}

    skill = np.nan
    if persistence is not None:
        per = persistence[mask]
        p_mask = np.isfinite(per)
        if p_mask.any():
            p_mae = float(mean_absolute_error(y[p_mask], per[p_mask]))
            model_mae = float(mean_absolute_error(y[p_mask], p[p_mask]))
            skill = float(1.0 - model_mae / p_mae) if p_mae > 0 else np.nan

    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(math.sqrt(mean_squared_error(y, p))),
        "cc": cc_score(y, p),
        "mae_skill_vs_27day": skill,
    }


def residual_metrics(residual_true: np.ndarray, residual_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(residual_true) & np.isfinite(residual_pred)
    if not mask.any():
        return {"residual_mae": np.nan, "residual_rmse": np.nan}
    y = residual_true[mask]
    p = residual_pred[mask]
    return {
        "residual_mae": float(mean_absolute_error(y, p)),
        "residual_rmse": float(math.sqrt(mean_squared_error(y, p))),
    }


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

    cme = cme[cme["cme_time"].notna()].sort_values("cme_time").reset_index(drop=True)
    if cme.empty:
        raise RuntimeError(f"CME catalog has no parseable event times: {path}")
    return cme


def build_base_tables() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    ch = chrun.load_ch()
    table_public = tab.build_feature_table(tab.FULL_CSV)
    table_all = tab.build_feature_table(
        tab.FULL_CSV,
        min_target_year=2011,
        max_target_year=2025,
        require_finite_target=False,
        require_finite_persistence=False,
    )
    tables_public_ch, features_by_set, _ = chrun.feature_sets(table_public, ch)
    tables_all_ch, _, _ = chrun.feature_sets(table_all, ch)
    return tables_public_ch[BASE_FEATURE_SET], tables_all_ch[BASE_FEATURE_SET], features_by_set[BASE_FEATURE_SET]


def build_cme_features(origin_times: pd.Series, cme: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    origins = pd.to_datetime(origin_times).reset_index(drop=True)
    event_times = pd.to_datetime(cme["cme_time"]).to_numpy(dtype="datetime64[ns]")
    speed = cme["cme_speed"].to_numpy(dtype=float)
    width = cme["cme_width"].to_numpy(dtype=float)
    halo = cme["is_halo"].to_numpy(dtype=bool)
    speed_width = np.where(np.isfinite(speed) & np.isfinite(width), speed * width, np.nan)
    rows: list[dict[str, float]] = []
    sanity_rows: list[dict[str, Any]] = []

    for i, origin in enumerate(origins):
        row: dict[str, float] = {}
        origin64 = np.datetime64(origin.to_datetime64())
        hi = int(np.searchsorted(event_times, origin64, side="right"))
        for label, hours_back in WINDOWS:
            start_time = origin - pd.Timedelta(hours=hours_back)
            lo = int(np.searchsorted(event_times, np.datetime64(start_time.to_datetime64()), side="right"))
            idx = np.arange(lo, hi)
            sp = speed[idx]
            wd = width[idx]
            sw = speed_width[idx]
            is_halo = halo[idx]
            finite_speed = sp[np.isfinite(sp)]
            finite_width = wd[np.isfinite(wd)]
            finite_sw = sw[np.isfinite(sw)]
            event_count = len(idx)

            row[f"cme_{label}_cme_count"] = float(event_count)
            row[f"cme_{label}_fast_cme_count_speed_gt_800"] = float(np.sum(sp > 800.0))
            row[f"cme_{label}_very_fast_cme_count_speed_gt_1200"] = float(np.sum(sp > 1200.0))
            row[f"cme_{label}_halo_cme_count"] = float(np.sum(is_halo))
            row[f"cme_{label}_max_cme_speed"] = float(np.max(finite_speed)) if len(finite_speed) else 0.0
            row[f"cme_{label}_mean_cme_speed"] = float(np.mean(finite_speed)) if len(finite_speed) else 0.0
            row[f"cme_{label}_sum_cme_speed"] = float(np.sum(finite_speed)) if len(finite_speed) else 0.0
            row[f"cme_{label}_max_cme_width"] = float(np.max(finite_width)) if len(finite_width) else 0.0
            row[f"cme_{label}_mean_cme_width"] = float(np.mean(finite_width)) if len(finite_width) else 0.0
            row[f"cme_{label}_sum_speed_width"] = float(np.sum(finite_sw)) if len(finite_sw) else 0.0

            if event_count:
                delta_hours = (origin64 - event_times[idx]).astype("timedelta64[s]").astype(float) / 3600.0
                row[f"cme_{label}_time_since_last_cme_hours"] = float(np.min(delta_hours))
                fast_hours = delta_hours[sp > 800.0]
                halo_hours = delta_hours[is_halo]
                row[f"cme_{label}_time_since_last_fast_cme_hours"] = float(np.min(fast_hours)) if len(fast_hours) else 9999.0
                row[f"cme_{label}_time_since_last_halo_cme_hours"] = float(np.min(halo_hours)) if len(halo_hours) else 9999.0
            else:
                row[f"cme_{label}_time_since_last_cme_hours"] = 9999.0
                row[f"cme_{label}_time_since_last_fast_cme_hours"] = 9999.0
                row[f"cme_{label}_time_since_last_halo_cme_hours"] = 9999.0

        rows.append(row)
        if i % max(1, len(origins) // 200) == 0:
            sanity_rows.append(
                {
                    "origin_time": origin,
                    "latest_cme_time_le_origin": pd.Timestamp(event_times[hi - 1]) if hi > 0 else pd.NaT,
                    "cme_count_last_72h": row["cme_last_72h_cme_count"],
                    "max_cme_speed_last_72h": row["cme_last_72h_max_cme_speed"],
                    "halo_cme_count_last_72h": row["cme_last_72h_halo_cme_count"],
                    "all_cme_times_le_origin": True,
                }
            )

    return pd.DataFrame(rows, index=origin_times.index), pd.DataFrame(sanity_rows)


def add_cme_features(table: pd.DataFrame, cme: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    features, sanity = build_cme_features(table["origin_datetime"], cme)
    out = pd.concat([table.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    return out, list(features.columns), sanity


def selected_base_models() -> dict[str, Any]:
    return chrun.model_configs()


def fit_predict_ensemble(models: dict[str, Any], X_train: pd.DataFrame, y_train: np.ndarray, X_eval: pd.DataFrame) -> dict[str, np.ndarray]:
    mlp_pred = chrun.fit_predict(models["direct_mlp"], X_train, y_train, X_eval)
    extratrees_pred = chrun.fit_predict(models["extratrees"], X_train, y_train, X_eval)
    ensemble = ENSEMBLE_WEIGHTS["direct_mlp"] * mlp_pred + ENSEMBLE_WEIGHTS["extratrees"] * extratrees_pred
    return {"direct_mlp_pred": mlp_pred, "extratrees_pred": extratrees_pred, "ensemble_pred": ensemble}


def build_oof_base_predictions(table: pd.DataFrame, base_features: list[str], out_dir: Path) -> pd.DataFrame:
    path = out_dir / "oof_base_predictions.csv"
    if path.exists():
        existing = pd.read_csv(path, parse_dates=["origin_datetime", "target_datetime"])
        required = {"origin_datetime", "target_datetime", "base_pred", "residual"}
        if required.issubset(existing.columns):
            print(f"Using cached OOF base predictions: {path}", flush=True)
            return existing

    models = selected_base_models()
    rows: list[pd.DataFrame] = []
    for year in range(2017, 2024):
        train_mask = (table["target_year"] >= 2011) & (table["target_year"] <= year - 1) & table["target_speed"].notna()
        val_mask = table["target_year"].eq(year) & table["target_speed"].notna()
        train_idx = np.flatnonzero(train_mask.to_numpy())
        val_idx = np.flatnonzero(val_mask.to_numpy())
        print(f"OOF base train 2011-{year - 1} -> {year}: train={len(train_idx)} val={len(val_idx)}", flush=True)
        preds = fit_predict_ensemble(
            models,
            table.iloc[train_idx][base_features],
            table.iloc[train_idx]["target_speed"].to_numpy(dtype=np.float32),
            table.iloc[val_idx][base_features],
        )
        frame = table.iloc[val_idx][["origin_datetime", "target_datetime", "target_speed", "target_year", "persistence_27day_target_aligned"]].copy()
        frame["oof_year"] = year
        frame["direct_mlp_pred"] = preds["direct_mlp_pred"]
        frame["extratrees_pred"] = preds["extratrees_pred"]
        frame["base_pred"] = preds["ensemble_pred"]
        frame["residual"] = frame["target_speed"].to_numpy(dtype=float) - frame["base_pred"].to_numpy(dtype=float)
        rows.append(frame)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def final_base_private_predictions(table: pd.DataFrame, base_features: list[str]) -> pd.DataFrame:
    models = selected_base_models()
    public_idx = np.flatnonzero(((table["target_year"] >= 2011) & (table["target_year"] <= 2023) & table["target_speed"].notna()).to_numpy())
    private_idx = np.flatnonzero(((table["target_year"] >= 2024) & (table["target_year"] <= 2025)).to_numpy())
    preds = fit_predict_ensemble(
        models,
        table.iloc[public_idx][base_features],
        table.iloc[public_idx]["target_speed"].to_numpy(dtype=np.float32),
        table.iloc[private_idx][base_features],
    )
    frame = table.iloc[private_idx][["origin_datetime", "target_datetime", "target_speed", "target_year", "persistence_27day_target_aligned"]].copy()
    frame["_table_row_id"] = private_idx
    frame["direct_mlp_pred"] = preds["direct_mlp_pred"]
    frame["extratrees_pred"] = preds["extratrees_pred"]
    frame["base_pred"] = preds["ensemble_pred"]
    frame["residual"] = frame["target_speed"].to_numpy(dtype=float) - frame["base_pred"].to_numpy(dtype=float)
    return frame.reset_index(drop=True)


def attach_oof_base(table: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    keys = ["origin_datetime", "target_datetime"]
    cols = keys + ["base_pred", "direct_mlp_pred", "extratrees_pred", "residual"]
    return table.merge(oof[cols], on=keys, how="left")


def evaluate_rows(frame: pd.DataFrame, pred_col: str, model_name: str, scheme: str, fold: str) -> dict[str, Any]:
    y = frame["target_speed"].to_numpy(dtype=float)
    pred = frame[pred_col].to_numpy(dtype=float)
    persistence = frame["persistence_27day_target_aligned"].to_numpy(dtype=float)
    row = {"model_name": model_name, "validation_scheme": scheme, "fold": fold, **prediction_metrics(y, pred, persistence)}
    if "residual_pred" in frame.columns:
        row.update(residual_metrics(frame["residual"].to_numpy(dtype=float), frame["residual_pred"].to_numpy(dtype=float)))
    else:
        row.update({"residual_mae": np.nan, "residual_rmse": np.nan})
    return row


def fit_predict_residual_correction(train: pd.DataFrame, eval_frame: pd.DataFrame, cme_cols: list[str]) -> dict[str, np.ndarray]:
    models = selected_base_models()
    preds = fit_predict_ensemble(
        models,
        train[cme_cols],
        train["residual"].to_numpy(dtype=np.float32),
        eval_frame[cme_cols],
    )
    return {
        "residual_mlp_pred": preds["direct_mlp_pred"],
        "residual_extratrees_pred": preds["extratrees_pred"],
        "residual_pred": preds["ensemble_pred"],
    }


def score_base_and_correction(train: pd.DataFrame, val: pd.DataFrame, cme_cols: list[str], scheme: str, fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [evaluate_rows(val.assign(final_pred=val["base_pred"]), "final_pred", BASE_MODEL, scheme, fold)]
    corrected = val.copy()
    residual_preds = fit_predict_residual_correction(train, val, cme_cols)
    corrected["residual_mlp_pred"] = residual_preds["residual_mlp_pred"]
    corrected["residual_extratrees_pred"] = residual_preds["residual_extratrees_pred"]
    corrected["residual_pred"] = residual_preds["residual_pred"]
    corrected["final_pred"] = corrected["base_pred"] + corrected["residual_pred"]
    rows.append(evaluate_rows(corrected, "final_pred", CORRECTED_MODEL, scheme, fold))
    return pd.DataFrame(rows), corrected


def evaluate_fixed(oof_table: pd.DataFrame, cme_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = oof_table[(oof_table["target_year"] >= 2017) & (oof_table["target_year"] <= 2021) & oof_table["residual"].notna()].copy()
    val = oof_table[oof_table["target_year"].isin([2022, 2023]) & oof_table["residual"].notna()].copy()
    return score_base_and_correction(train, val, cme_cols, "fixed_2022_2023", "fixed")


def evaluate_private(private_base: pd.DataFrame, oof_table: pd.DataFrame, cme_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = oof_table[oof_table["residual"].notna()].copy()
    rows, corrected = score_base_and_correction(train, private_base, cme_cols, "private_diagnostic", "2024_2025")

    yearly_rows: list[dict[str, Any]] = []
    for source_name, source in [(BASE_MODEL, private_base.assign(final_pred=private_base["base_pred"], residual_pred=0.0)), (CORRECTED_MODEL, corrected)]:
        for year_label, years in [("2024", [2024]), ("2025", [2025]), ("2024_2025", [2024, 2025])]:
            part = source[source["target_year"].isin(years)].copy()
            yearly_rows.append({"year": year_label, **evaluate_rows(part, "final_pred", source_name, "private_diagnostic", year_label)})

    predictions = pd.concat(
        [
            private_base.assign(final_pred=private_base["base_pred"], residual_pred=0.0, model_name=BASE_MODEL),
            corrected.assign(model_name=CORRECTED_MODEL),
        ],
        ignore_index=True,
    )
    return rows, pd.DataFrame(yearly_rows), predictions, corrected


def choose_model(fixed: pd.DataFrame) -> tuple[str, bool, str]:
    base = fixed[fixed["model_name"].eq(BASE_MODEL)].iloc[0]
    corrected = fixed[fixed["model_name"].eq(CORRECTED_MODEL)].iloc[0]
    if corrected["cc"] > base["cc"]:
        return CORRECTED_MODEL, True, "CME residual correction improved public fixed CC"
    return BASE_MODEL, False, "CME residual correction did not improve public fixed CC"


def aggregate_summary(fixed: pd.DataFrame, private_yearly: pd.DataFrame) -> pd.DataFrame:
    fixed_s = fixed[["model_name", "mae", "rmse", "cc", "residual_mae", "residual_rmse"]].rename(
        columns={
            "mae": "fixed_mae",
            "rmse": "fixed_rmse",
            "cc": "fixed_cc",
            "residual_mae": "fixed_residual_mae",
            "residual_rmse": "fixed_residual_rmse",
        }
    )
    if private_yearly.empty:
        priv_s = pd.DataFrame(columns=["model_name", "private_2024_mae", "private_2025_mae", "private_2024_2025_mae"])
    else:
        pivot = private_yearly.pivot_table(index="model_name", columns="year", values=["mae", "rmse", "cc"], aggfunc="first")
        pivot.columns = [f"private_{year}_{metric}" for metric, year in pivot.columns]
        priv_s = pivot.reset_index()
    return fixed_s.merge(priv_s, on="model_name", how="left")


def best_private_prediction(predictions: pd.DataFrame, selected_model: str) -> pd.DataFrame:
    chosen = predictions[predictions["model_name"].eq(selected_model)].copy()
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(chosen["target_datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_speed": chosen["final_pred"],
        }
    )


def may_event_examples(private_predictions: pd.DataFrame) -> pd.DataFrame:
    frame = private_predictions[private_predictions["model_name"].eq(CORRECTED_MODEL)].copy()
    frame = frame[(frame["target_datetime"] >= "2024-05-10") & (frame["target_datetime"] <= "2024-05-14")].copy()
    frame["origin_time"] = frame["origin_datetime"]
    frame["target_time"] = frame["target_datetime"]
    frame["observed"] = frame["target_speed"]
    frame["residual_true"] = frame["target_speed"] - frame["base_pred"]
    out_cols = [
        "origin_time",
        "target_time",
        "observed",
        "base_pred",
        "residual_true",
        "residual_pred",
        "final_pred",
        "cme_last_72h_cme_count",
        "cme_last_72h_max_cme_speed",
        "cme_last_72h_halo_cme_count",
    ]
    renamed = {
        "cme_last_72h_cme_count": "cme_count_last_72h",
        "cme_last_72h_max_cme_speed": "max_cme_speed_last_72h",
        "cme_last_72h_halo_cme_count": "halo_cme_count_last_72h",
    }
    return frame.sort_values("observed", ascending=False)[out_cols].rename(columns=renamed).head(20)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir if args.output_dir.is_absolute() else HERE / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cme_path = resolve_cme_catalog(args.cme_catalog)
    cme = load_cme_catalog(cme_path)
    print(f"Loaded CME catalog {cme_path} rows={len(cme)} {cme['cme_time'].min()} to {cme['cme_time'].max()}", flush=True)

    base_public, base_all, base_features = build_base_tables()
    public_cme, cme_cols, sanity_public = add_cme_features(base_public, cme)
    all_cme, _, sanity_all = add_cme_features(base_all, cme)

    oof = build_oof_base_predictions(public_cme, base_features, out_dir)
    oof_table = attach_oof_base(public_cme, oof)
    fixed, _ = evaluate_fixed(oof_table, cme_cols)
    selected, adopted, reason = choose_model(fixed)

    private = pd.DataFrame()
    private_yearly = pd.DataFrame()
    private_predictions = pd.DataFrame()
    private_corrected = pd.DataFrame()
    if not args.skip_private:
        private_base = final_base_private_predictions(all_cme, base_features)
        private_row_id = private_base["_table_row_id"].to_numpy(dtype=int)
        private_base = pd.concat(
            [private_base.reset_index(drop=True), all_cme.iloc[private_row_id][cme_cols].reset_index(drop=True)],
            axis=1,
        )
        private, private_yearly, private_predictions, private_corrected = evaluate_private(private_base, oof_table, cme_cols)

    summary = aggregate_summary(fixed, private_yearly)
    best_private = (
        best_private_prediction(private_predictions, selected)
        if not private_predictions.empty
        else pd.DataFrame(columns=["datetime", "predicted_speed"])
    )

    fixed.to_csv(out_dir / "fixed_results.csv", index=False)
    private.to_csv(out_dir / "private_diagnostic.csv", index=False)
    private_yearly.to_csv(out_dir / "private_yearly_diagnostic.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    public_cme[["origin_datetime", "target_datetime", *cme_cols]].sample(
        n=min(50, len(public_cme)), random_state=20260626
    ).to_csv(out_dir / "cme_feature_examples.csv", index=False)
    best_private.to_csv(out_dir / "best_private_prediction.csv", index=False)
    if not private_predictions.empty:
        private_predictions[
            [
                "model_name",
                "origin_datetime",
                "target_datetime",
                "target_speed",
                "base_pred",
                "residual_pred",
                "final_pred",
            ]
        ].to_csv(out_dir / "private_predictions.csv", index=False)
        may_event_examples(private_predictions).to_csv(out_dir / "may_2024_event_examples.csv", index=False)

    config = {
        "cme_catalog": str(cme_path),
        "base_feature_set": BASE_FEATURE_SET,
        "base_model": "0.7 Direct MLP + 0.3 ExtraTrees",
        "correction_model": "0.7 residual Direct MLP + 0.3 residual ExtraTrees",
        "residual_target": "Speed(t+72h) - out_of_fold_base_pred(t)",
        "cme_features": CME_FEATURE_BASES,
        "cme_windows": [label for label, _ in WINDOWS],
        "selected_model": selected,
        "adopt_residual_correction": adopted,
        "selection_reason": reason,
        "causality": [
            "CME events are included only when cme_time <= origin time t",
            "Public residual labels use out-of-fold base predictions",
            "Private diagnostics train residual correction only on public OOF residuals",
            "Private labels are not used for model selection",
        ],
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, allow_nan=True))

    print("\nBase fixed metrics", flush=True)
    print(fixed[fixed["model_name"].eq(BASE_MODEL)].to_string(index=False), flush=True)
    print("\nCorrected fixed metrics", flush=True)
    print(fixed[fixed["model_name"].eq(CORRECTED_MODEL)].to_string(index=False), flush=True)
    if not private_yearly.empty:
        print("\nBase private yearly metrics", flush=True)
        print(private_yearly[private_yearly["model_name"].eq(BASE_MODEL)].to_string(index=False), flush=True)
        print("\nCorrected private yearly metrics", flush=True)
        print(private_yearly[private_yearly["model_name"].eq(CORRECTED_MODEL)].to_string(index=False), flush=True)
    if not private_predictions.empty:
        examples = may_event_examples(private_predictions)
        print("\n2024 May event examples", flush=True)
        print(examples.to_string(index=False), flush=True)
    print(f"\nSelection: {selected}", flush=True)
    print(f"Adopt residual correction: {adopted}. {reason}", flush=True)
    print(f"Saved outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
