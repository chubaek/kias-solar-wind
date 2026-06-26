"""Causal ballistic-backmapped HMI flux feature experiment for 72h speed.

For each forecast origin t, this uses only solar-wind speed observed at or
before t to estimate when the currently observed plasma was launched from the
Sun.  Daily HMI magnetogram flux rows are then matched backward to that source
time and earlier lookbacks.  The resulting coarse daily flux-change features
are evaluated with the current selected Direct MLP / ExtraTrees family.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_ch_feature_addition_72h as chrun
import train_tabular_models_72h as tab


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs" / "backmapped_flux_features_72h"
MAG_CSV = HERE / "data" / "magnetograms" / "magnetogram_window_features_daily_2011_2025.csv"
AU_KM = 149_597_870.7

BASELINE = "baseline_current_best"
DAILY_ALL = "baseline_plus_backmapped_flux_daily"
DAILY_24H = "baseline_plus_backmapped_flux_daily_24h_only"
ENSEMBLE_MODEL = "ensemble_0p7_mlp_0p3_extratrees"

SPEED_DEFS = {
    "speed_t": "speed_current",
    "speed_roll3h": "Speed_km_s_roll_mean_3h",
    "speed_roll6h": "Speed_km_s_roll_mean_6h",
    "speed_roll12h": "Speed_km_s_roll_mean_12h",
}
LOOKBACK_HOURS = [10, 24, 48]
FLUX_CANDIDATES = [
    "full_disk_unsigned_flux_proxy",
    "central_disk_unsigned_flux_proxy",
    "W_lon7p5_lat15_sum_abs_B",
    "W_lon30_lat15_sum_abs_B",
    "W_lon30_lat30_sum_abs_B",
    "W_lon60_lat60_sum_abs_B",
    "high_flux_roi_unsigned_flux_proxy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--skip-private", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--sanity-rows", type=int, default=200)
    return parser.parse_args()


def load_daily_magnetogram_features(path: Path = MAG_CSV) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise RuntimeError(f"Missing daily HMI flux feature table: {path}")
    mag = pd.read_csv(path, parse_dates=["magnetogram_time"]).sort_values("magnetogram_time")
    mag = mag.drop_duplicates("magnetogram_time", keep="last").reset_index(drop=True)
    flux_cols = [col for col in FLUX_CANDIDATES if col in mag.columns]
    if not flux_cols:
        raise RuntimeError(f"No requested flux columns found in {path}")
    return mag, flux_cols


def add_origin_speed_roll3(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    if "Speed_km_s_roll_mean_3h" in out.columns:
        return out
    raw = pd.read_csv(tab.FULL_CSV, parse_dates=["datetime"]).sort_values("datetime")
    raw["Speed_km_s_roll_mean_3h"] = raw[tab.TARGET].rolling(3, min_periods=1).mean()
    out = out.merge(
        raw[["datetime", "Speed_km_s_roll_mean_3h"]].rename(columns={"datetime": "origin_datetime"}),
        on="origin_datetime",
        how="left",
    )
    return out


def source_times_for_speed(table: pd.DataFrame, speed_col: str) -> tuple[pd.Series, np.ndarray]:
    origin = pd.to_datetime(table["origin_datetime"])
    speed = table[speed_col].to_numpy(dtype=float)
    valid = np.isfinite(speed) & (speed > 250.0) & (speed <= 1200.0)
    travel_hours = np.full(len(table), np.nan, dtype=float)
    travel_hours[valid] = AU_KM / speed[valid] / 3600.0
    valid &= travel_hours <= 8.0 * 24.0

    source = pd.Series(pd.NaT, index=table.index, dtype="datetime64[ns]")
    source_values = origin - pd.to_timedelta(travel_hours, unit="h")
    valid &= source_values < origin
    source.loc[valid] = source_values.loc[valid]
    return source, travel_hours


def backward_match_flux(
    requested: pd.Series,
    mag: pd.DataFrame,
    flux_col: str,
) -> tuple[np.ndarray, pd.Series]:
    result = np.full(len(requested), np.nan, dtype=float)
    matched_time = pd.Series(pd.NaT, index=requested.index, dtype="datetime64[ns]")
    valid = requested.notna()
    if not valid.any():
        return result, matched_time

    left = pd.DataFrame(
        {
            "_row_id": np.flatnonzero(valid.to_numpy()),
            "requested_flux_time": pd.to_datetime(requested[valid]).to_numpy(),
        }
    ).sort_values("requested_flux_time")
    right = mag[["magnetogram_time", flux_col]].sort_values("magnetogram_time")
    merged = pd.merge_asof(
        left,
        right,
        left_on="requested_flux_time",
        right_on="magnetogram_time",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("_row_id")

    row_id = merged["_row_id"].to_numpy(dtype=int)
    result[row_id] = merged[flux_col].to_numpy(dtype=float)
    matched_time.iloc[row_id] = pd.to_datetime(merged["magnetogram_time"]).to_numpy()
    return result, matched_time


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(len(num), np.nan, dtype=float)
    valid = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-12)
    out[valid] = num[valid] / den[valid]
    return out


def add_backmapped_flux_features(
    table: pd.DataFrame,
    mag: pd.DataFrame,
    flux_cols: list[str],
    *,
    sanity_rows: int,
    random_state: int = 20260626,
) -> tuple[pd.DataFrame, list[str], list[str], pd.DataFrame, pd.DataFrame]:
    out = add_origin_speed_roll3(table)
    origin = pd.to_datetime(out["origin_datetime"])
    all_features: list[str] = []
    daily_24h_features: list[str] = []
    source_info: dict[str, tuple[pd.Series, np.ndarray]] = {}
    matched_cache: dict[tuple[str, str, int], tuple[np.ndarray, pd.Series, pd.Series]] = {}
    new_columns: dict[str, np.ndarray] = {}

    for speed_name, speed_col in SPEED_DEFS.items():
        source_time, travel_hours = source_times_for_speed(out, speed_col)
        source_info[speed_name] = (source_time, travel_hours)
        new_columns[f"backmap_{speed_name}_travel_time_hours"] = travel_hours

        for flux_col in flux_cols:
            flux_safe = flux_col.replace("_sum_abs_B", "").replace("_unsigned_flux_proxy", "")
            prefix = f"bmf_daily__{speed_name}__{flux_safe}"
            requested_times: dict[int, pd.Series] = {0: source_time}
            for lookback in LOOKBACK_HOURS:
                requested_times[lookback] = source_time - pd.to_timedelta(lookback, unit="h")

            flux_values: dict[int, np.ndarray] = {}
            matched_times: dict[int, pd.Series] = {}
            for lookback, requested in requested_times.items():
                valid_request = requested.where(requested <= origin)
                values, matched = backward_match_flux(valid_request, mag, flux_col)
                flux_values[lookback] = values
                matched_times[lookback] = matched
                matched_cache[(speed_name, flux_col, lookback)] = (values, valid_request, matched)

            at_col = f"{prefix}__flux_at_source_time"
            new_columns[at_col] = flux_values[0]
            all_features.append(at_col)

            for lookback in LOOKBACK_HOURS:
                lag_col = f"{prefix}__flux_source_minus_{lookback}h"
                d_col = f"{prefix}__dflux_{lookback}h"
                rate_col = f"{prefix}__dflux_rate_{lookback}h"
                ratio_col = f"{prefix}__flux_ratio_{lookback}h"
                d_values = flux_values[0] - flux_values[lookback]
                new_columns[lag_col] = flux_values[lookback]
                new_columns[d_col] = d_values
                new_columns[rate_col] = d_values / float(lookback)
                new_columns[ratio_col] = safe_ratio(flux_values[0], flux_values[lookback])
                all_features.extend([lag_col, d_col, rate_col, ratio_col])
                if lookback in {24, 48}:
                    daily_24h_features.extend([d_col, rate_col, ratio_col])

    out = pd.concat([out, pd.DataFrame(new_columns, index=out.index)], axis=1).copy()
    sample_n = min(sanity_rows, len(out))
    sample_idx = sorted(out.sample(n=sample_n, random_state=random_state).index.tolist()) if sample_n else []
    sanity = build_timestamp_sanity(out, source_info, matched_cache, flux_cols, sample_idx)
    examples = build_source_examples(out, source_info, matched_cache, flux_cols, sample_idx[:20])
    return out, all_features, daily_24h_features, sanity, examples


def build_timestamp_sanity(
    table: pd.DataFrame,
    source_info: dict[str, tuple[pd.Series, np.ndarray]],
    matched_cache: dict[tuple[str, str, int], tuple[np.ndarray, pd.Series, pd.Series]],
    flux_cols: list[str],
    sample_idx: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    origin = pd.to_datetime(table["origin_datetime"])
    for idx in sample_idx:
        for speed_name, (source_time, travel_hours) in source_info.items():
            for flux_col in flux_cols:
                for lookback in [0, *LOOKBACK_HOURS]:
                    values, requested, matched = matched_cache[(speed_name, flux_col, lookback)]
                    requested_time = requested.iloc[idx]
                    matched_time = matched.iloc[idx]
                    rows.append(
                        {
                            "_row_id": int(idx),
                            "origin_time": origin.iloc[idx],
                            "target_time": table.iloc[idx]["target_datetime"],
                            "speed_definition": speed_name,
                            "source_time_est": source_time.iloc[idx],
                            "travel_time_hours": travel_hours[idx],
                            "flux_column": flux_col,
                            "lookback_hours": lookback,
                            "requested_flux_time": requested_time,
                            "matched_hmi_time": matched_time,
                            "matched_hmi_time_le_requested_flux_time": bool(pd.notna(matched_time) and pd.notna(requested_time) and matched_time <= requested_time),
                            "requested_flux_time_le_origin_time": bool(pd.notna(requested_time) and requested_time <= origin.iloc[idx]),
                            "source_time_est_le_origin_time": bool(pd.notna(source_time.iloc[idx]) and source_time.iloc[idx] <= origin.iloc[idx]),
                            "flux_value": values[idx],
                        }
                    )
    return pd.DataFrame(rows)


def build_source_examples(
    table: pd.DataFrame,
    source_info: dict[str, tuple[pd.Series, np.ndarray]],
    matched_cache: dict[tuple[str, str, int], tuple[np.ndarray, pd.Series, pd.Series]],
    flux_cols: list[str],
    sample_idx: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    flux_col = "W_lon30_lat30_sum_abs_B" if "W_lon30_lat30_sum_abs_B" in flux_cols else flux_cols[0]
    speed_name = "speed_t"
    source_time, travel_hours = source_info[speed_name]
    at_values, at_requested, at_matched = matched_cache[(speed_name, flux_col, 0)]
    ten_values, _, _ = matched_cache[(speed_name, flux_col, 10)]
    day_values, _, _ = matched_cache[(speed_name, flux_col, 24)]
    for idx in sample_idx:
        rows.append(
            {
                "origin_time": table.iloc[idx]["origin_datetime"],
                "target_time": table.iloc[idx]["target_datetime"],
                "Speed(t)": table.iloc[idx]["speed_current"],
                "estimated_source_time": source_time.iloc[idx],
                "travel_time_hours": travel_hours[idx],
                "requested_flux_time": at_requested.iloc[idx],
                "matched_HMI_time": at_matched.iloc[idx],
                "matched_HMI_time_le_requested_flux_time": bool(pd.notna(at_matched.iloc[idx]) and pd.notna(at_requested.iloc[idx]) and at_matched.iloc[idx] <= at_requested.iloc[idx]),
                "requested_flux_time_le_origin_time": bool(pd.notna(at_requested.iloc[idx]) and at_requested.iloc[idx] <= pd.Timestamp(table.iloc[idx]["origin_datetime"])),
                "flux_column": flux_col,
                "flux_at_source_time": at_values[idx],
                "dflux_10h": at_values[idx] - ten_values[idx] if np.isfinite(at_values[idx]) and np.isfinite(ten_values[idx]) else np.nan,
                "dflux_24h": at_values[idx] - day_values[idx] if np.isfinite(at_values[idx]) and np.isfinite(day_values[idx]) else np.nan,
            }
        )
    return pd.DataFrame(rows)


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
    return (
        tables_public_ch["current_plus_representative_mrmr_ch"],
        tables_all_ch["current_plus_representative_mrmr_ch"],
        features_by_set["current_plus_representative_mrmr_ch"],
    )


def evaluate_feature_sets(
    tables_public: dict[str, pd.DataFrame],
    tables_all: dict[str, pd.DataFrame],
    features_by_set: dict[str, list[str]],
    *,
    skip_cv: bool,
    skip_private: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    models = chrun.model_configs()
    fixed_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    private_predictions: list[pd.DataFrame] = []

    for name, table in tables_public.items():
        print(f"\nFixed evaluation: {name}", flush=True)
        rows, _, _ = chrun.evaluate_split(table, features_by_set[name], tab.fixed_split(), name, models)
        fixed_rows.extend(rows)
        for row in rows:
            print(f"{name} {row['model_name']} fixed CC={row['cc']:.3f} MAE={row['mae']:.2f}", flush=True)
        if not skip_cv:
            for split in tab.cv_splits():
                print(f"CV {name} {split.fold}", flush=True)
                rows, _, _ = chrun.evaluate_split(table, features_by_set[name], split, name, models)
                cv_rows.extend(rows)

    if not skip_private:
        for name, table in tables_all.items():
            print(f"\nPrivate diagnostic: {name}", flush=True)
            rows, pred = chrun.evaluate_private(table, features_by_set[name], name, models)
            private_rows.extend(rows)
            private_predictions.append(pred)
            for row in rows:
                print(f"{name} {row['model_name']} private CC={row['cc']:.3f} MAE={row['mae']:.2f}", flush=True)

    return pd.DataFrame(fixed_rows), pd.DataFrame(cv_rows), pd.DataFrame(private_rows), private_predictions


def summarize_results(fixed: pd.DataFrame, cv: pd.DataFrame, private: pd.DataFrame) -> pd.DataFrame:
    if cv.empty:
        cv = pd.DataFrame(columns=["feature_set_name", "model_name", "mae", "rmse", "cc", "mae_skill_vs_27day"])
    if private.empty:
        private = pd.DataFrame(columns=["feature_set_name", "model_name", "mae", "rmse", "cc", "mae_skill_vs_27day"])
    return chrun.summarize(fixed, cv, private)


def select_feature_set(summary: pd.DataFrame, cv_empty: bool) -> tuple[str, pd.DataFrame, bool, str]:
    ensemble = summary[summary["model_name"] == ENSEMBLE_MODEL].copy()
    baseline = ensemble[ensemble["feature_set_name"] == BASELINE].iloc[0]
    adopted = False
    reasons: list[str] = []
    for _, row in ensemble[ensemble["feature_set_name"] != BASELINE].iterrows():
        fixed_improves = bool(row["fixed_cc"] > baseline["fixed_cc"])
        cv_improves = False if cv_empty else bool(row["cv_mean_cc"] > baseline["cv_mean_cc"])
        if fixed_improves or cv_improves:
            adopted = True
            reasons.append(
                f"{row['feature_set_name']} improves fixed={fixed_improves} cv={cv_improves}"
            )
    if adopted:
        sort_cols = ["cv_mean_cc", "fixed_cc"] if not cv_empty else ["fixed_cc"]
        selected = ensemble.sort_values(sort_cols, ascending=False).iloc[0]["feature_set_name"]
    else:
        selected = BASELINE
        reasons.append("no backmapped flux feature set improved public fixed validation or public CV")
    return str(selected), ensemble, adopted, "; ".join(reasons)


def print_examples(examples: pd.DataFrame) -> None:
    cols = [
        "origin_time",
        "target_time",
        "Speed(t)",
        "estimated_source_time",
        "travel_time_hours",
        "requested_flux_time",
        "matched_HMI_time",
        "matched_HMI_time_le_requested_flux_time",
        "requested_flux_time_le_origin_time",
        "flux_at_source_time",
        "dflux_10h",
        "dflux_24h",
    ]
    print("\nSource-time sanity examples", flush=True)
    print(examples[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"), flush=True)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir if args.output_dir.is_absolute() else HERE / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mag, flux_cols = load_daily_magnetogram_features()
    base_public, base_all, base_features = build_base_tables()

    print(f"Using daily HMI flux columns: {flux_cols}", flush=True)
    print("Building causal coarse_daily_backmapped_flux_features", flush=True)
    public_flux, all_flux_features, daily_24h_features, public_sanity, public_examples = add_backmapped_flux_features(
        base_public,
        mag,
        flux_cols,
        sanity_rows=args.sanity_rows,
    )
    all_flux, _, _, all_sanity, all_examples = add_backmapped_flux_features(
        base_all,
        mag,
        flux_cols,
        sanity_rows=args.sanity_rows,
        random_state=20260627,
    )

    tables_public = {
        BASELINE: base_public,
        DAILY_ALL: public_flux,
        DAILY_24H: public_flux,
    }
    tables_all = {
        BASELINE: base_all,
        DAILY_ALL: all_flux,
        DAILY_24H: all_flux,
    }
    features_by_set = {
        BASELINE: base_features,
        DAILY_ALL: base_features + all_flux_features,
        DAILY_24H: base_features + daily_24h_features,
    }

    fixed_df, cv_df, private_df, private_predictions = evaluate_feature_sets(
        tables_public,
        tables_all,
        features_by_set,
        skip_cv=args.skip_cv,
        skip_private=args.skip_private,
    )
    summary_df = summarize_results(fixed_df, cv_df, private_df)

    timestamp_sanity = pd.concat([public_sanity, all_sanity], ignore_index=True)
    source_examples = pd.concat([public_examples, all_examples], ignore_index=True).head(20)
    selected, ensemble_summary, adopted, reason = select_feature_set(summary_df, cv_df.empty)

    fixed_df.to_csv(out_dir / "fixed_results.csv", index=False)
    cv_df.to_csv(out_dir / "cv_results.csv", index=False)
    private_df.to_csv(out_dir / "private_diagnostic.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    timestamp_sanity.to_csv(out_dir / "timestamp_sanity_check.csv", index=False)
    source_examples.to_csv(out_dir / "source_time_examples.csv", index=False)

    if private_predictions:
        pred_all = pd.concat(private_predictions, ignore_index=True)
        best_pred = pred_all[pred_all["feature_set_name"] == selected][["datetime", "predicted_speed"]]
        best_pred.to_csv(out_dir / "best_private_prediction.csv", index=False)
    else:
        pd.DataFrame(columns=["datetime", "predicted_speed"]).to_csv(out_dir / "best_private_prediction.csv", index=False)

    config = {
        "feature_label": "coarse_daily_backmapped_flux_features",
        "magnetogram_feature_file": str(MAG_CSV.relative_to(HERE)),
        "flux_columns": flux_cols,
        "speed_definitions": SPEED_DEFS,
        "lookback_hours": LOOKBACK_HOURS,
        "causality_rules": [
            "source_time_est <= origin_datetime",
            "requested_flux_time <= origin_datetime",
            "matched_HMI_time <= requested_flux_time using backward asof match",
            "target Speed(t+72h) is not used in source-time or flux-feature construction",
        ],
        "selected_feature_set": selected,
        "adopt_backmapped_flux_features": adopted,
        "selection_reason": reason,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, allow_nan=True))

    print_examples(source_examples)
    print("\nEnsemble summary", flush=True)
    print(
        ensemble_summary[
            [
                "feature_set_name",
                "fixed_mae",
                "fixed_rmse",
                "fixed_cc",
                "cv_mean_mae",
                "cv_mean_rmse",
                "cv_mean_cc",
                "private_mae",
                "private_rmse",
                "private_cc",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        flush=True,
    )
    print(f"\nSelection: {selected}", flush=True)
    print(f"Adopt backmapped flux features: {adopted}. {reason}", flush=True)
    print(f"Saved outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
