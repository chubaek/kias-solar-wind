"""Evaluate magnetogram-enhanced CH features when local magnetic maps exist.

This script intentionally does not fabricate magnetic features. If no local
HMI/GONG or precomputed magnetogram feature table is present, it writes a clear
availability report and evaluates only the current CH baseline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_ch_feature_addition_72h as chrun
import train_tabular_models_72h as tab


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs" / "magnetogram_ch_features_72h"

MAGNETOGRAM_FEATURE_CANDIDATES = [
    HERE / "data" / "magnetograms" / "magnetogram_window_features_daily.csv",
    HERE / "magnetogram_ch_features" / "magnetogram_features_by_time.csv",
    HERE / "magnetogram_feature" / "magnetogram_features_by_time.csv",
    HERE / "magnetogram_features_by_time.csv",
]

WINDOW_FEATURES = [
    "mag_W_lon7p5_lat15_mean_abs_Br",
    "mag_W_lon7p5_lat15_median_abs_Br",
    "mag_W_lon7p5_lat15_signed_mean_Br",
    "mag_W_lon7p5_lat15_polarity_imbalance",
    "mag_W_lon7p5_lat15_dominant_polarity",
    "mag_W_lon7p5_lat15_area_times_mean_abs_Br",
    "mag_W_lon30_lat30_mean_abs_Br",
    "mag_W_lon30_lat30_signed_mean_Br",
    "mag_W_lon30_lat30_polarity_imbalance",
    "mag_W_lon30_lat30_area_times_mean_abs_Br",
    "mag_W_lon30_lat15_mean_abs_Br",
    "mag_W_lon30_lat15_signed_mean_Br",
    "mag_W_lon30_lat15_polarity_imbalance",
    "mag_W_lon30_lat15_area_times_mean_abs_Br",
    "mag_W_lon60_lat60_mean_abs_Br",
    "mag_W_lon60_lat60_signed_mean_Br",
    "mag_W_lon60_lat60_polarity_imbalance",
    "mag_A_source_if_overlap_W_lon60_lat60_source_mean_abs_Br",
    "mag_A_source_if_overlap_W_lon60_lat60_source_signed_mean_Br",
    "mag_A_source_if_overlap_W_lon60_lat60_source_area_times_mean_abs_Br",
]


@dataclass(frozen=True)
class MagSpec:
    window: str
    target_lag_days: int
    origin_offset_days: int
    features: tuple[str, ...]
    feature_set: str


MAGNETOGRAM_SPECS = [
    MagSpec(
        "W_lon7p5_lat15",
        4,
        -1,
        (
            "mag_W_lon7p5_lat15_mean_abs_Br",
            "mag_W_lon7p5_lat15_median_abs_Br",
            "mag_W_lon7p5_lat15_signed_mean_Br",
            "mag_W_lon7p5_lat15_polarity_imbalance",
            "mag_W_lon7p5_lat15_dominant_polarity",
            "mag_W_lon7p5_lat15_area_times_mean_abs_Br",
        ),
        "magnetogram_window_features",
    ),
    MagSpec(
        "W_lon30_lat30",
        5,
        -2,
        (
            "mag_W_lon30_lat30_mean_abs_Br",
            "mag_W_lon30_lat30_signed_mean_Br",
            "mag_W_lon30_lat30_polarity_imbalance",
            "mag_W_lon30_lat30_area_times_mean_abs_Br",
        ),
        "magnetogram_window_features",
    ),
    MagSpec(
        "W_lon30_lat15",
        4,
        -1,
        (
            "mag_W_lon30_lat15_mean_abs_Br",
            "mag_W_lon30_lat15_signed_mean_Br",
            "mag_W_lon30_lat15_polarity_imbalance",
            "mag_W_lon30_lat15_area_times_mean_abs_Br",
        ),
        "magnetogram_window_features",
    ),
    MagSpec(
        "W_lon60_lat60",
        4,
        -1,
        (
            "mag_W_lon60_lat60_mean_abs_Br",
            "mag_W_lon60_lat60_signed_mean_Br",
            "mag_W_lon60_lat60_polarity_imbalance",
        ),
        "magnetogram_window_features",
    ),
    MagSpec(
        "A_source_if_overlap_W_lon60_lat60",
        4,
        -1,
        (
            "mag_A_source_if_overlap_W_lon60_lat60_source_mean_abs_Br",
            "mag_A_source_if_overlap_W_lon60_lat60_source_signed_mean_Br",
            "mag_A_source_if_overlap_W_lon60_lat60_source_area_times_mean_abs_Br",
        ),
        "magnetogram_ch_mask_features",
    ),
]


def find_magnetogram_feature_file() -> Path | None:
    for path in MAGNETOGRAM_FEATURE_CANDIDATES:
        if path.exists() and path.stat().st_size > 1:
            return path
    return None


def load_magnetogram_features(path: Path) -> pd.DataFrame:
    mag = pd.read_csv(path)
    if "time" in mag.columns:
        time_col = "time"
    elif "magnetogram_time" in mag.columns:
        time_col = "magnetogram_time"
    else:
        raise RuntimeError(f"{path} must contain a 'time' or 'magnetogram_time' column.")
    mag["mag_time"] = pd.to_datetime(mag[time_col], utc=True).dt.tz_convert(None)
    mag = adapt_stage0_approximate_columns(mag)
    return mag.sort_values("mag_time").reset_index(drop=True)


def adapt_stage0_approximate_columns(mag: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    metric_map = {
        "mean_abs_B": "mean_abs_Br",
        "median_abs_B": "median_abs_Br",
        "signed_mean_B": "signed_mean_Br",
        "polarity_imbalance": "polarity_imbalance",
        "dominant_polarity": "dominant_polarity",
    }
    for window in ["W_lon7p5_lat15", "W_lon30_lat30", "W_lon30_lat15", "W_lon60_lat60"]:
        for old_metric, new_metric in metric_map.items():
            old = f"{window}_{old_metric}"
            if old in mag.columns:
                rename[old] = f"mag_{window}_{new_metric}"
        area_col = f"{window}_area_times_mean_abs_B"
        if area_col in mag.columns:
            rename[area_col] = f"mag_{window}_area_times_mean_abs_Br"
        mean_abs = f"{window}_mean_abs_B"
        n_pix = f"{window}_n_pix"
        if mean_abs in mag.columns and n_pix in mag.columns and area_col not in mag.columns:
            mag[f"mag_{window}_area_times_mean_abs_Br"] = mag[mean_abs] * mag[n_pix]
    if rename:
        mag = mag.rename(columns=rename)
    return mag


def available_specs(mag: pd.DataFrame | None) -> list[MagSpec]:
    if mag is None or mag.empty:
        return []
    cols = set(mag.columns)
    specs = []
    for spec in MAGNETOGRAM_SPECS:
        if not all(col in cols for col in spec.features):
            continue
        if not any(mag[col].notna().any() for col in spec.features):
            continue
        specs.append(spec)
    return specs


def add_magnetogram_features(
    table: pd.DataFrame,
    mag: pd.DataFrame,
    specs: list[MagSpec],
    feature_set: str,
    tolerance_hours: int,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    selected = [spec for spec in specs if spec.feature_set == feature_set]
    if not selected:
        return table.copy(), [], pd.DataFrame()

    out = table.copy()
    base = out[["origin_datetime", "target_datetime"]].copy()
    base["origin_datetime"] = pd.to_datetime(base["origin_datetime"])
    base["target_datetime"] = pd.to_datetime(base["target_datetime"])
    base["_row_id"] = np.arange(len(base))

    new_cols: list[str] = []
    sanity_parts: list[pd.DataFrame] = []
    for spec in selected:
        request = base[["_row_id", "origin_datetime", "target_datetime"]].copy()
        request["requested_magnetogram_time"] = request["origin_datetime"] + pd.to_timedelta(
            spec.origin_offset_days, unit="D"
        )
        request = request.sort_values("requested_magnetogram_time")
        cols = ["mag_time", *spec.features]
        merged = pd.merge_asof(
            request,
            mag[cols].sort_values("mag_time"),
            left_on="requested_magnetogram_time",
            right_on="mag_time",
            direction="backward",
            tolerance=pd.Timedelta(hours=tolerance_hours),
        ).sort_values("_row_id")

        for feature in spec.features:
            out_col = f"{feature}__target_lag_{spec.target_lag_days}d__origin_offset_{spec.origin_offset_days}d"
            out[out_col] = merged[feature].to_numpy()
            out[f"{out_col}__missing"] = out[out_col].isna().astype(np.float32)
            new_cols.extend([out_col, f"{out_col}__missing"])

        sanity = merged[["_row_id", "origin_datetime", "target_datetime", "requested_magnetogram_time", "mag_time"]].copy()
        sanity["window"] = spec.window
        sanity["feature_set_name"] = feature_set
        sanity["target_lag_days"] = spec.target_lag_days
        sanity["origin_offset_days"] = spec.origin_offset_days
        sanity["matched_magnetogram_time_le_requested"] = sanity["mag_time"].isna() | (
            sanity["mag_time"] <= sanity["requested_magnetogram_time"]
        )
        sanity["requested_time_le_origin"] = sanity["requested_magnetogram_time"] <= sanity["origin_datetime"]
        sanity["target_is_origin_plus_72h"] = (sanity["target_datetime"] - sanity["origin_datetime"]) == pd.Timedelta(hours=72)
        sanity_parts.append(sanity)

    sanity_all = pd.concat(sanity_parts, ignore_index=True) if sanity_parts else pd.DataFrame()
    return out, new_cols, sanity_all


def build_baseline_tables() -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    ch = chrun.load_ch()
    table_public_base = tab.build_feature_table(tab.FULL_CSV)
    table_all_base = tab.build_feature_table(
        tab.FULL_CSV,
        min_target_year=2011,
        max_target_year=2025,
        require_finite_target=False,
        require_finite_persistence=False,
    )
    tables_public_ch, features_ch, sanity_public = chrun.feature_sets(table_public_base, ch)
    tables_all_ch, _, sanity_all = chrun.feature_sets(table_all_base, ch)
    return (
        tables_public_ch["current_plus_representative_mrmr_ch"],
        tables_all_ch["current_plus_representative_mrmr_ch"],
        features_ch["current_plus_representative_mrmr_ch"],
        pd.concat([sanity_public, sanity_all], ignore_index=True),
    )


def build_feature_tables(
    tolerance_hours: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]], pd.DataFrame, pd.DataFrame]:
    base_public, base_all, base_features, ch_sanity = build_baseline_tables()
    mag_path = find_magnetogram_feature_file()
    mag = load_magnetogram_features(mag_path) if mag_path is not None else None
    specs = available_specs(mag)

    tables = {
        "baseline_current_best": base_public,
        "baseline_current_best__all": base_all,
    }
    features = {"baseline_current_best": base_features}
    sanity_parts = []

    if mag is not None:
        for feature_set in ["magnetogram_window_features", "magnetogram_ch_mask_features"]:
            public_mag, mag_cols, sanity_public = add_magnetogram_features(
                base_public, mag, specs, feature_set, tolerance_hours
            )
            all_mag, _, sanity_all = add_magnetogram_features(base_all, mag, specs, feature_set, tolerance_hours)
            if mag_cols:
                tables[feature_set] = public_mag
                tables[f"{feature_set}__all"] = all_mag
                features[feature_set] = base_features + mag_cols
                sanity_parts.extend([sanity_public, sanity_all])

    mag_sanity = pd.concat(sanity_parts, ignore_index=True) if sanity_parts else pd.DataFrame()
    report = availability_report(base_all, mag_path, mag, specs, mag_sanity)
    timestamp_sanity = timestamp_sanity_check(ch_sanity, mag_sanity, report)
    return tables, features, report, timestamp_sanity


def availability_report(
    table_all: pd.DataFrame,
    mag_path: Path | None,
    mag: pd.DataFrame | None,
    specs: list[MagSpec],
    mag_sanity: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    usable_mag = mag is not None and not mag.empty
    source = "none"
    if mag_path is not None:
        lower = str(mag_path).lower()
        source = "HMI" if "hmi" in lower else "GONG" if "gong" in lower else "precomputed_unknown_source"

    feature_mode = "not_available"
    if specs:
        has_mask = any(spec.feature_set == "magnetogram_ch_mask_features" for spec in specs)
        has_window = any(spec.feature_set == "magnetogram_window_features" for spec in specs)
        if has_mask and has_window:
            feature_mode = "window_and_ch_mask"
        elif has_mask:
            feature_mode = "ch_mask"
        elif has_window:
            feature_mode = "window"

    matched = int(mag_sanity["mag_time"].notna().sum()) if not mag_sanity.empty else 0
    requested = int(len(mag_sanity)) if not mag_sanity.empty else 0
    rows.append(
        {
            "report_type": "overall",
            "item": "magnetogram_availability",
            "value": "available" if usable_mag else "missing",
            "source": source,
            "feature_mode": feature_mode,
            "matched_rows": matched,
            "requested_rows": requested,
            "missing_rate": float(1.0 - matched / requested) if requested else 1.0,
            "note": availability_note(mag_path, mag),
        }
    )

    if not usable_mag:
        years = sorted(pd.to_datetime(table_all["origin_datetime"]).dt.year.dropna().unique())
        for year in years:
            rows.append(
                {
                    "report_type": "missing_rate_by_year",
                    "item": int(year),
                    "value": np.nan,
                    "source": source,
                    "feature_mode": feature_mode,
                    "matched_rows": 0,
                    "requested_rows": int((pd.to_datetime(table_all["origin_datetime"]).dt.year == year).sum()),
                    "missing_rate": 1.0,
                    "note": "No magnetogram data source available locally.",
                }
            )
        for feature in WINDOW_FEATURES:
            rows.append(
                {
                    "report_type": "missing_rate_by_feature",
                    "item": feature,
                    "value": np.nan,
                    "source": source,
                    "feature_mode": feature_mode,
                    "matched_rows": 0,
                    "requested_rows": 0,
                    "missing_rate": 1.0,
                    "note": "Feature not computed because magnetogram data is missing.",
                }
            )
        return pd.DataFrame(rows)

    if not mag_sanity.empty:
        by_year = mag_sanity.assign(origin_year=pd.to_datetime(mag_sanity["origin_datetime"]).dt.year)
        for year, group in by_year.groupby("origin_year"):
            matched_y = int(group["mag_time"].notna().sum())
            rows.append(
                {
                    "report_type": "missing_rate_by_year",
                    "item": int(year),
                    "value": np.nan,
                    "source": source,
                    "feature_mode": feature_mode,
                    "matched_rows": matched_y,
                    "requested_rows": int(len(group)),
                    "missing_rate": float(1.0 - matched_y / len(group)),
                    "note": "",
                }
            )
    available = {feature for spec in specs for feature in spec.features}
    for feature in WINDOW_FEATURES:
        rows.append(
            {
                "report_type": "missing_rate_by_feature",
                "item": feature,
                "value": np.nan,
                "source": source,
                "feature_mode": feature_mode,
                "matched_rows": matched,
                "requested_rows": requested,
                "missing_rate": 0.0 if feature in available and requested else 1.0,
                "note": "" if feature in available else "Feature column missing from precomputed table.",
            }
        )
    return pd.DataFrame(rows)


def availability_note(mag_path: Path | None, mag: pd.DataFrame | None) -> str:
    if mag_path is None:
        return "No local HMI/GONG/precomputed magnetogram feature file found."
    if mag is None:
        return "Magnetogram feature file could not be loaded."
    if mag.empty:
        return f"Magnetogram feature file exists but has zero rows: {mag_path}"
    return str(mag_path)


def timestamp_sanity_check(ch_sanity: pd.DataFrame, mag_sanity: pd.DataFrame, report: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not ch_sanity.empty:
        sample = ch_sanity.sample(n=min(10, len(ch_sanity)), random_state=20260624)
        for _, row in sample.iterrows():
            rows.append(
                {
                    "source": "CH",
                    "feature_set_name": "baseline_current_best",
                    "origin_datetime": row["origin_datetime"],
                    "target_datetime": row["target_datetime"],
                    "requested_time": row["requested_ch_time"],
                    "matched_time": row["ch_time"],
                    "matched_time_le_requested": row["matched_ch_time_le_requested"],
                    "requested_time_le_origin": row["requested_time_le_origin"],
                    "target_is_origin_plus_72h": row["target_is_origin_plus_72h"],
                    "note": row["feature"],
                }
            )
    if not mag_sanity.empty:
        sample = mag_sanity.sample(n=min(20, len(mag_sanity)), random_state=20260624)
        for _, row in sample.iterrows():
            rows.append(
                {
                    "source": "magnetogram",
                    "feature_set_name": row["feature_set_name"],
                    "origin_datetime": row["origin_datetime"],
                    "target_datetime": row["target_datetime"],
                    "requested_time": row["requested_magnetogram_time"],
                    "matched_time": row["mag_time"],
                    "matched_time_le_requested": row["matched_magnetogram_time_le_requested"],
                    "requested_time_le_origin": row["requested_time_le_origin"],
                    "target_is_origin_plus_72h": row["target_is_origin_plus_72h"],
                    "note": row["window"],
                }
            )
    if mag_sanity.empty:
        rows.append(
            {
                "source": "magnetogram",
                "feature_set_name": "magnetogram_window_features",
                "origin_datetime": pd.NaT,
                "target_datetime": pd.NaT,
                "requested_time": pd.NaT,
                "matched_time": pd.NaT,
                "matched_time_le_requested": True,
                "requested_time_le_origin": True,
                "target_is_origin_plus_72h": True,
                "note": str(report.iloc[0]["note"]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_split(
    table: pd.DataFrame,
    features: list[str],
    split: tab.Split,
    feature_set: str,
    models: dict[str, Any],
) -> list[dict[str, Any]]:
    rows, _, _ = chrun.evaluate_split(table, features, split, feature_set, models)
    return rows


def evaluate_private(
    table: pd.DataFrame,
    features: list[str],
    feature_set: str,
    models: dict[str, Any],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    rows, pred = chrun.evaluate_private(table, features, feature_set, models)
    return rows, pred


def summarize(fixed: pd.DataFrame, cv: pd.DataFrame, private: pd.DataFrame) -> pd.DataFrame:
    cv_mean = (
        cv.groupby(["feature_set_name", "model_name"], dropna=False)
        .agg(
            cv_mean_mae=("mae", "mean"),
            cv_mean_rmse=("rmse", "mean"),
            cv_mean_cc=("cc", "mean"),
            cv_mean_skill=("mae_skill_vs_27day", "mean"),
        )
        .reset_index()
    )
    fixed_s = fixed[["feature_set_name", "model_name", "mae", "rmse", "cc", "mae_skill_vs_27day"]].rename(
        columns={"mae": "fixed_mae", "rmse": "fixed_rmse", "cc": "fixed_cc", "mae_skill_vs_27day": "fixed_skill"}
    )
    priv_s = private[["feature_set_name", "model_name", "mae", "rmse", "cc", "mae_skill_vs_27day"]].rename(
        columns={"mae": "private_mae", "rmse": "private_rmse", "cc": "private_cc", "mae_skill_vs_27day": "private_skill"}
    )
    return fixed_s.merge(cv_mean, on=["feature_set_name", "model_name"], how="left").merge(
        priv_s, on=["feature_set_name", "model_name"], how="left"
    )


def public_selection(fixed_df: pd.DataFrame, cv_df: pd.DataFrame) -> tuple[str, bool]:
    model = "ensemble_0p7_mlp_0p3_extratrees"
    ensemble_cv = (
        cv_df[cv_df["model_name"] == model]
        .groupby("feature_set_name")
        .agg(cv_mean_cc=("cc", "mean"), cv_mean_mae=("mae", "mean"))
        .reset_index()
    )
    ensemble_fixed = fixed_df[fixed_df["model_name"] == model][["feature_set_name", "cc", "mae"]].rename(
        columns={"cc": "fixed_cc", "mae": "fixed_mae"}
    )
    public = ensemble_cv.merge(ensemble_fixed, on="feature_set_name")
    baseline = public[public["feature_set_name"] == "baseline_current_best"].iloc[0]
    magnetogram_public = public[public["feature_set_name"].str.startswith("magnetogram_")]
    adopt = False
    if not magnetogram_public.empty:
        adopt = bool(
            (magnetogram_public["fixed_cc"].max() > baseline["fixed_cc"])
            or (magnetogram_public["cv_mean_cc"].max() > baseline["cv_mean_cc"])
        )
    if adopt:
        candidates = magnetogram_public
    else:
        candidates = public[public["feature_set_name"] == "baseline_current_best"]
    best = candidates.sort_values(["cv_mean_cc", "fixed_cc", "cv_mean_mae"], ascending=[False, False, True]).iloc[0]
    return str(best["feature_set_name"]), adopt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--tolerance-hours", type=int, default=12)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables, features_by_set, report, timestamp_sanity = build_feature_tables(args.tolerance_hours)
    models = chrun.model_configs()
    feature_sets = [name for name in tables if not name.endswith("__all")]

    report.to_csv(OUT_DIR / "data_availability_report.csv", index=False)
    timestamp_sanity.to_csv(OUT_DIR / "timestamp_sanity_check.csv", index=False)

    print("\nData availability report", flush=True)
    overall = report[report["report_type"] == "overall"]
    print(overall.to_string(index=False), flush=True)
    print("\nMissing rate by year", flush=True)
    print(report[report["report_type"] == "missing_rate_by_year"].to_string(index=False), flush=True)
    print("\nMissing rate by feature", flush=True)
    print(report[report["report_type"] == "missing_rate_by_feature"][["item", "missing_rate", "note"]].to_string(index=False), flush=True)

    fixed_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    private_predictions: list[pd.DataFrame] = []

    for name in feature_sets:
        print(f"\nFixed evaluation: {name}", flush=True)
        rows = evaluate_split(tables[name], features_by_set[name], tab.fixed_split(), name, models)
        fixed_rows.extend(rows)
        for row in rows:
            print(f"{name} {row['model_name']} fixed_CC={row['cc']:.3f} MAE={row['mae']:.2f}", flush=True)

        if not args.skip_cv:
            for split in tab.cv_splits():
                rows = evaluate_split(tables[name], features_by_set[name], split, name, models)
                cv_rows.extend(rows)
                ensemble_row = next(r for r in rows if r["model_name"] == "ensemble_0p7_mlp_0p3_extratrees")
                print(f"{name} {split.fold} ensemble_CV_CC={ensemble_row['cc']:.3f}", flush=True)

    for name in feature_sets:
        print(f"\nPrivate diagnostic: {name}", flush=True)
        rows, pred = evaluate_private(tables[f"{name}__all"], features_by_set[name], name, models)
        private_rows.extend(rows)
        private_predictions.append(pred)
        for row in rows:
            print(f"{name} {row['model_name']} private_CC={row['cc']:.3f} MAE={row['mae']:.2f}", flush=True)

    fixed_df = pd.DataFrame(fixed_rows)
    cv_df = pd.DataFrame(cv_rows)
    private_df = pd.DataFrame(private_rows)
    summary_df = summarize(fixed_df, cv_df, private_df)

    fixed_df.to_csv(OUT_DIR / "fixed_results.csv", index=False)
    cv_df.to_csv(OUT_DIR / "cv_results.csv", index=False)
    private_df.to_csv(OUT_DIR / "private_diagnostic.csv", index=False)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)

    best_name, adopt = public_selection(fixed_df, cv_df)
    best_pred = pd.concat(private_predictions, ignore_index=True)
    best_pred = best_pred[best_pred["feature_set_name"] == best_name][["datetime", "predicted_speed"]]
    best_pred.to_csv(OUT_DIR / "best_private_prediction.csv", index=False)

    concise = summary_df[summary_df["model_name"] == "ensemble_0p7_mlp_0p3_extratrees"][
        ["feature_set_name", "fixed_cc", "cv_mean_cc", "private_cc", "private_mae", "private_rmse"]
    ].rename(columns={"feature_set_name": "feature_set"})
    print("\nConcise comparison", flush=True)
    print(concise.to_string(index=False), flush=True)
    print(f"\nAdopt magnetogram features by public rule: {adopt}", flush=True)
    print(f"Best public-selected feature set: {best_name}", flush=True)
    print(f"Saved outputs to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
