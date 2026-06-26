"""Evaluate derived OMNI regime features on top of representative CH features."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_ch_feature_addition_72h as chrun
import train_tabular_models_72h as tab


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs" / "omni_derived_features_72h"

BASIC_DERIVED = [
    "dynamic_pressure_proxy",
    "alfven_speed_proxy",
    "alfven_mach_proxy",
    "entropy_proxy",
]
TEMP_DERIVED = [
    "expected_temperature_from_speed",
    "temperature_ratio",
]
LAGS = [1, 6, 24, 72]
ROLL_MEANS = [24, 72]
ROLL_STDS = [24, 72]


def safe_expected_temperature(speed: pd.Series) -> pd.Series:
    """Empirical expected proton temperature in K from speed in km/s.

    The exact physical calibration varies by paper; here it is used only as a
    causal regime proxy. Values are clipped to avoid invalid low-speed tails.
    """

    v = speed.astype(float)
    texp = 1_000.0 * 0.031 * (v - 259.0) ** 2
    return texp.where((v > 0) & np.isfinite(v) & (texp > 1.0), np.nan)


def build_derived_origin_features(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    speed = df["Speed (km/s)"].astype(float)
    density = df["Density (1/cm^3)"].astype(float)
    temp = df["Temperature (K)"].astype(float)
    bmag = df["B (nT)"].astype(float)

    valid_density = density > 0
    valid_b = bmag > 0

    derived: dict[str, pd.Series] = {}
    derived["dynamic_pressure_proxy"] = (density * speed**2).where(valid_density & np.isfinite(speed), np.nan)
    derived["alfven_speed_proxy"] = (bmag / np.sqrt(density)).where(valid_density & valid_b, np.nan)
    derived["alfven_mach_proxy"] = (speed / derived["alfven_speed_proxy"]).where(
        valid_density & valid_b & (derived["alfven_speed_proxy"] > 0), np.nan
    )
    derived["entropy_proxy"] = (temp / np.power(density, 2.0 / 3.0)).where(valid_density & np.isfinite(temp), np.nan)
    derived["expected_temperature_from_speed"] = safe_expected_temperature(speed)
    derived["temperature_ratio"] = (temp / derived["expected_temperature_from_speed"]).where(
        np.isfinite(temp) & (derived["expected_temperature_from_speed"] > 0), np.nan
    )

    out: dict[str, Any] = {"origin_datetime": df["datetime"]}
    for name, series in derived.items():
        out[f"omni_{name}_current"] = series
        for lag in LAGS:
            out[f"omni_{name}_lag_{lag}h"] = series.shift(lag)
        for window in ROLL_MEANS:
            out[f"omni_{name}_roll_mean_{window}h"] = series.rolling(window, min_periods=max(2, window // 4)).mean()
        for window in ROLL_STDS:
            out[f"omni_{name}_roll_std_{window}h"] = series.rolling(window, min_periods=max(2, window // 4)).std()

    features = pd.DataFrame(out)
    for col in list(features.columns):
        if col == "origin_datetime":
            continue
        features[f"{col}__missing"] = features[col].isna().astype(np.float32)
    return features


def add_derived_features(table: pd.DataFrame, derived_origin: pd.DataFrame, variables: list[str]) -> tuple[pd.DataFrame, list[str]]:
    wanted_prefixes = [f"omni_{name}_" for name in variables]
    cols = [
        col
        for col in derived_origin.columns
        if col != "origin_datetime" and any(col.startswith(prefix) for prefix in wanted_prefixes)
    ]
    merged = table.merge(derived_origin[["origin_datetime"] + cols], on="origin_datetime", how="left")
    return merged, cols


def build_feature_tables() -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    ch = chrun.load_ch()
    derived = build_derived_origin_features(tab.FULL_CSV)

    table_public_base = tab.build_feature_table(tab.FULL_CSV)
    table_all_base = tab.build_feature_table(
        tab.FULL_CSV,
        min_target_year=2011,
        max_target_year=2025,
        require_finite_target=False,
        require_finite_persistence=False,
    )

    tables_public_ch, features_ch, _ = chrun.feature_sets(table_public_base, ch)
    tables_all_ch, _, _ = chrun.feature_sets(table_all_base, ch)

    base_public = tables_public_ch["current_plus_representative_mrmr_ch"]
    base_all = tables_all_ch["current_plus_representative_mrmr_ch"]
    base_features = features_ch["current_plus_representative_mrmr_ch"]

    public_basic, basic_cols = add_derived_features(base_public, derived, BASIC_DERIVED)
    all_basic, _ = add_derived_features(base_all, derived, BASIC_DERIVED)
    public_temp, temp_cols = add_derived_features(base_public, derived, BASIC_DERIVED + TEMP_DERIVED)
    all_temp, _ = add_derived_features(base_all, derived, BASIC_DERIVED + TEMP_DERIVED)

    tables = {
        "baseline_current_best": base_public,
        "baseline_current_best__all": base_all,
        "omni_derived_basic": public_basic,
        "omni_derived_basic__all": all_basic,
        "omni_derived_with_temperature_ratio": public_temp,
        "omni_derived_with_temperature_ratio__all": all_temp,
    }
    features = {
        "baseline_current_best": base_features,
        "omni_derived_basic": base_features + basic_cols,
        "omni_derived_with_temperature_ratio": base_features + temp_cols,
    }
    return tables, features


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables, features_by_set = build_feature_tables()
    models = chrun.model_configs()
    feature_sets = [
        "baseline_current_best",
        "omni_derived_basic",
        "omni_derived_with_temperature_ratio",
    ]

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

    ensemble_cv = (
        cv_df[cv_df["model_name"] == "ensemble_0p7_mlp_0p3_extratrees"]
        .groupby("feature_set_name")
        .agg(cv_mean_cc=("cc", "mean"), cv_mean_mae=("mae", "mean"))
        .reset_index()
    )
    ensemble_fixed = fixed_df[fixed_df["model_name"] == "ensemble_0p7_mlp_0p3_extratrees"][
        ["feature_set_name", "cc", "mae"]
    ].rename(columns={"cc": "fixed_cc", "mae": "fixed_mae"})
    public_select = (
        ensemble_cv.merge(ensemble_fixed, on="feature_set_name")
        .sort_values(["cv_mean_cc", "fixed_cc", "cv_mean_mae"], ascending=[False, False, True])
        .iloc[0]
    )
    best_name = str(public_select["feature_set_name"])
    best_pred = pd.concat(private_predictions, ignore_index=True)
    best_pred = best_pred[best_pred["feature_set_name"] == best_name][["datetime", "predicted_speed"]]
    best_pred.to_csv(OUT_DIR / "best_private_prediction.csv", index=False)

    concise = summary_df[summary_df["model_name"] == "ensemble_0p7_mlp_0p3_extratrees"][
        ["feature_set_name", "fixed_cc", "cv_mean_cc", "private_cc", "private_mae", "private_rmse"]
    ].rename(columns={"feature_set_name": "feature_set"})
    print("\nConcise comparison", flush=True)
    print(concise.to_string(index=False), flush=True)
    print(f"\nBest public-selected feature set: {best_name}", flush=True)
    print(f"Saved outputs to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
