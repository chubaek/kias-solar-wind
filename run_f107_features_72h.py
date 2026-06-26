"""Evaluate causal daily F10.7 features on top of representative CH features."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_ch_feature_addition_72h as chrun
import train_tabular_models_72h as tab


HERE = Path(__file__).resolve().parent
OMNI2_TXT = HERE / "OMNI2_H0_MRG1HR_563544.txt"
OUT_DIR = HERE / "outputs" / "f107_features_72h"

F107_BASE = "f107_origin"
F107_BASIC = [
    "f107_origin",
    "f107_roll_mean_27d",
    "f107_roll_mean_81d",
]
F107_FULL = [
    "f107_origin",
    "f107_roll_mean_7d",
    "f107_roll_mean_27d",
    "f107_roll_mean_81d",
    "f107_minus_roll_mean_27d",
    "f107_roll_mean_27d_minus_81d",
    "f107_trend_7d",
    "f107_trend_27d",
]


def load_daily_f107(path: Path = OMNI2_TXT) -> pd.DataFrame:
    """Load observed daily F10.7 from the local OMNI2 hourly file.

    The OMNI2 file repeats the daily F10.7 value on hourly rows. Values after
    the forecast origin are never used; downstream merging is by origin date.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Provide a daily observed F10.7 source covering 2011-2025."
        )

    rows: list[tuple[pd.Timestamp, float]] = []
    non_comment = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            non_comment += 1
            if non_comment <= 2:
                continue
            parts = line.split()
            if len(parts) <= 16:
                continue
            dt = pd.to_datetime(f"{parts[0]} {parts[1]}", format="%d-%m-%Y %H:%M:%S.%f")
            f107 = float(parts[16])
            if f107 >= 999.0:
                f107 = np.nan
            rows.append((dt.normalize(), f107))

    hourly = pd.DataFrame(rows, columns=["origin_date", "f107_origin"])
    if hourly.empty:
        raise RuntimeError(f"No F10.7 rows could be parsed from {path}.")

    daily = (
        hourly.groupby("origin_date", as_index=False)
        .agg(f107_origin=("f107_origin", "mean"))
        .sort_values("origin_date")
        .reset_index(drop=True)
    )
    return daily


def build_f107_origin_features() -> pd.DataFrame:
    daily = load_daily_f107()
    f107 = daily[F107_BASE].astype(float)

    features = pd.DataFrame({"origin_date": daily["origin_date"]})
    features["f107_origin"] = f107
    for window in [7, 27, 81]:
        features[f"f107_roll_mean_{window}d"] = f107.rolling(window, min_periods=max(2, window // 4)).mean()
    features["f107_minus_roll_mean_27d"] = f107 - features["f107_roll_mean_27d"]
    features["f107_roll_mean_27d_minus_81d"] = features["f107_roll_mean_27d"] - features["f107_roll_mean_81d"]
    features["f107_trend_7d"] = f107 - f107.shift(7)
    features["f107_trend_27d"] = f107 - f107.shift(27)

    for col in F107_FULL:
        features[f"{col}__missing"] = features[col].isna().astype(np.float32)
    return features


def add_f107_features(table: pd.DataFrame, f107: pd.DataFrame, base_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    cols = base_cols + [f"{col}__missing" for col in base_cols]
    out = table.copy()
    out["origin_date"] = pd.to_datetime(out["origin_datetime"]).dt.normalize()
    out = out.merge(f107[["origin_date"] + cols], on="origin_date", how="left")
    out = out.drop(columns=["origin_date"])

    for col in base_cols:
        missing_col = f"{col}__missing"
        if missing_col in out.columns:
            out[missing_col] = out[missing_col].fillna(1.0).astype(np.float32)
    return out, cols


def build_feature_tables() -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    ch = chrun.load_ch()
    f107 = build_f107_origin_features()

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

    public_basic, basic_cols = add_f107_features(base_public, f107, F107_BASIC)
    all_basic, _ = add_f107_features(base_all, f107, F107_BASIC)
    public_full, full_cols = add_f107_features(base_public, f107, F107_FULL)
    all_full, _ = add_f107_features(base_all, f107, F107_FULL)

    tables = {
        "baseline_current_best": base_public,
        "baseline_current_best__all": base_all,
        "f107_basic": public_basic,
        "f107_basic__all": all_basic,
        "f107_full": public_full,
        "f107_full__all": all_full,
    }
    features = {
        "baseline_current_best": base_features,
        "f107_basic": base_features + basic_cols,
        "f107_full": base_features + full_cols,
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


def public_selection(
    fixed_df: pd.DataFrame,
    cv_df: pd.DataFrame,
) -> tuple[str, pd.DataFrame, bool]:
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
    f107_public = public[public["feature_set_name"].isin(["f107_basic", "f107_full"])]
    adopt = bool(
        (f107_public["fixed_cc"].max() > baseline["fixed_cc"])
        or (f107_public["cv_mean_cc"].max() > baseline["cv_mean_cc"])
    )
    if adopt:
        candidates = f107_public
    else:
        candidates = public[public["feature_set_name"] == "baseline_current_best"]
    best = candidates.sort_values(["cv_mean_cc", "fixed_cc", "cv_mean_mae"], ascending=[False, False, True]).iloc[0]
    return str(best["feature_set_name"]), public, adopt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables, features_by_set = build_feature_tables()
    models = chrun.model_configs()
    feature_sets = ["baseline_current_best", "f107_basic", "f107_full"]

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

    best_name, _, adopt = public_selection(fixed_df, cv_df)
    best_pred = pd.concat(private_predictions, ignore_index=True)
    best_pred = best_pred[best_pred["feature_set_name"] == best_name][["datetime", "predicted_speed"]]
    best_pred.to_csv(OUT_DIR / "best_private_prediction.csv", index=False)

    concise = summary_df[summary_df["model_name"] == "ensemble_0p7_mlp_0p3_extratrees"][
        ["feature_set_name", "fixed_cc", "cv_mean_cc", "private_cc", "private_mae", "private_rmse"]
    ].rename(columns={"feature_set_name": "feature_set"})
    print("\nConcise comparison", flush=True)
    print(concise.to_string(index=False), flush=True)
    print(f"\nAdopt F10.7 features by public rule: {adopt}", flush=True)
    print(f"Best public-selected feature set: {best_name}", flush=True)
    print(f"Saved outputs to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
