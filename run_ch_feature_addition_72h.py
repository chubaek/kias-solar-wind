"""Evaluate selected CH morphology/intensity features for the frozen 72h family."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

import train_tabular_models_72h as tab


HERE = Path(__file__).resolve().parent
CH_CSV = HERE / "ch_feature" / "ch_features_by_time.csv"
OUT_DIR = HERE / "outputs" / "ch_feature_addition_72h"

ENSEMBLE_WEIGHTS = {"mlp": 0.70, "extratrees": 0.30}


@dataclass(frozen=True)
class CHSpec:
    column: str
    target_lag_days: int
    origin_offset_days: int

    @property
    def output_name(self) -> str:
        safe = self.column.replace(".", "p")
        return f"ch_{safe}__target_lag_{self.target_lag_days}d__origin_offset_{self.origin_offset_days}d"


REPRESENTATIVE_CH = [
    CHSpec("I_mean_W_lon7p5_lat15", 4, -1),
    CHSpec("A_W_lon30_lat30_km2", 5, -2),
    CHSpec("A_source_if_overlap_W_lon60_lat60_km2", 4, -1),
    CHSpec("A_W_lon30_lat15_km2", 4, -1),
    CHSpec("lat_width_eff_10_full_if_overlap_largest_W_lon7p5_lat15", 3, 0),
    CHSpec("A_grid_4x3_lat1_lon1_km2", 4, -1),
]

EXPANDED_EXTRA_CH = [
    CHSpec("log_I_mean_W_lon7p5_lat15", 4, -1),
    CHSpec("log_I_mean_W_lon7p5_lat15", 5, -2),
    CHSpec("I_mean_W_lon7p5_lat15", 5, -2),
    CHSpec("A_frac_visible_hemisphere_W_lon30_lat30", 5, -2),
]


def metric_row(y: np.ndarray, pred: np.ndarray, persistence: np.ndarray) -> dict[str, float]:
    return tab.metrics(y, pred, persistence)


def load_ch() -> pd.DataFrame:
    ch = pd.read_csv(CH_CSV)
    excluded = {
        "solar_wind_speed_kms",
        "omni_V1800",
        "omni_time",
        "omni_matched",
        "omni_speed_source_column",
    }
    if "time" not in ch.columns:
        raise RuntimeError("CH file must contain a 'time' column.")
    forbidden = sorted(set(ch.columns) & excluded)
    print(f"excluded_target_or_omni_columns={forbidden}")
    ch["ch_time"] = pd.to_datetime(ch["time"], utc=True).dt.tz_convert(None)
    return ch.sort_values("ch_time").reset_index(drop=True)


def validate_specs(ch: pd.DataFrame, specs: list[CHSpec]) -> None:
    missing = sorted({spec.column for spec in specs if spec.column not in ch.columns})
    if missing:
        raise RuntimeError(f"CH feature columns missing from file: {missing}")


def add_ch_features(
    table: pd.DataFrame,
    ch: pd.DataFrame,
    specs: list[CHSpec],
    tolerance_hours: int = 12,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    out = table.copy()
    base = out[["origin_datetime", "target_datetime"]].copy()
    base["origin_datetime"] = pd.to_datetime(base["origin_datetime"])
    base["target_datetime"] = pd.to_datetime(base["target_datetime"])
    base["_row_id"] = np.arange(len(base))
    sanity_parts = []
    new_cols = []

    for spec in specs:
        request = base[["_row_id", "origin_datetime", "target_datetime"]].copy()
        request["requested_ch_time"] = request["origin_datetime"] + pd.to_timedelta(spec.origin_offset_days, unit="D")
        request = request.sort_values("requested_ch_time")
        merged = pd.merge_asof(
            request,
            ch[["ch_time", spec.column]].sort_values("ch_time"),
            left_on="requested_ch_time",
            right_on="ch_time",
            direction="backward",
            tolerance=pd.Timedelta(hours=tolerance_hours),
        ).sort_values("_row_id")
        col = spec.output_name
        out[col] = merged[spec.column].to_numpy()
        out[f"{col}__missing"] = out[col].isna().astype(np.float32)
        new_cols.extend([col, f"{col}__missing"])

        sanity = merged[["_row_id", "origin_datetime", "target_datetime", "requested_ch_time", "ch_time", spec.column]].copy()
        sanity["feature"] = spec.column
        sanity["target_lag_days"] = spec.target_lag_days
        sanity["origin_offset_days"] = spec.origin_offset_days
        sanity["matched_ch_time_le_requested"] = sanity["ch_time"].isna() | (sanity["ch_time"] <= sanity["requested_ch_time"])
        sanity["requested_time_le_origin"] = sanity["requested_ch_time"] <= sanity["origin_datetime"]
        sanity["target_is_origin_plus_72h"] = (sanity["target_datetime"] - sanity["origin_datetime"]) == pd.Timedelta(hours=72)
        sanity_parts.append(sanity)

    sanity_all = pd.concat(sanity_parts, ignore_index=True) if sanity_parts else pd.DataFrame()
    return out, new_cols, sanity_all


def feature_sets(table: pd.DataFrame, ch: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]], pd.DataFrame]:
    base_features = tab.feature_columns(table)
    specs_a = REPRESENTATIVE_CH
    specs_b = REPRESENTATIVE_CH + EXPANDED_EXTRA_CH
    validate_specs(ch, specs_b)

    table_a, cols_a, sanity_a = add_ch_features(table, ch, specs_a)
    table_b, cols_b, sanity_b = add_ch_features(table, ch, specs_b)
    return (
        {
            "current_final_features": table,
            "current_plus_representative_mrmr_ch": table_a,
            "current_plus_expanded_mrmr_ch": table_b,
        },
        {
            "current_final_features": base_features,
            "current_plus_representative_mrmr_ch": base_features + cols_a,
            "current_plus_expanded_mrmr_ch": base_features + cols_b,
        },
        pd.concat([sanity_a, sanity_b], ignore_index=True),
    )


def model_configs() -> dict[str, Any]:
    configs = tab.candidate_models("initial")
    mlp = next(c for c in configs if c["name"].startswith("mlp_hidden128x64"))
    extra = next(c for c in configs if c["name"] == "extratrees_n300_depth12_min5_feat0.8")
    return {"direct_mlp": mlp["estimator"], "extratrees": extra["estimator"]}


def fit_predict(estimator: Any, X_train: pd.DataFrame, y_train: np.ndarray, X_eval: pd.DataFrame) -> np.ndarray:
    model = clone(estimator)
    model.fit(X_train, y_train)
    return model.predict(X_eval)


def evaluate_split(
    table: pd.DataFrame,
    features: list[str],
    split: tab.Split,
    feature_set_name: str,
    models: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], np.ndarray]:
    train_idx, val_idx = tab.split_rows(table, split)
    X_train = table.iloc[train_idx][features]
    X_val = table.iloc[val_idx][features]
    y_train = table.iloc[train_idx]["target_speed"].to_numpy(dtype=np.float32)
    y_val = table.iloc[val_idx]["target_speed"].to_numpy(dtype=np.float32)
    persistence = table.iloc[val_idx]["persistence_27day_target_aligned"].to_numpy(dtype=np.float32)

    preds = {
        "direct_mlp": fit_predict(models["direct_mlp"], X_train, y_train, X_val),
        "extratrees": fit_predict(models["extratrees"], X_train, y_train, X_val),
    }
    preds["ensemble_0p7_mlp_0p3_extratrees"] = ENSEMBLE_WEIGHTS["mlp"] * preds["direct_mlp"] + ENSEMBLE_WEIGHTS["extratrees"] * preds["extratrees"]

    rows = []
    for name, pred in preds.items():
        rows.append(
            {
                "feature_set_name": feature_set_name,
                "model_name": name,
                "validation_scheme": split.scheme,
                "fold": split.fold,
                **metric_row(y_val, pred, persistence),
            }
        )
    return rows, preds, val_idx


def evaluate_private(
    table: pd.DataFrame,
    features: list[str],
    feature_set_name: str,
    models: dict[str, Any],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    public_idx = np.flatnonzero(((table["target_year"] >= 2011) & (table["target_year"] <= 2023) & table["target_speed"].notna()).to_numpy())
    private_idx = np.flatnonzero(((table["target_year"] >= 2024) & (table["target_year"] <= 2025) & table["target_speed"].notna()).to_numpy())
    X_train = table.iloc[public_idx][features]
    X_private = table.iloc[private_idx][features]
    y_train = table.iloc[public_idx]["target_speed"].to_numpy(dtype=np.float32)
    y_private = table.iloc[private_idx]["target_speed"].to_numpy(dtype=np.float32)
    persistence = table.iloc[private_idx]["persistence_27day_target_aligned"].to_numpy(dtype=np.float32)
    persistence_fill = np.where(np.isfinite(persistence), persistence, float(np.mean(y_train)))

    preds = {
        "direct_mlp": fit_predict(models["direct_mlp"], X_train, y_train, X_private),
        "extratrees": fit_predict(models["extratrees"], X_train, y_train, X_private),
    }
    preds["ensemble_0p7_mlp_0p3_extratrees"] = ENSEMBLE_WEIGHTS["mlp"] * preds["direct_mlp"] + ENSEMBLE_WEIGHTS["extratrees"] * preds["extratrees"]

    rows = []
    for name, pred in preds.items():
        rows.append(
            {
                "feature_set_name": feature_set_name,
                "model_name": name,
                "validation_scheme": "private_diagnostic",
                "fold": "private_2024_2025",
                **metric_row(y_private, pred, persistence_fill),
            }
        )
    pred_frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(table.iloc[private_idx]["target_datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_speed": preds["ensemble_0p7_mlp_0p3_extratrees"],
            "observed_speed": y_private,
            "feature_set_name": feature_set_name,
        }
    )
    return rows, pred_frame


def summarize(fixed: pd.DataFrame, cv: pd.DataFrame, private: pd.DataFrame) -> pd.DataFrame:
    cv_mean = (
        cv.groupby(["feature_set_name", "model_name"], dropna=False)
        .agg(cv_mean_mae=("mae", "mean"), cv_mean_rmse=("rmse", "mean"), cv_mean_cc=("cc", "mean"), cv_mean_skill=("mae_skill_vs_27day", "mean"))
        .reset_index()
    )
    fixed_s = fixed[fixed["fold"] == "fixed"][["feature_set_name", "model_name", "mae", "rmse", "cc", "mae_skill_vs_27day"]].rename(
        columns={"mae": "fixed_mae", "rmse": "fixed_rmse", "cc": "fixed_cc", "mae_skill_vs_27day": "fixed_skill"}
    )
    priv_s = private[["feature_set_name", "model_name", "mae", "rmse", "cc", "mae_skill_vs_27day"]].rename(
        columns={"mae": "private_mae", "rmse": "private_rmse", "cc": "private_cc", "mae_skill_vs_27day": "private_skill"}
    )
    return fixed_s.merge(cv_mean, on=["feature_set_name", "model_name"], how="left").merge(priv_s, on=["feature_set_name", "model_name"], how="left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--tolerance-hours", type=int, default=12)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ch = load_ch()
    table_public = tab.build_feature_table(tab.FULL_CSV)
    table_all = tab.build_feature_table(
        tab.FULL_CSV,
        min_target_year=2011,
        max_target_year=2025,
        require_finite_target=False,
        require_finite_persistence=False,
    )
    tables_public, features_by_set, sanity_public = feature_sets(table_public, ch)
    tables_all, _, sanity_all = feature_sets(table_all, ch)
    models = model_configs()

    sanity = sanity_all.sample(n=min(10, len(sanity_all)), random_state=20260624).sort_values(["origin_datetime", "feature"])
    sanity_cols = [
        "origin_datetime",
        "target_datetime",
        "feature",
        "target_lag_days",
        "origin_offset_days",
        "requested_ch_time",
        "ch_time",
        "matched_ch_time_le_requested",
        "requested_time_le_origin",
        "target_is_origin_plus_72h",
    ]
    sanity[sanity_cols].to_csv(OUT_DIR / "timestamp_sanity_check.csv", index=False)
    print("\nTimestamp sanity check")
    print(sanity[sanity_cols].to_string(index=False))

    fixed_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    private_predictions: list[pd.DataFrame] = []

    for name, table in tables_public.items():
        print(f"\nFixed evaluation: {name}")
        rows, _, _ = evaluate_split(table, features_by_set[name], tab.fixed_split(), name, models)
        fixed_rows.extend(rows)
        for row in rows:
            print(f"{name} {row['model_name']} fixed CC={row['cc']:.3f} MAE={row['mae']:.2f}")

        if not args.skip_cv:
            for split in tab.cv_splits():
                rows, _, _ = evaluate_split(table, features_by_set[name], split, name, models)
                cv_rows.extend(rows)

    for name, table in tables_all.items():
        print(f"\nPrivate diagnostic: {name}")
        rows, pred_frame = evaluate_private(table, features_by_set[name], name, models)
        private_rows.extend(rows)
        private_predictions.append(pred_frame)
        for row in rows:
            print(f"{name} {row['model_name']} private CC={row['cc']:.3f} MAE={row['mae']:.2f}")

    fixed_df = pd.DataFrame(fixed_rows)
    cv_df = pd.DataFrame(cv_rows)
    private_df = pd.DataFrame(private_rows)
    summary_df = summarize(fixed_df, cv_df, private_df) if not cv_df.empty else summarize(fixed_df, pd.DataFrame(columns=["feature_set_name", "model_name", "mae", "rmse", "cc", "mae_skill_vs_27day"]), private_df)

    fixed_df.to_csv(OUT_DIR / "fixed_results.csv", index=False)
    cv_df.to_csv(OUT_DIR / "cv_results.csv", index=False)
    private_df.to_csv(OUT_DIR / "private_diagnostic.csv", index=False)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)

    # Select by public fixed ensemble CC if CV is skipped, otherwise by mean CV ensemble CC.
    if not cv_df.empty:
        public_select = (
            cv_df[cv_df["model_name"] == "ensemble_0p7_mlp_0p3_extratrees"]
            .groupby("feature_set_name")
            .agg(public_score=("cc", "mean"))
            .reset_index()
            .sort_values("public_score", ascending=False)
            .iloc[0]
        )
    else:
        public_select = (
            fixed_df[fixed_df["model_name"] == "ensemble_0p7_mlp_0p3_extratrees"]
            .sort_values("cc", ascending=False)
            .rename(columns={"cc": "public_score"})
            .iloc[0]
        )
    best_name = str(public_select["feature_set_name"])
    best_pred = pd.concat(private_predictions, ignore_index=True)
    best_pred = best_pred[best_pred["feature_set_name"] == best_name][["datetime", "predicted_speed"]]
    best_pred.to_csv(OUT_DIR / "best_private_prediction.csv", index=False)

    print("\nSummary")
    print(summary_df.sort_values(["model_name", "fixed_cc"], ascending=[True, False]).to_string(index=False))
    print(f"\nBest public-selected private prediction feature set: {best_name}")
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
