"""Freeze final representative-CH 72h ensemble outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_ch_feature_addition_72h as chrun
import train_tabular_models_72h as tab


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"

FINAL_FEATURE_SET = "current_plus_representative_mrmr_ch"
EXPANDED_FEATURE_SET = "current_plus_expanded_mrmr_ch"
MODEL_NAME = "ensemble_0p7_mlp_0p3_extratrees"


def score_submission(pred: pd.DataFrame, private: pd.DataFrame) -> dict[str, float]:
    merged = private[["datetime", "Speed (km/s)"]].merge(pred, on="datetime", how="left")
    finite = merged["Speed (km/s)"].notna()
    scored = merged[finite & merged["predicted_speed"].notna()]
    y = scored["Speed (km/s)"].to_numpy(dtype=float)
    p = scored["predicted_speed"].to_numpy(dtype=float)
    err = p - y
    return {
        "private_prediction_rows": int(len(pred)),
        "private_scored_finite_target_rows": int(len(scored)),
        "private_mae": float(np.mean(np.abs(err))),
        "private_rmse": float(np.sqrt(np.mean(err**2))),
        "private_cc": float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 1e-8 and np.std(p) > 1e-8 else float("nan"),
    }


def make_full_private_prediction(feature_set_name: str) -> pd.DataFrame:
    ch = chrun.load_ch()
    table_all = tab.build_feature_table(
        tab.FULL_CSV,
        min_target_year=2011,
        max_target_year=2025,
        require_finite_target=False,
        require_finite_persistence=False,
    )
    tables_all, features_by_set, sanity_all = chrun.feature_sets(table_all, ch)
    table = tables_all[feature_set_name]
    features = features_by_set[feature_set_name]
    models = chrun.model_configs()

    public_idx = np.flatnonzero(((table["target_year"] >= 2011) & (table["target_year"] <= 2023) & table["target_speed"].notna()).to_numpy())
    private_idx = np.flatnonzero(((table["target_year"] >= 2024) & (table["target_year"] <= 2025)).to_numpy())
    X_train = table.iloc[public_idx][features]
    y_train = table.iloc[public_idx]["target_speed"].to_numpy(dtype=np.float32)
    X_private = table.iloc[private_idx][features]

    mlp_pred = chrun.fit_predict(models["direct_mlp"], X_train, y_train, X_private)
    extra_pred = chrun.fit_predict(models["extratrees"], X_train, y_train, X_private)
    pred = 0.7 * mlp_pred + 0.3 * extra_pred

    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(table.iloc[private_idx]["target_datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_speed": pred,
        }
    )
    sanity = sanity_all.sample(n=min(10, len(sanity_all)), random_state=20260624)
    sanity.to_csv(OUT_DIR / "final_ch_representative_72h_sanity_check.csv", index=False)
    return out


def row_from_summary(summary: pd.DataFrame, feature_set_name: str, model_name: str) -> dict[str, float | str]:
    row = summary[(summary["feature_set_name"] == feature_set_name) & (summary["model_name"] == model_name)].iloc[0]
    return {
        "public_fixed_mae": float(row["fixed_mae"]),
        "public_fixed_rmse": float(row["fixed_rmse"]),
        "public_fixed_cc": float(row["fixed_cc"]),
        "public_cv_mean_mae": float(row["cv_mean_mae"]),
        "public_cv_mean_rmse": float(row["cv_mean_rmse"]),
        "public_cv_mean_cc": float(row["cv_mean_cc"]),
        "private_diagnostic_mae": float(row["private_mae"]),
        "private_diagnostic_rmse": float(row["private_rmse"]),
        "private_diagnostic_cc": float(row["private_cc"]),
    }


def main() -> None:
    summary = pd.read_csv(OUT_DIR / "ch_feature_addition_72h" / "summary.csv")
    private = pd.read_csv("solar_wind-private.csv", parse_dates=["datetime"])

    final_pred = make_full_private_prediction(FINAL_FEATURE_SET)
    final_path = OUT_DIR / "final_ch_representative_72h_ensemble.csv"
    final_pred.to_csv(final_path, index=False)

    old_pred = pd.read_csv(OUT_DIR / "final_private_72h_ensemble_full.csv", parse_dates=["datetime"])
    old_score = score_submission(old_pred, private)
    new_score = score_submission(pd.read_csv(final_path, parse_dates=["datetime"]), private)

    old_summary = pd.read_csv(OUT_DIR / "final_72h_freeze" / "final_comparison_table.csv")
    old_row = old_summary[old_summary["model"] == "ensemble 0.7 MLP + 0.3 ExtraTrees"].iloc[0]
    persistence_row = old_summary[old_summary["model"] == "27-day persistence"].iloc[0]

    comparison_rows = [
        {
            "model": "27-day persistence",
            "public_fixed_mae": float(persistence_row["public_fixed_mae"]),
            "public_fixed_rmse": float(persistence_row["public_fixed_rmse"]),
            "public_fixed_cc": float(persistence_row["public_fixed_cc"]),
            "public_cv_mean_mae": float(persistence_row["public_cv_mean_mae"]),
            "public_cv_mean_rmse": float(persistence_row["public_cv_mean_rmse"]),
            "public_cv_mean_cc": float(persistence_row["public_cv_mean_cc"]),
            "private_diagnostic_mae": float(persistence_row["private_mae"]),
            "private_diagnostic_rmse": float(persistence_row["private_rmse"]),
            "private_diagnostic_cc": float(persistence_row["private_cc"]),
            "private_prediction_rows": int(len(private)),
            "private_scored_finite_target_rows": int(private["Speed (km/s)"].notna().sum()),
        },
        {
            "model": "old final ensemble without new CH features",
            "public_fixed_mae": float(old_row["public_fixed_mae"]),
            "public_fixed_rmse": float(old_row["public_fixed_rmse"]),
            "public_fixed_cc": float(old_row["public_fixed_cc"]),
            "public_cv_mean_mae": None,
            "public_cv_mean_rmse": None,
            "public_cv_mean_cc": None,
            "private_diagnostic_mae": old_score["private_mae"],
            "private_diagnostic_rmse": old_score["private_rmse"],
            "private_diagnostic_cc": old_score["private_cc"],
            "private_prediction_rows": old_score["private_prediction_rows"],
            "private_scored_finite_target_rows": old_score["private_scored_finite_target_rows"],
        },
        {
            "model": "new representative CH ensemble",
            **row_from_summary(summary, FINAL_FEATURE_SET, MODEL_NAME),
            "private_prediction_rows": new_score["private_prediction_rows"],
            "private_scored_finite_target_rows": new_score["private_scored_finite_target_rows"],
        },
        {
            "model": "expanded CH ensemble",
            **row_from_summary(summary, EXPANDED_FEATURE_SET, MODEL_NAME),
            "private_prediction_rows": int(private["datetime"].nunique()),
            "private_scored_finite_target_rows": int(private["Speed (km/s)"].notna().sum()),
        },
    ]
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUT_DIR / "final_ch_representative_72h_comparison.csv", index=False)

    config = {
        "model": "0.7 Direct MLP + 0.3 ExtraTrees",
        "feature_set": FINAL_FEATURE_SET,
        "use_expanded_mrmr_ch": False,
        "weights": {"direct_mlp": 0.7, "extratrees": 0.3},
        "ch_matching": {
            "method": "merge_asof backward",
            "tolerance_hours": 12,
            "feature_specs": [
                {
                    "column": spec.column,
                    "target_lag_days": spec.target_lag_days,
                    "origin_offset_days": spec.origin_offset_days,
                }
                for spec in chrun.REPRESENTATIVE_CH
            ],
        },
        "selection_basis": "public fixed validation and expanding-window CV; private is diagnostic only",
        "final_prediction_file": str(final_path.relative_to(HERE)),
    }
    (OUT_DIR / "final_ch_representative_72h_config.json").write_text(json.dumps(config, indent=2))

    print("\nFinal CH representative comparison")
    print(comparison.to_string(index=False))
    print(f"\nSaved {final_path}")


if __name__ == "__main__":
    main()
