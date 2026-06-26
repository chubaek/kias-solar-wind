"""Recent-5-years to next-year 72h speed experiment.

This diagnostic uses the current official feature family:
current tabular features plus representative_mrmr_ch.  It intentionally avoids
magnetogram features, residual learning, and model search.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error

import run_ch_feature_addition_72h as chrun
import train_tabular_models_72h as tab


HERE = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = HERE / "outputs" / "recent5_predict1_72h"

FEATURE_SET_NAME = "current_plus_representative_mrmr_ch"
ENSEMBLE_NAME = "ensemble_0p7_mlp_0p3_extratrees"
ENSEMBLE_WEIGHTS = {"direct_mlp": 0.70, "extratrees": 0.30}

BASELINE_FILES = [
    (
        "current_official_best",
        HERE / "outputs" / "final_ch_representative_72h_ensemble.csv",
    ),
    (
        "full_public_representative_ch_ensemble",
        HERE / "outputs" / "ch_feature_addition_72h" / "best_private_prediction.csv",
    ),
]


def cc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) <= 1e-8 or np.std(y_pred) <= 1e-8:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, persistence: np.ndarray) -> dict[str, float]:
    score_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_score = y_true[score_mask]
    pred_score = y_pred[score_mask]
    persistence_score = persistence[score_mask]
    if len(y_score) == 0:
        return {
            "scored_rows": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "cc": float("nan"),
            "mae_skill_vs_27day": float("nan"),
        }

    mae = float(mean_absolute_error(y_score, pred_score))
    p_mask = np.isfinite(persistence_score)
    if p_mask.any():
        model_mae_for_skill = float(mean_absolute_error(y_score[p_mask], pred_score[p_mask]))
        p_mae = float(mean_absolute_error(y_score[p_mask], persistence_score[p_mask]))
        skill = float(1.0 - model_mae_for_skill / p_mae) if p_mae > 0 else float("nan")
    else:
        skill = float("nan")
    return {
        "scored_rows": int(len(y_score)),
        "mae": mae,
        "rmse": float(math.sqrt(mean_squared_error(y_score, pred_score))),
        "cc": cc_score(y_score, pred_score),
        "mae_skill_vs_27day": skill,
    }


def build_representative_ch_table() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    table = tab.build_feature_table(
        tab.FULL_CSV,
        min_target_year=2011,
        max_target_year=2025,
        require_finite_target=False,
        require_finite_persistence=False,
    )
    ch = chrun.load_ch()
    chrun.validate_specs(ch, chrun.REPRESENTATIVE_CH)
    table_ch, ch_cols, sanity = chrun.add_ch_features(table, ch, chrun.REPRESENTATIVE_CH)
    features = tab.feature_columns(table) + ch_cols
    return table_ch, features, sanity


def selected_models() -> dict[str, Any]:
    configs = chrun.model_configs()
    return {
        "direct_mlp": configs["direct_mlp"],
        "extratrees": configs["extratrees"],
    }


def fit_predict(estimator: Any, X_train: pd.DataFrame, y_train: np.ndarray, X_eval: pd.DataFrame) -> np.ndarray:
    model = clone(estimator)
    model.fit(X_train, y_train)
    return model.predict(X_eval)


def year_idx(table: pd.DataFrame, start_year: int, end_year: int, require_target: bool) -> np.ndarray:
    target_year = pd.to_datetime(table["target_datetime"]).dt.year
    mask = (target_year >= start_year) & (target_year <= end_year)
    if require_target:
        mask &= table["target_speed"].notna()
    return np.flatnonzero(mask.to_numpy())


def evaluate_prediction(
    rows: list[dict[str, Any]],
    model: str,
    train_years: str,
    predict_year: int,
    y_true: np.ndarray,
    pred: np.ndarray,
    persistence: np.ndarray,
    prediction_rows: int,
    source_file: str | None = None,
) -> None:
    rows.append(
        {
            "model": model,
            "train_years": train_years,
            "predict_year": int(predict_year),
            **metric_dict(y_true, pred, persistence),
            "prediction_rows": int(prediction_rows),
            "source_file": source_file,
        }
    )


def align_prediction_file(path: Path, target_datetimes: pd.Series) -> tuple[np.ndarray, int, int]:
    pred = pd.read_csv(path, parse_dates=["datetime"])
    pred = pred[["datetime", "predicted_speed"]].drop_duplicates("datetime", keep="last")
    frame = pd.DataFrame({"datetime": pd.to_datetime(target_datetimes)})
    merged = frame.merge(pred, on="datetime", how="left")
    values = merged["predicted_speed"].to_numpy(dtype=np.float32)
    matched_rows = int(np.isfinite(values).sum())
    return values, matched_rows, int(len(pred))


def timestamp_sanity_for_prediction_year(ch_sanity: pd.DataFrame, predict_idx: np.ndarray) -> pd.DataFrame:
    sanity = ch_sanity[ch_sanity["_row_id"].isin(set(map(int, predict_idx)))].copy()
    if sanity.empty:
        return sanity
    sanity["target_year"] = pd.to_datetime(sanity["target_datetime"]).dt.year
    cols = [
        "_row_id",
        "origin_datetime",
        "target_datetime",
        "target_year",
        "feature",
        "target_lag_days",
        "origin_offset_days",
        "requested_ch_time",
        "ch_time",
        "matched_ch_time_le_requested",
        "requested_time_le_origin",
        "target_is_origin_plus_72h",
    ]
    return sanity[cols].sort_values(["target_datetime", "feature"]).reset_index(drop=True)


def run_one_window(
    table: pd.DataFrame,
    features: list[str],
    models: dict[str, Any],
    train_start_year: int,
    train_end_year: int,
    predict_year: int,
    baseline_files: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    train_idx = year_idx(table, train_start_year, train_end_year, require_target=True)
    predict_idx = year_idx(table, predict_year, predict_year, require_target=False)
    if len(train_idx) == 0:
        raise RuntimeError(f"No training rows for target years {train_start_year}-{train_end_year}.")
    if len(predict_idx) == 0:
        raise RuntimeError(f"No prediction rows for target year {predict_year}.")

    X_train = table.iloc[train_idx][features]
    y_train = table.iloc[train_idx]["target_speed"].to_numpy(dtype=np.float32)
    X_predict = table.iloc[predict_idx][features]
    y_eval = table.iloc[predict_idx]["target_speed"].to_numpy(dtype=np.float32)
    persistence = table.iloc[predict_idx]["persistence_27day_target_aligned"].to_numpy(dtype=np.float32)
    train_years = f"{train_start_year}-{train_end_year}"

    preds = {
        "direct_mlp": fit_predict(models["direct_mlp"], X_train, y_train, X_predict),
        "extratrees": fit_predict(models["extratrees"], X_train, y_train, X_predict),
    }
    preds[ENSEMBLE_NAME] = (
        ENSEMBLE_WEIGHTS["direct_mlp"] * preds["direct_mlp"]
        + ENSEMBLE_WEIGHTS["extratrees"] * preds["extratrees"]
    )

    rows: list[dict[str, Any]] = []
    for name in ["direct_mlp", "extratrees", ENSEMBLE_NAME]:
        evaluate_prediction(
            rows,
            name,
            train_years,
            predict_year,
            y_eval,
            preds[name],
            persistence,
            len(predict_idx),
        )

    evaluate_prediction(
        rows,
        "persistence_27day",
        "baseline",
        predict_year,
        y_eval,
        persistence,
        persistence,
        int(np.isfinite(persistence).sum()),
    )

    if baseline_files:
        target_datetimes = table.iloc[predict_idx]["target_datetime"]
        seen_paths: set[Path] = set()
        for model_name, path in BASELINE_FILES:
            if not path.exists() or path in seen_paths:
                continue
            seen_paths.add(path)
            pred_values, matched_rows, source_rows = align_prediction_file(path, target_datetimes)
            evaluate_prediction(
                rows,
                model_name,
                "external_file",
                predict_year,
                y_eval,
                pred_values,
                persistence,
                matched_rows,
                str(path.relative_to(HERE)),
            )
            rows[-1]["source_file_rows"] = source_rows

    prediction_frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(table.iloc[predict_idx]["target_datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_speed": preds[ENSEMBLE_NAME],
        }
    )
    return pd.DataFrame(rows), prediction_frame, preds, predict_idx


def print_metrics_table(metrics: pd.DataFrame) -> None:
    cols = [
        "model",
        "train_years",
        "predict_year",
        "mae",
        "rmse",
        "cc",
        "mae_skill_vs_27day",
        "scored_rows",
        "prediction_rows",
    ]
    display = metrics[cols].copy()
    rename = {
        "mae": "MAE",
        "rmse": "RMSE",
        "cc": "CC",
        "mae_skill_vs_27day": "MAE Skill vs 27-day persistence",
    }
    display = display.rename(columns=rename)
    print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def save_single_outputs(
    out_dir: Path,
    table: pd.DataFrame,
    features: list[str],
    ch_sanity: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    predict_idx: np.ndarray,
    train_start_year: int,
    train_end_year: int,
    predict_year: int,
) -> None:
    stem = f"single_{train_start_year}_{train_end_year}_to_{predict_year}"
    metrics.to_csv(out_dir / f"{stem}_metrics.csv", index=False)
    predictions.to_csv(out_dir / f"{stem}_predictions.csv", index=False)
    timestamp_sanity_for_prediction_year(ch_sanity, predict_idx).to_csv(
        out_dir / f"{stem}_timestamp_sanity.csv",
        index=False,
    )
    train_idx = year_idx(table, train_start_year, train_end_year, require_target=True)
    config = {
        "target": "Speed(t + 72h)",
        "split_key": "target_datetime year",
        "train_start_year": train_start_year,
        "train_end_year": train_end_year,
        "predict_year": predict_year,
        "feature_set": FEATURE_SET_NAME,
        "feature_count": len(features),
        "models": {
            "direct_mlp": "selected TorchMLPRegressor from run_ch_feature_addition_72h.model_configs",
            "extratrees": "selected ExtraTreesRegressor from run_ch_feature_addition_72h.model_configs",
            ENSEMBLE_NAME: ENSEMBLE_WEIGHTS,
        },
        "row_counts": {
            "train_rows_finite_target": int(len(train_idx)),
            "prediction_rows": int(len(predictions)),
            "finite_target_prediction_rows": int(table.iloc[predict_idx]["target_speed"].notna().sum()),
        },
        "data_rules": {
            "magnetogram_features": "not_used",
            "residual_learning": "not_used",
            "input_nan_rows": "kept; model pipelines use train-fitted imputers",
            "scaling": "train-fitted StandardScaler only where the selected pipeline includes scaling",
            "evaluation_drop_rule": "metrics score rows with finite target and finite prediction",
        },
        "ch_feature_specs": [
            {
                "column": spec.column,
                "target_lag_days": spec.target_lag_days,
                "origin_offset_days": spec.origin_offset_days,
            }
            for spec in chrun.REPRESENTATIVE_CH
        ],
    }
    (out_dir / f"{stem}_config.json").write_text(json.dumps(config, indent=2, allow_nan=True))


def should_run_rolling(single_metrics: pd.DataFrame) -> tuple[bool, str]:
    recent = single_metrics[single_metrics["model"] == ENSEMBLE_NAME]
    official = single_metrics[single_metrics["model"] == "current_official_best"]
    if recent.empty:
        return False, "recent ensemble metric missing"
    if official.empty or not np.isfinite(float(official.iloc[0]["mae"])):
        return False, "current official best prediction file metric unavailable"
    recent_mae = float(recent.iloc[0]["mae"])
    official_mae = float(official.iloc[0]["mae"])
    if recent_mae < official_mae:
        return True, f"recent ensemble MAE {recent_mae:.4f} improved over official MAE {official_mae:.4f}"
    return False, f"recent ensemble MAE {recent_mae:.4f} did not improve over official MAE {official_mae:.4f}"


def run_rolling(
    table: pd.DataFrame,
    features: list[str],
    models: dict[str, Any],
    out_dir: Path,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for predict_year in range(2016, 2026):
        train_start = predict_year - 5
        train_end = predict_year - 1
        include_external = predict_year >= 2024
        print(f"\nRolling window {train_start}-{train_end} -> {predict_year}")
        metrics, _, _, _ = run_one_window(
            table,
            features,
            models,
            train_start,
            train_end,
            predict_year,
            baseline_files=include_external,
        )
        rows.append(metrics)
        print_metrics_table(metrics)

    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(out_dir / "rolling_5yr_to_1yr_summary.csv", index=False)

    ensemble = summary[summary["model"] == ENSEMBLE_NAME]
    public = ensemble[(ensemble["predict_year"] >= 2016) & (ensemble["predict_year"] <= 2023)]
    diagnostics = ensemble[ensemble["predict_year"].isin([2024, 2025])]
    print("\nRolling public mean, ensemble only (2016-2023)")
    print(
        public[["mae", "rmse", "cc"]]
        .mean(numeric_only=True)
        .rename({"mae": "MAE", "rmse": "RMSE", "cc": "CC"})
        .to_string(float_format=lambda x: f"{x:.4f}")
    )
    print("\nRolling diagnostics, ensemble only (2024-2025)")
    print_metrics_table(diagnostics)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-start-year", type=int, default=2019)
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--predict-year", type=int, default=2024)
    parser.add_argument("--run-single", action="store_true")
    parser.add_argument("--run-rolling", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = HERE / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    table, features, ch_sanity = build_representative_ch_table()
    models = selected_models()

    print(
        f"feature_set={FEATURE_SET_NAME} rows={len(table)} features={len(features)} "
        f"output_dir={out_dir.relative_to(HERE)}"
    )

    run_single = args.run_single or not args.run_rolling
    if run_single or args.run_rolling:
        print(f"\nSingle experiment {args.train_start_year}-{args.train_end_year} -> {args.predict_year}")
        metrics, predictions, _, predict_idx = run_one_window(
            table,
            features,
            models,
            args.train_start_year,
            args.train_end_year,
            args.predict_year,
            baseline_files=True,
        )
        save_single_outputs(
            out_dir,
            table,
            features,
            ch_sanity,
            metrics,
            predictions,
            predict_idx,
            args.train_start_year,
            args.train_end_year,
            args.predict_year,
        )
        print_metrics_table(metrics)

        if args.run_rolling:
            do_rolling, reason = should_run_rolling(metrics)
            print(f"\nRolling gate: {reason}")
            if do_rolling:
                run_rolling(table, features, models, out_dir)
            else:
                print("Rolling experiment skipped by selection rule.")

    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
