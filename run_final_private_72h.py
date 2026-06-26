"""Run the final selected public-only configuration on private data once.

Do not execute this script until the final public-only configuration has been
selected and the user explicitly requests private evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import train_tabular_models_72h as tab


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs" / "tabular_72h_private"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=tab.OUT_DIR / "selected_config.json")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    selected = config["selected"]
    model_name = selected["model_name"]
    target_type = selected["target_type"]
    weighted = selected["sample_weighting"] == "yes"

    table_all = tab.build_feature_table_including_private(tab.FULL_CSV)
    features = config["features"]
    public_idx = np.flatnonzero(((table_all["target_year"] >= 2011) & (table_all["target_year"] <= 2023)).to_numpy())
    private_idx = np.flatnonzero(((table_all["target_year"] >= 2024) & (table_all["target_year"] <= 2025)).to_numpy())

    configs = tab.candidate_models(config.get("preset", "initial"))
    model_cfg = next((cfg for cfg in configs if cfg["name"] == model_name), None)
    if model_cfg is None:
        raise RuntimeError(f"Selected model {model_name!r} is not available in candidate config list.")

    X_train = table_all.iloc[public_idx][features]
    X_private = table_all.iloc[private_idx][features]
    y_train = tab.adjusted_target(table_all, public_idx, target_type)
    weights = tab.target_weights(table_all.iloc[public_idx]["target_speed"].to_numpy(dtype=np.float32)) if weighted else None
    pred_model_target = tab.fit_predict(model_cfg["estimator"], X_train, y_train, X_private, weights)
    pred = tab.restore_target(table_all, private_idx, pred_model_target, target_type)

    y_private = table_all.iloc[private_idx]["target_speed"].to_numpy(dtype=np.float32)
    persistence = table_all.iloc[private_idx]["persistence_27day_target_aligned"].to_numpy(dtype=np.float32)
    result = {
        "selected_config": selected,
        "metrics": tab.metrics(y_private, pred, persistence),
        "private_evaluation": "run_once",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=True))
    tab.save_predictions(OUT_DIR / "private_predictions.csv", table_all, private_idx, pred, str(selected["model_id"]))
    print(json.dumps(result, indent=2, allow_nan=True))
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
