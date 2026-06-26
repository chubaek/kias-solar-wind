"""Inspect public tabular results and run requested private diagnostics.

The private diagnostics here are explicitly requested by .order. Model/ensemble
selection is done from public fixed validation or public CV only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone

import train_first_try_72h as direct
import train_tabular_models_72h as tab


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs" / "tabular_72h_diagnostics"

DIRECT_SEEDS = [11, 42, 77, 101, 123]


def direct_persistence(samples: direct.Samples) -> np.ndarray:
    return direct.persistence_predictions(samples)


def train_direct_ensemble(
    train: direct.Samples,
    eval_samples: direct.Samples,
    epochs: int,
    selected_epoch: int | None,
    seeds: list[int],
    hidden: int,
    dropout: float,
    lr: float,
    batch_size: int,
    patience: int,
    device: torch.device,
) -> tuple[np.ndarray, list[int], dict[str, Any]]:
    eval_preds: list[np.ndarray] = []
    selected_epochs: list[int] = []
    seed_metrics: dict[str, Any] = {}

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        prep = direct.fit_preprocessor(train)
        x_train = direct.transform_x(train, prep)
        y_train = direct.transform_y(train, prep)
        x_eval = direct.transform_x(eval_samples, prep)

        loader = direct.make_loader(x_train, y_train, batch_size, shuffle=True)
        model = direct.MLP(x_train.shape[1], hidden, dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = torch.nn.SmoothL1Loss()

        best_cc = -float("inf")
        best_state = None
        best_epoch = selected_epoch or epochs
        stale = 0
        target_epochs = selected_epoch or epochs

        for epoch in range(1, target_epochs + 1):
            model.train()
            losses = []
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            pred = direct.predict(model, x_eval, prep, device)
            m = direct.metrics(eval_samples.y, pred, direct_persistence(eval_samples))
            cc = float(m["cc"])
            if selected_epoch is None:
                if cc > best_cc + 1e-5:
                    best_cc = cc
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        break

        if selected_epoch is None and best_state is not None:
            model.load_state_dict(best_state)

        pred = direct.predict(model, x_eval, prep, device)
        eval_preds.append(pred)
        selected_epochs.append(best_epoch)
        seed_metrics[str(seed)] = {
            "selected_epoch": int(best_epoch),
            **direct.metrics(eval_samples.y, pred, direct_persistence(eval_samples)),
        }
        print(f"direct_mlp seed={seed} selected_epoch={best_epoch} eval_CC={seed_metrics[str(seed)]['cc']:.3f}")

    return np.mean(eval_preds, axis=0), selected_epochs, seed_metrics


def cv_summary(cv: pd.DataFrame) -> pd.DataFrame:
    return (
        cv[cv["model_id"].notna()]
        .groupby(["model_id", "model_name", "target_type", "sample_weighting"], dropna=False)
        .agg(
            mean_cv_cc=("cc", "mean"),
            mean_cv_mae=("mae", "mean"),
            mean_cv_rmse=("rmse", "mean"),
            std_cv_cc=("cc", "std"),
            folds=("fold", "count"),
        )
        .reset_index()
        .sort_values(["mean_cv_cc", "mean_cv_mae"], ascending=[False, True])
    )


def find_selected_estimator(config: dict[str, Any]) -> dict[str, Any]:
    selected = config["selected"]
    configs = tab.candidate_models(config.get("preset", "initial"))
    for cfg in configs:
        if cfg["name"] == selected["model_name"]:
            return cfg
    raise RuntimeError(f"Could not find estimator for {selected['model_name']}")


def train_tabular_predict(
    cfg: dict[str, Any],
    table: pd.DataFrame,
    features: list[str],
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    target_type: str,
    weighted: bool,
) -> np.ndarray:
    model = clone(cfg["estimator"])
    X_train = table.iloc[train_idx][features]
    X_eval = table.iloc[eval_idx][features]
    y_train = tab.adjusted_target(table, train_idx, target_type)
    sample_weight = (
        tab.target_weights(table.iloc[train_idx]["target_speed"].to_numpy(dtype=np.float32))
        if weighted
        else None
    )
    if sample_weight is not None:
        model.fit(X_train, y_train, model__sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    pred = model.predict(X_eval)
    return tab.restore_target(table, eval_idx, pred, target_type)


def search_three_way_weights(y: np.ndarray, direct_pred: np.ndarray, extra_pred: np.ndarray, persistence: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for wd in np.arange(0.0, 1.0001, 0.05):
        for we in np.arange(0.0, 1.0001 - wd, 0.05):
            wp = round(1.0 - wd - we, 10)
            if wp < -1e-9:
                continue
            pred = wd * direct_pred + we * extra_pred + wp * persistence
            m = tab.metrics(y, pred, persistence)
            candidate = {"direct_mlp": float(wd), "extratrees": float(we), "persistence_27day": float(wp), **m}
            if best is None or candidate["cc"] > best["cc"]:
                best = candidate
    assert best is not None
    return best


def add_row(rows: list[dict[str, Any]], name: str, public_fixed: dict[str, float], cv: dict[str, float] | None, private: dict[str, float]) -> None:
    rows.append(
        {
            "model": name,
            "public_fixed_mae": public_fixed.get("mae"),
            "public_fixed_rmse": public_fixed.get("rmse"),
            "public_fixed_cc": public_fixed.get("cc"),
            "public_cv_mean_mae": None if cv is None else cv.get("mean_cv_mae"),
            "public_cv_mean_rmse": None if cv is None else cv.get("mean_cv_rmse"),
            "public_cv_mean_cc": None if cv is None else cv.get("mean_cv_cc"),
            "private_mae": private.get("mae"),
            "private_rmse": private.get("rmse"),
            "private_cc": private.get("cc"),
            "private_mae_skill_vs_27day": private.get("mae_skill_vs_27day"),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-epochs", type=int, default=40)
    parser.add_argument("--direct-hidden", type=int, default=128)
    parser.add_argument("--direct-dropout", type=float, default=0.1)
    parser.add_argument("--direct-lr", type=float, default=1e-3)
    parser.add_argument("--direct-batch-size", type=int, default=512)
    parser.add_argument("--direct-patience", type=int, default=7)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cv = pd.read_csv(tab.OUT_DIR / "cv_results.csv")
    fixed = pd.read_csv(tab.OUT_DIR / "fixed_results.csv")
    config = json.loads((tab.OUT_DIR / "selected_config.json").read_text())
    selected_id = config["selected"]["model_id"]

    summary = cv_summary(cv)
    top_cv = summary.head(20)
    top_fixed = (
        fixed[fixed["model_id"].notna()]
        .sort_values(["cc", "mae"], ascending=[False, True])
        .head(20)
    )
    selected_folds = cv[cv["model_id"] == selected_id].sort_values("fold")

    print("\nTop 20 configs by mean CV CC")
    print(top_cv.to_string(index=False))
    print("\nTop 20 configs by fixed 2022-2023 CC")
    print(top_fixed[["model_id", "mae", "rmse", "cc", "mae_skill_vs_27day"]].to_string(index=False))
    print("\nSelected ExtraTrees fold-by-fold")
    print(selected_folds[["fold", "mae", "rmse", "cc", "mae_skill_vs_27day"]].to_string(index=False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\ndevice={device}")

    timestamps, data = direct.load_csv(direct.FULL_CSV)
    direct_train_fixed = direct.build_samples(timestamps, data, 2011, 2021, daily_origins=False)
    direct_val_fixed = direct.build_samples(timestamps, data, 2022, 2023, daily_origins=False)
    direct_train_public = direct.build_samples(timestamps, data, 2011, 2023, daily_origins=False)
    direct_private = direct.build_samples(timestamps, data, 2024, 2025, daily_origins=False)

    direct_fixed_pred, direct_epochs, direct_seed_metrics = train_direct_ensemble(
        direct_train_fixed,
        direct_val_fixed,
        args.direct_epochs,
        selected_epoch=None,
        seeds=DIRECT_SEEDS,
        hidden=args.direct_hidden,
        dropout=args.direct_dropout,
        lr=args.direct_lr,
        batch_size=args.direct_batch_size,
        patience=args.direct_patience,
        device=device,
    )
    direct_fixed_metrics = direct.metrics(direct_val_fixed.y, direct_fixed_pred, direct_persistence(direct_val_fixed))
    selected_direct_epoch = int(round(float(np.median(direct_epochs))))
    print("\nPrevious direct MLP reproduced on fixed 2022-2023")
    print({"selected_epoch_median": selected_direct_epoch, **direct_fixed_metrics})

    selected_fixed = fixed[fixed["model_id"] == selected_id].iloc[0].to_dict()
    extra_worse_than_direct = bool(selected_fixed["cc"] < direct_fixed_metrics["cc"])
    print(f"\nSelected ExtraTrees fixed CC worse than reproduced direct MLP? {extra_worse_than_direct}")

    table_public = tab.build_feature_table(tab.FULL_CSV)
    features = config["features"]
    cfg = find_selected_estimator(config)
    target_type = config["selected"]["target_type"]
    weighted = config["selected"]["sample_weighting"] == "yes"
    train_idx, val_idx = tab.split_rows(table_public, tab.fixed_split())
    extra_fixed_pred = train_tabular_predict(cfg, table_public, features, train_idx, val_idx, target_type, weighted)
    y_fixed = table_public.iloc[val_idx]["target_speed"].to_numpy(dtype=np.float32)
    persistence_fixed = table_public.iloc[val_idx]["persistence_27day_target_aligned"].to_numpy(dtype=np.float32)

    # Align direct predictions to the same target timestamps used by the tabular feature table.
    direct_fixed_frame = pd.DataFrame(
        {
            "target_datetime": pd.to_datetime([str(x).replace("T", " ") + ":30:00" for x in direct_val_fixed.target_times]),
            "direct_pred": direct_fixed_pred,
        }
    )
    fixed_frame = table_public.iloc[val_idx][["target_datetime"]].copy()
    fixed_frame["target_datetime"] = pd.to_datetime(fixed_frame["target_datetime"])
    aligned_direct = fixed_frame.merge(direct_fixed_frame, on="target_datetime", how="left")["direct_pred"].to_numpy(dtype=np.float32)
    mask = np.isfinite(aligned_direct)

    ensemble_public = search_three_way_weights(
        y_fixed[mask],
        aligned_direct[mask],
        extra_fixed_pred[mask],
        persistence_fixed[mask],
    )
    ensemble_fixed_pred = (
        ensemble_public["direct_mlp"] * aligned_direct
        + ensemble_public["extratrees"] * extra_fixed_pred
        + ensemble_public["persistence_27day"] * persistence_fixed
    )
    ensemble_fixed_metrics = tab.metrics(y_fixed, ensemble_fixed_pred, persistence_fixed)
    print("\nBest fixed-public ensemble weights")
    print(ensemble_public)

    table_all = tab.build_feature_table_including_private(tab.FULL_CSV)
    public_idx = np.flatnonzero(((table_all["target_year"] >= 2011) & (table_all["target_year"] <= 2023)).to_numpy())
    private_idx = np.flatnonzero(((table_all["target_year"] >= 2024) & (table_all["target_year"] <= 2025)).to_numpy())
    extra_private_pred = train_tabular_predict(cfg, table_all, features, public_idx, private_idx, target_type, weighted)
    y_private = table_all.iloc[private_idx]["target_speed"].to_numpy(dtype=np.float32)
    persistence_private = table_all.iloc[private_idx]["persistence_27day_target_aligned"].to_numpy(dtype=np.float32)
    extra_private_metrics = tab.metrics(y_private, extra_private_pred, persistence_private)
    persistence_private_metrics = tab.metrics(y_private, persistence_private, persistence_private)

    direct_private_pred, _, direct_private_seed_metrics = train_direct_ensemble(
        direct_train_public,
        direct_private,
        selected_direct_epoch,
        selected_epoch=selected_direct_epoch,
        seeds=DIRECT_SEEDS,
        hidden=args.direct_hidden,
        dropout=args.direct_dropout,
        lr=args.direct_lr,
        batch_size=args.direct_batch_size,
        patience=args.direct_patience,
        device=device,
    )
    direct_private_metrics = direct.metrics(direct_private.y, direct_private_pred, direct_persistence(direct_private))

    direct_private_frame = pd.DataFrame(
        {
            "target_datetime": pd.to_datetime([str(x).replace("T", " ") + ":30:00" for x in direct_private.target_times]),
            "direct_pred": direct_private_pred,
        }
    )
    private_frame = table_all.iloc[private_idx][["target_datetime"]].copy()
    private_frame["target_datetime"] = pd.to_datetime(private_frame["target_datetime"])
    aligned_private_direct = private_frame.merge(direct_private_frame, on="target_datetime", how="left")["direct_pred"].to_numpy(dtype=np.float32)
    ensemble_private_pred = (
        ensemble_public["direct_mlp"] * aligned_private_direct
        + ensemble_public["extratrees"] * extra_private_pred
        + ensemble_public["persistence_27day"] * persistence_private
    )
    ensemble_private_metrics = tab.metrics(y_private, ensemble_private_pred, persistence_private)

    selected_cv = summary[summary["model_id"] == selected_id].iloc[0].to_dict()
    direct_cv_proxy = None
    ensemble_cv_proxy = None
    persistence_fixed_metrics = tab.metrics(y_fixed, persistence_fixed, persistence_fixed)

    final_rows: list[dict[str, Any]] = []
    add_row(final_rows, "previous_direct_mlp_reproduced", direct_fixed_metrics, direct_cv_proxy, direct_private_metrics)
    add_row(final_rows, "selected_extratrees", selected_fixed, selected_cv, extra_private_metrics)
    add_row(final_rows, "ensemble_public_fixed_selected", ensemble_fixed_metrics, ensemble_cv_proxy, ensemble_private_metrics)
    add_row(final_rows, "persistence_27day", persistence_fixed_metrics, None, persistence_private_metrics)
    final = pd.DataFrame(final_rows)

    print("\nFinal comparison")
    print(final.to_string(index=False))

    top_cv.to_csv(OUT_DIR / "top20_cv_cc.csv", index=False)
    top_fixed.to_csv(OUT_DIR / "top20_fixed_cc.csv", index=False)
    selected_folds.to_csv(OUT_DIR / "selected_extratrees_folds.csv", index=False)
    final.to_csv(OUT_DIR / "final_comparison.csv", index=False)
    diagnostic = {
        "direct_seed_metrics_fixed": direct_seed_metrics,
        "direct_seed_metrics_private": direct_private_seed_metrics,
        "selected_direct_epoch": selected_direct_epoch,
        "selected_extratrees_worse_than_direct_fixed_cc": extra_worse_than_direct,
        "ensemble_weights_selected_on_fixed_public": ensemble_public,
    }
    (OUT_DIR / "diagnostic_summary.json").write_text(json.dumps(diagnostic, indent=2, allow_nan=True))
    print(f"\nSaved diagnostics to {OUT_DIR}")


if __name__ == "__main__":
    main()
