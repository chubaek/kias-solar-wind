"""Public-only tabular experiments for 72-hour solar-wind speed forecasts.

This script intentionally avoids private 2024-2025 data. It builds causal
features at forecast origin t and evaluates candidate models by fixed public
validation and expanding-window public CV.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
FULL_CSV = HERE / "solar_wind_data.csv"
OUT_DIR = HERE / "outputs" / "tabular_72h"
PRED_DIR = OUT_DIR / "validation_predictions"

HORIZON_HOURS = 72
TARGET = "Speed (km/s)"
BASE_COLUMNS = [
    "Speed (km/s)",
    "Density (1/cm^3)",
    "Temperature (K)",
    "B (nT)",
    "Sunspot Number",
    "Coronal Hole Area",
]
PERSISTENCE_LAG = 648 - HORIZON_HOURS


@dataclass
class Split:
    scheme: str
    fold: str
    train_start: int
    train_end: int
    val_start: int
    val_end: int


def cc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) <= 1e-8 or np.std(y_pred) <= 1e-8:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def metrics(y_true: np.ndarray, y_pred: np.ndarray, persistence: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(persistence)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    persistence = persistence[mask]
    p_mae = float(mean_absolute_error(y_true, persistence))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "n": int(len(y_true)),
        "mae": mae,
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "bias": float(np.mean(y_pred - y_true)),
        "cc": cc_score(y_true, y_pred),
        "mae_skill_vs_27day": float(1.0 - mae / p_mae) if p_mae > 0 else float("nan"),
    }


def target_weights(y: np.ndarray) -> np.ndarray:
    weights = np.ones(len(y), dtype=np.float32)
    weights[(y >= 500) & (y < 600)] = 2.0
    weights[y >= 600] = 3.0
    return weights


def add_lag_features(frame: pd.DataFrame, features: dict[str, pd.Series], col: str, lags: list[int]) -> None:
    safe = col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    for lag in lags:
        features[f"{safe}_lag_{lag}h"] = frame[col].shift(lag)


def add_roll_features(
    frame: pd.DataFrame,
    features: dict[str, pd.Series],
    col: str,
    means: list[int],
    stds: list[int],
    minmax: list[int] | None = None,
) -> None:
    safe = col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    for window in means:
        features[f"{safe}_roll_mean_{window}h"] = frame[col].rolling(window, min_periods=max(2, window // 4)).mean()
    for window in stds:
        features[f"{safe}_roll_std_{window}h"] = frame[col].rolling(window, min_periods=max(2, window // 4)).std()
    for window in minmax or []:
        roll = frame[col].rolling(window, min_periods=max(2, window // 4))
        features[f"{safe}_roll_min_{window}h"] = roll.min()
        features[f"{safe}_roll_max_{window}h"] = roll.max()


def build_feature_table(
    csv_path: Path,
    min_target_year: int = 2011,
    max_target_year: int = 2023,
    require_finite_target: bool = True,
    require_finite_persistence: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    features: dict[str, pd.Series] = {}
    speed = df[TARGET]

    features["origin_datetime"] = df["datetime"]
    features["target_datetime"] = df["datetime"].shift(-HORIZON_HOURS)
    features["target_speed"] = speed.shift(-HORIZON_HOURS)
    features["persistence_27day_target_aligned"] = speed.shift(PERSISTENCE_LAG)
    features["speed_current"] = speed

    add_lag_features(df, features, TARGET, [1, 3, 6, 12, 24, 48, 72, 96, 168])

    for center in [648, PERSISTENCE_LAG]:
        label = "source_surface_648h" if center == 648 else "target_aligned_576h"
        features[f"speed_recurrence_{label}"] = speed.shift(center)
        for offset in [6, 12, 24, 48]:
            features[f"speed_recurrence_{label}_minus_{offset}h"] = speed.shift(center + offset)
            features[f"speed_recurrence_{label}_plus_{offset}h"] = speed.shift(center - offset)

    add_roll_features(df, features, TARGET, [6, 12, 24, 72, 168], [24, 72, 168], [24, 72])
    features["speed_trend_current_minus_24h"] = speed - speed.shift(24)
    features["speed_trend_current_minus_roll24"] = speed - features["Speed_km_s_roll_mean_24h"]
    features["speed_trend_roll24_minus_roll72"] = (
        features["Speed_km_s_roll_mean_24h"] - features["Speed_km_s_roll_mean_72h"]
    )

    for col in ["Density (1/cm^3)", "Temperature (K)", "B (nT)"]:
        add_lag_features(df, features, col, [1, 6, 24, 72])
        add_roll_features(df, features, col, [24, 72], [24, 72])

    ch = df["Coronal Hole Area"]
    features["coronal_hole_area_current"] = ch
    add_lag_features(df, features, "Coronal Hole Area", [24, 48, 72, 96, 120, 144, 168])
    for window in [72, 120, 168]:
        roll = ch.rolling(window, min_periods=max(2, window // 4))
        features[f"coronal_hole_area_roll_mean_{window}h"] = roll.mean()
        features[f"coronal_hole_area_roll_max_{window}h"] = roll.max()
    features["coronal_hole_area_trend_3d_minus_5d"] = ch.shift(72) - ch.shift(120)
    features["coronal_hole_area_trend_2d_minus_6d"] = ch.shift(48) - ch.shift(144)

    sunspot = df["Sunspot Number"]
    features["sunspot_current"] = sunspot
    features["sunspot_lag_648h"] = sunspot.shift(648)
    features["sunspot_roll_mean_27d"] = sunspot.rolling(648, min_periods=72).mean()

    dt = df["datetime"]
    day_angle = 2.0 * np.pi * (dt.dt.dayofyear - 1) / 365.25
    month_angle = 2.0 * np.pi * (dt.dt.month - 1) / 12.0
    features["day_of_year_sin"] = np.sin(day_angle)
    features["day_of_year_cos"] = np.cos(day_angle)
    features["month_sin"] = np.sin(month_angle)
    features["month_cos"] = np.cos(month_angle)

    table = pd.DataFrame(features)
    table["target_year"] = table["target_datetime"].dt.year
    table["anomaly_base_27d_origin"] = speed.rolling(648, min_periods=72).mean()
    mask = (table["target_year"] >= min_target_year) & (table["target_year"] <= max_target_year)
    if require_finite_target:
        mask &= table["target_speed"].notna()
    if require_finite_persistence:
        mask &= table["persistence_27day_target_aligned"].notna()
    table = table[mask].reset_index(drop=True)
    return table


def build_feature_table_including_private(csv_path: Path) -> pd.DataFrame:
    return build_feature_table(csv_path, min_target_year=2011, max_target_year=2025)


def feature_columns(table: pd.DataFrame) -> list[str]:
    excluded = {
        "origin_datetime",
        "target_datetime",
        "target_speed",
        "target_year",
        "anomaly_base_27d_origin",
    }
    return [c for c in table.columns if c not in excluded]


def split_rows(table: pd.DataFrame, split: Split) -> tuple[np.ndarray, np.ndarray]:
    train = (table["target_year"] >= split.train_start) & (table["target_year"] <= split.train_end)
    val = (table["target_year"] >= split.val_start) & (table["target_year"] <= split.val_end)
    return np.flatnonzero(train.to_numpy()), np.flatnonzero(val.to_numpy())


def fixed_split() -> Split:
    return Split("fixed_2022_2023", "fixed", 2011, 2021, 2022, 2023)


def cv_splits() -> list[Split]:
    return [
        Split("expanding_cv", f"val_{year}", 2011, year - 1, year, year)
        for year in range(2017, 2024)
    ]


def make_pipeline(model: BaseEstimator, scaler: bool, impute: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if impute:
        steps.append(("imputer", SimpleImputer(strategy="median", add_indicator=True)))
    if scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 80,
        batch_size: int = 1024,
        patience: int = 8,
        random_state: int = 42,
    ) -> None:
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "TorchMLPRegressor":
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        self.y_mean_ = float(np.mean(y))
        self.y_std_ = float(np.std(y)) if float(np.std(y)) > 1e-6 else 1.0
        y_scaled = ((y - self.y_mean_) / self.y_std_).astype(np.float32)

        x = X.astype(np.float32)
        y_t = y_scaled.astype(np.float32)
        if sample_weight is None:
            w_t = np.ones(len(y_t), dtype=np.float32)
        else:
            w_t = sample_weight.astype(np.float32)

        n_val = max(256, int(0.15 * len(x)))
        train_x, val_x = x[:-n_val], x[-n_val:]
        train_y, val_y = y_t[:-n_val], y_t[-n_val:]
        train_w = w_t[:-n_val]

        layers: list[nn.Module] = []
        in_dim = x.shape[1]
        for hidden in self.hidden_sizes:
            layers += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(self.dropout)]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = nn.Sequential(*layers).to(self.device_)

        ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(train_w))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        val_x_t = torch.from_numpy(val_x).to(self.device_)
        best_cc = -float("inf")
        best_state = None
        stale = 0
        for _ in range(self.epochs):
            self.model_.train()
            for bx, by, bw in loader:
                bx, by, bw = bx.to(self.device_), by.to(self.device_), bw.to(self.device_)
                opt.zero_grad(set_to_none=True)
                pred = self.model_(bx).squeeze(-1)
                loss = (torch.nn.functional.smooth_l1_loss(pred, by, reduction="none") * bw).mean()
                loss.backward()
                opt.step()

            pred = self._predict_scaled_tensor(val_x_t)
            cc = cc_score(val_y, pred)
            if cc > best_cc + 1e-5:
                best_cc = cc
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def _predict_scaled_tensor(self, x_t: Any) -> np.ndarray:
        import torch

        self.model_.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x_t), 8192):
                pred = self.model_(x_t[start : start + 8192]).squeeze(-1).detach().cpu().numpy()
                preds.append(pred)
        return np.concatenate(preds)

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        x = X.astype(np.float32)
        preds: list[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(x), 8192):
                bx = torch.from_numpy(x[start : start + 8192]).to(self.device_)
                preds.append(self.model_(bx).squeeze(-1).detach().cpu().numpy())
        return np.concatenate(preds) * self.y_std_ + self.y_mean_


def candidate_models(preset: str) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []

    for alpha in ([0.1, 1.0, 10.0, 100.0, 1000.0] if preset == "full" else [10.0, 100.0, 1000.0]):
        configs.append(
            {
                "name": f"ridge_alpha_{alpha:g}",
                "estimator": make_pipeline(Ridge(alpha=alpha), scaler=True, impute=True),
                "supports_weight": True,
                "target_types": ["raw", "anomaly"] if preset == "full" else ["raw"],
            }
        )

    configs.append(
        {
            "name": "elasticnet_alpha_0.01_l1_0.2",
            "estimator": make_pipeline(ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=5000), scaler=True, impute=True),
            "supports_weight": True,
            "target_types": ["raw"],
        }
    )
    configs.append(
        {
            "name": "huber_epsilon_1.35",
            "estimator": make_pipeline(HuberRegressor(epsilon=1.35, max_iter=300), scaler=True, impute=True),
            "supports_weight": True,
            "target_types": ["raw"],
        }
    )

    hgb_grid = (
        itertools.product(["squared_error", "absolute_error"], [0.03, 0.05, 0.1], [300, 600], [15, 31, 63], [20, 50, 100], [0.0, 0.01, 0.1])
        if preset == "full"
        else itertools.product(["squared_error", "absolute_error"], [0.05], [300], [31], [50], [0.01])
    )
    for loss, lr, max_iter, leaves, min_leaf, l2 in hgb_grid:
        configs.append(
            {
                "name": f"hgb_{loss}_lr{lr}_iter{max_iter}_leaf{leaves}_min{min_leaf}_l2{l2}",
                "estimator": make_pipeline(
                    HistGradientBoostingRegressor(
                        loss=loss,
                        learning_rate=lr,
                        max_iter=max_iter,
                        max_leaf_nodes=leaves,
                        min_samples_leaf=min_leaf,
                        l2_regularization=l2,
                        random_state=42,
                    ),
                    scaler=False,
                    impute=False,
                ),
                "supports_weight": True,
                "target_types": ["raw", "anomaly"] if preset == "full" else ["raw"],
            }
        )

    et_grid = (
        itertools.product([300, 600], [None, 8, 12, 20], [1, 5, 20], [0.5, 0.8, 1.0])
        if preset == "full"
        else [(300, 12, 5, 0.8), (300, None, 5, 0.8)]
    )
    for n_estimators, max_depth, min_leaf, max_features in et_grid:
        configs.append(
            {
                "name": f"extratrees_n{n_estimators}_depth{max_depth}_min{min_leaf}_feat{max_features}",
                "estimator": make_pipeline(
                    ExtraTreesRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_leaf,
                        max_features=max_features,
                        n_jobs=-1,
                        random_state=42,
                    ),
                    scaler=False,
                    impute=True,
                ),
                "supports_weight": True,
                "target_types": ["raw"],
            }
        )

    configs.append(
        {
            "name": "randomforest_n300_depth12_min5",
            "estimator": make_pipeline(
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=5,
                    max_features=0.8,
                    n_jobs=-1,
                    random_state=42,
                ),
                scaler=False,
                impute=True,
            ),
            "supports_weight": True,
            "target_types": ["raw"],
        }
    )

    mlp_grid = (
        itertools.product([(64, 64), (128, 64), (160,)], [0.05, 0.1, 0.2], [1e-5, 1e-4, 1e-3])
        if preset == "full"
        else [((128, 64), 0.1, 1e-4)]
    )
    for hidden, dropout, wd in mlp_grid:
        configs.append(
            {
                "name": f"mlp_hidden{'x'.join(map(str, hidden))}_drop{dropout}_wd{wd}",
                "estimator": make_pipeline(
                    TorchMLPRegressor(hidden_sizes=hidden, dropout=dropout, weight_decay=wd),
                    scaler=True,
                    impute=True,
                ),
                "supports_weight": True,
                "target_types": ["raw"],
            }
        )

    return configs


def fit_predict(
    estimator: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    sample_weight: np.ndarray | None,
) -> np.ndarray:
    model = clone(estimator)
    if sample_weight is not None:
        model.fit(X_train, y_train, model__sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    return model.predict(X_val)


def adjusted_target(table: pd.DataFrame, idx: np.ndarray, target_type: str) -> np.ndarray:
    y = table.iloc[idx]["target_speed"].to_numpy(dtype=np.float32)
    if target_type == "anomaly":
        base = table.iloc[idx]["anomaly_base_27d_origin"].to_numpy(dtype=np.float32)
        return y - base
    return y


def restore_target(table: pd.DataFrame, idx: np.ndarray, pred: np.ndarray, target_type: str) -> np.ndarray:
    if target_type == "anomaly":
        base = table.iloc[idx]["anomaly_base_27d_origin"].to_numpy(dtype=np.float32)
        return pred + base
    return pred


def save_predictions(path: Path, table: pd.DataFrame, idx: np.ndarray, pred: np.ndarray, model_name: str) -> None:
    out = table.iloc[idx][["origin_datetime", "target_datetime", "target_speed", "persistence_27day_target_aligned"]].copy()
    out["model_name"] = model_name
    out["predicted_speed"] = pred
    out.to_csv(path, index=False)


def evaluate_baselines(table: pd.DataFrame, split: Split) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    train_idx, val_idx = split_rows(table, split)
    y_train = table.iloc[train_idx]["target_speed"].to_numpy(dtype=np.float32)
    y_val = table.iloc[val_idx]["target_speed"].to_numpy(dtype=np.float32)
    persistence = table.iloc[val_idx]["persistence_27day_target_aligned"].to_numpy(dtype=np.float32)
    recent = table.iloc[val_idx]["speed_current"].to_numpy(dtype=np.float32)
    mean_pred = np.full(len(y_val), float(np.mean(y_train)), dtype=np.float32)

    preds = {
        "public_train_mean": mean_pred,
        "persistence_27day": persistence,
        "recent_speed_t": recent,
    }
    rows = []
    for name, pred in preds.items():
        rows.append(
            {
                "model_name": name,
                "feature_set_name": "causal_tabular_v1",
                "target_type": "raw",
                "sample_weighting": "no",
                "validation_scheme": split.scheme,
                "fold": split.fold,
                **metrics(y_val, pred, persistence),
            }
        )
    return rows, preds


def run_split(table: pd.DataFrame, features: list[str], split: Split, configs: list[dict[str, Any]], write_preds: bool) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    train_idx, val_idx = split_rows(table, split)
    X_train = table.iloc[train_idx][features]
    X_val = table.iloc[val_idx][features]
    y_val = table.iloc[val_idx]["target_speed"].to_numpy(dtype=np.float32)
    persistence = table.iloc[val_idx]["persistence_27day_target_aligned"].to_numpy(dtype=np.float32)

    rows, pred_map = evaluate_baselines(table, split)

    for cfg in configs:
        for target_type in cfg["target_types"]:
            for weighted in [False, True] if cfg["supports_weight"] else [False]:
                y_train_model = adjusted_target(table, train_idx, target_type)
                if target_type == "anomaly" and np.isnan(y_train_model).any():
                    continue
                weights = target_weights(table.iloc[train_idx]["target_speed"].to_numpy(dtype=np.float32)) if weighted else None
                model_id = f"{cfg['name']}__target_{target_type}__weighted_{int(weighted)}"
                try:
                    pred_model_target = fit_predict(cfg["estimator"], X_train, y_train_model, X_val, weights)
                    pred = restore_target(table, val_idx, pred_model_target, target_type)
                except Exception as exc:
                    print(f"SKIP {model_id} {split.fold}: {exc}")
                    continue

                row = {
                    "model_name": cfg["name"],
                    "model_id": model_id,
                    "feature_set_name": "causal_tabular_v1",
                    "target_type": target_type,
                    "sample_weighting": "yes" if weighted else "no",
                    "validation_scheme": split.scheme,
                    "fold": split.fold,
                    **metrics(y_val, pred, persistence),
                }
                rows.append(row)
                pred_map[model_id] = pred
                if write_preds:
                    pred_path = PRED_DIR / f"{split.scheme}_{split.fold}_{model_id}.csv"
                    save_predictions(pred_path, table, val_idx, pred, model_id)
                print(
                    f"{split.scheme}/{split.fold} {model_id}: "
                    f"MAE={row['mae']:.2f} RMSE={row['rmse']:.2f} CC={row['cc']:.3f}"
                )
    return rows, pred_map


def ensemble_search(rows: pd.DataFrame, table: pd.DataFrame, features: list[str], configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fixed = fixed_split()
    train_idx, val_idx = split_rows(table, fixed)
    y_val = table.iloc[val_idx]["target_speed"].to_numpy(dtype=np.float32)
    persistence = table.iloc[val_idx]["persistence_27day_target_aligned"].to_numpy(dtype=np.float32)

    ranked = (
        rows[(rows["validation_scheme"] == "fixed_2022_2023") & rows["model_id"].notna()]
        .sort_values("cc", ascending=False)
        .head(4)
    )
    pred_candidates: dict[str, np.ndarray] = {"persistence_27day": persistence}
    X_train = table.iloc[train_idx][features]
    X_val = table.iloc[val_idx][features]
    for _, row in ranked.iterrows():
        cfg = next((c for c in configs if c["name"] == row["model_name"]), None)
        if cfg is None:
            continue
        weighted = row["sample_weighting"] == "yes"
        target_type = row["target_type"]
        y_train = adjusted_target(table, train_idx, target_type)
        weights = target_weights(table.iloc[train_idx]["target_speed"].to_numpy(dtype=np.float32)) if weighted else None
        pred = fit_predict(cfg["estimator"], X_train, y_train, X_val, weights)
        pred_candidates[str(row["model_id"])] = restore_target(table, val_idx, pred, target_type)

    out: list[dict[str, Any]] = []
    names = list(pred_candidates)
    for k in [2, 3]:
        for combo in itertools.combinations(names, min(k, len(names))):
            pred = np.mean([pred_candidates[name] for name in combo], axis=0)
            out.append(
                {
                    "model_name": f"average_best_{len(combo)}",
                    "model_id": "ensemble_avg__" + "__".join(combo),
                    "feature_set_name": "causal_tabular_v1",
                    "target_type": "raw",
                    "sample_weighting": "mixed",
                    "validation_scheme": fixed.scheme,
                    "fold": fixed.fold,
                    **metrics(y_val, pred, persistence),
                }
            )

    if len(names) >= 3:
        for weights in itertools.product(np.arange(0.0, 1.01, 0.05), repeat=min(3, len(names))):
            if abs(sum(weights) - 1.0) > 1e-9:
                continue
            combo = names[: len(weights)]
            pred = sum(w * pred_candidates[name] for w, name in zip(weights, combo))
            out.append(
                {
                    "model_name": "weighted_public_validation",
                    "model_id": "ensemble_weighted__" + "__".join(f"{n}:{w:.2f}" for n, w in zip(combo, weights)),
                    "feature_set_name": "causal_tabular_v1",
                    "target_type": "raw",
                    "sample_weighting": "mixed",
                    "validation_scheme": fixed.scheme,
                    "fold": fixed.fold,
                    **metrics(y_val, pred, persistence),
                }
            )
    return out


def select_config(cv_results: pd.DataFrame, fixed_results: pd.DataFrame, preset: str, features: list[str]) -> dict[str, Any]:
    model_rows = cv_results[cv_results["model_id"].notna()].copy()
    grouped = (
        model_rows.groupby(["model_id", "model_name", "target_type", "sample_weighting"], dropna=False)
        .agg(mean_cv_cc=("cc", "mean"), mean_cv_mae=("mae", "mean"), mean_cv_rmse=("rmse", "mean"), std_cv_cc=("cc", "std"), folds=("fold", "count"))
        .reset_index()
        .sort_values(["mean_cv_cc", "mean_cv_mae", "mean_cv_rmse", "std_cv_cc"], ascending=[False, True, True, True])
    )
    selected = grouped.iloc[0].to_dict()
    fixed_match = fixed_results[fixed_results["model_id"] == selected["model_id"]]
    return {
        "selection_rule": "highest mean expanding-window CV CC; ties by MAE, RMSE, CC variance, fixed validation sanity",
        "preset": preset,
        "feature_set_name": "causal_tabular_v1",
        "features": features,
        "selected": selected,
        "fixed_2022_2023": fixed_match.iloc[0].to_dict() if not fixed_match.empty else None,
        "private_evaluation": "not_run",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["initial", "full"], default="initial")
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--write-predictions", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    table = build_feature_table(FULL_CSV)
    features = feature_columns(table)
    configs = candidate_models(args.preset)
    if args.skip_mlp:
        configs = [cfg for cfg in configs if not cfg["name"].startswith("mlp_")]

    print(f"public_only_rows={len(table)} features={len(features)} configs={len(configs)} preset={args.preset}")

    fixed_rows, _ = run_split(table, features, fixed_split(), configs, args.write_predictions)
    fixed_df = pd.DataFrame(fixed_rows)
    ensemble_rows = ensemble_search(fixed_df, table, features, configs)
    if ensemble_rows:
        fixed_df = pd.concat([fixed_df, pd.DataFrame(ensemble_rows)], ignore_index=True)
    fixed_df.to_csv(OUT_DIR / "fixed_results.csv", index=False)

    cv_rows: list[dict[str, Any]] = []
    for split in cv_splits():
        rows, _ = run_split(table, features, split, configs, args.write_predictions)
        cv_rows.extend(rows)
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(OUT_DIR / "cv_results.csv", index=False)

    selected = select_config(cv_df, fixed_df, args.preset, features)
    (OUT_DIR / "selected_config.json").write_text(json.dumps(selected, indent=2, allow_nan=True))

    summary = (
        cv_df[cv_df["model_id"].notna()]
        .groupby(["model_id", "model_name", "target_type", "sample_weighting"], dropna=False)
        .agg(mean_cv_cc=("cc", "mean"), mean_cv_mae=("mae", "mean"), mean_cv_rmse=("rmse", "mean"), folds=("fold", "count"))
        .reset_index()
        .sort_values("mean_cv_cc", ascending=False)
        .head(12)
    )
    print("\nTop CV results")
    print(summary.to_string(index=False))
    print(f"\nSelected: {selected['selected']['model_id']}")
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
