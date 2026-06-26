# FINAL_REPORT_72H_BLOCK_MULTIHORIZON.md

## 1. Project overview

이 프로젝트의 목표는 near-Earth solar-wind speed를 예측하는 것이다. 기본 관측 시계열은 1시간 간격이며, 모델 개발에는 public 기간인 2011-2023년을 사용하고, 최종 성능 확인에는 private diagnostic 기간인 2024-2025년을 사용한다.

중요한 문제 정의 수정이 있었다. 최종 task는 매시간 forecast origin을 굴려 `Speed(t+72h)` 하나만 예측하는 single-horizon 문제가 아니다. 최종 task는 다음과 같은 **72h block multi-horizon forecast**이다.

```text
- forecast origin은 72시간마다 한 번 발행한다.
- 각 forecast origin t는 다음 72시간의 hourly speed profile을 예측한다.
- horizons: h = 1, 2, ..., 72
- 각 target은 Speed(t+h)이다.
- 72h forecast block들을 non-overlapping 방식으로 이어 붙인다.
- final private prediction은 2024-01-01 00:30부터 2025-12-31 23:30까지 모든 hourly target timestamp를 정확히 한 번씩 덮는다.
```

최종 private output은 target timestamp 기준으로 한 행당 하나의 예측값을 가진다.

```csv
datetime,predicted_speed
2024-01-01 00:30:00,310.5190
...
2025-12-31 23:30:00,490.6789
```

## 2. Data

### Main solar-wind CSV

주요 solar-wind 입력 데이터는 repository root에 놓이는 다음 CSV 파일들이다.

```text
solar_wind-public.csv   # public development period, 2011-2023
solar_wind-private.csv  # private diagnostic period, 2024-2025
solar_wind_data.csv     # public + private combined, 2011-2025
```

공통 columns:

| Column | Meaning |
|---|---|
| `datetime` | UTC hourly timestamp |
| `Speed (km/s)` | target solar-wind speed |
| `Density (1/cm^3)` | proton density |
| `Temperature (K)` | proton temperature |
| `B (nT)` | IMF magnitude |
| `Sunspot Number` | daily solar activity proxy |
| `Coronal Hole Area` | daily CH area proxy |

### Coronal-hole feature CSV

추가 CH morphology/intensity features는 다음 파일에서 온다.

```text
ch_feature/ch_features_by_time.csv
```

최종 모델은 `representative_mrmr_ch` feature set만 사용한다. Expanded CH feature set은 최종 모델에 포함하지 않는다.

### CME catalog

CME catalog는 다음 파일에 있다.

```text
data/donki_cme_catalog.csv
```

이 파일은 CME residual correction 실험에만 사용되었고, 최종 **72h block multi-horizon forecast** 모델에는 사용하지 않는다.

### Magnetogram / F10.7 / other attempted inputs

F10.7, OMNI-derived additional features, magnetogram features, CME features는 모두 실험적으로 검토되었지만 최종 모델에는 포함하지 않았다. Public validation에서 충분한 개선이 없거나, local data coverage가 없거나, private diagnostic에서 악화되는 문제가 있었다.

### GitHub commit policy

다음 대형 데이터와 산출물은 GitHub code repository에 commit하지 않는다.

```text
solar_wind-public.csv
solar_wind-private.csv
solar_wind_data.csv
ch_feature/ch_features_by_time.csv
data/
outputs/
FITS files
PNG plots
model checkpoints
local caches
.venv/
```

GitHub repository는 code, Markdown documentation, lightweight config, experiment records 중심으로 유지한다.

## 3. Forecast formulation

Notation:

```text
forecast origin: t
horizon: h = 1, ..., 72
target: Speed(t+h)
forecast block: {Speed(t+1h), Speed(t+2h), ..., Speed(t+72h)}
```

Forecast origins are spaced every 72 hours. Consecutive blocks do not overlap in target time.

Text diagram:

```text
origin t0
  -> targets: t0+1h, t0+2h, ..., t0+72h

origin t1 = t0 + 72h
  -> targets: t1+1h, t1+2h, ..., t1+72h

origin t2 = t1 + 72h
  -> targets: t2+1h, t2+2h, ..., t2+72h
```

Train/validation/private membership is determined by **target timestamp**, not by origin timestamp alone. This matters because the first private target at `2024-01-01 00:30` has forecast origin `2023-12-31 23:30`, which lies in the public calendar year but is valid because all input data are at or before the origin and the target is private.

## 4. Feature engineering

최종 모델은 `run_multihorizon_72h_blocks.py`에서 causal origin-time feature table을 만든다. 모든 features는 forecast origin `t`에서 이미 관측되었거나 그 이전 시각의 값만 사용한다.

### Current tabular features

Base tabular features include:

```text
speed_current
Speed lag features: 1, 3, 6, 12, 24, 48, 72, 96, 168h
Density / Temperature / B lag features: 1, 6, 24, 72h
rolling means: Speed 6, 12, 24, 72, 168h
rolling std: Speed 24, 72, 168h
rolling min/max: Speed 24, 72h
trend features:
  speed_current - Speed(t-24h)
  speed_current - Speed rolling mean 24h
  Speed rolling mean 24h - Speed rolling mean 72h
Coronal Hole Area current/lags/rolling features
Sunspot Number current, lag, rolling mean
Calendar features:
  day_of_year_sin/cos
  month_sin/cos
```

### Target-aligned 27-day recurrence features

태양 자전 recurrence를 반영하기 위해 target-aligned 27-day recurrence features를 사용한다. Multi-horizon task에서는 horizon별로 target-aligned recurrence center가 달라진다.

```text
For horizon h:
  target time = t + h
  27-day recurrence target-aligned source = t + h - 648h
                                      = t - (648h - h)
```

따라서 `run_multihorizon_72h_blocks.py`는 horizon별 feature frame을 만들 때 `persistence_27day_target_aligned`와 target-aligned recurrence 주변 features를 h에 맞게 갱신한다. 이 값들은 모두 forecast origin `t`보다 과거이다.

### representative_mrmr_ch features

최종 CH feature set은 `run_ch_feature_addition_72h.py`의 `REPRESENTATIVE_CH`이다.

| Feature | Target lag | Origin offset |
|---|---:|---:|
| `I_mean_W_lon7p5_lat15` | 4d | -1d |
| `A_W_lon30_lat30_km2` | 5d | -2d |
| `A_source_if_overlap_W_lon60_lat60_km2` | 4d | -1d |
| `A_W_lon30_lat15_km2` | 4d | -1d |
| `lat_width_eff_10_full_if_overlap_largest_W_lon7p5_lat15` | 3d | 0d |
| `A_grid_4x3_lat1_lon1_km2` | 4d | -1d |

72h는 3일이므로:

```text
target-lag 3d  -> target time - 3d = origin time
target-lag 4d  -> target time - 4d = origin - 1d
target-lag 5d  -> target time - 5d = origin - 2d
```

CH merge는 `merge_asof(..., direction="backward")`로 수행되어 requested CH time 이후의 CH observation을 사용하지 않는다.

### Strict causality

```text
- 모든 solar-wind lag/rolling features는 forecast origin t 또는 그 이전 값만 사용한다.
- CH features는 requested time 이하의 가장 최근 CH row만 backward match한다.
- target Speed(t+h)는 feature로 들어가지 않는다.
- private labels는 model selection이나 training input에 사용하지 않는다.
```

## 5. Model scheme

최종 모델은 horizon별 direct model이다.

For each horizon `h = 1..72`:

```text
Train Direct MLP_h:
  X(t) -> Speed(t+h)

Train ExtraTrees_h:
  X(t) -> Speed(t+h)

Final ensemble:
  ensemble_h = 0.7 * MLP_h + 0.3 * ExtraTrees_h
```

각 forecast block에서 h=1부터 h=72까지 해당 horizon model을 호출하고, 72개 예측을 시간순으로 이어 붙인다.

Final model includes:

```text
current tabular features
representative_mrmr_ch features
Direct MLP_h
ExtraTrees_h
0.7 MLP_h + 0.3 ExtraTrees_h ensemble
```

Final model excludes:

```text
CME residual correction
F10.7 features
magnetogram features
expanded CH feature set
OMNI-derived extra feature sets
smooth-anchor interpolation
single h=72-only formulation
```

## 6. Training and evaluation splits

### Public fixed validation

Public fixed validation uses non-overlapping 72h forecast blocks in 2022-2023.

```text
Train target period:
  2011-01-01 00:30 through 2021-12-31 23:30

Validation target period:
  2022-01-01 00:30 through 2023-12-31 23:30
```

### Private diagnostic

Private diagnostic trains on public years only and predicts the non-overlapping 72h blocks covering 2024-2025.

```text
Train target period:
  2011-01-01 00:30 through 2023-12-31 23:30

Private prediction target period:
  2024-01-01 00:30 through 2025-12-31 23:30
```

Private results are diagnostic only. They are not used to choose features, tune hyperparameters, tune ensemble weights, or decide whether to adopt CME/F10.7/magnetogram features.

## 7. Evaluation metrics

Metrics are computed over finite observed target labels.

```text
N     = number of finite target/prediction pairs
MAE   = mean(|y - prediction|)
RMSE  = sqrt(mean((y - prediction)^2))
CC    = Pearson correlation(y, prediction)
```

The final block forecast metrics are not directly comparable to older h=72-only metrics because the block forecast includes horizons h=1..72. Short horizons are naturally easier, so overall block metrics are expected to be better than a pure h=72 metric. The h=72-only slice is still useful as a sanity check against the older single-horizon private CC.

## 8. Final results

All values below are read from saved outputs in:

```text
outputs/multihorizon_72h_blocks/
```

### Overall results

| Model | Public fixed N | Public MAE | Public RMSE | Public CC | Private N | Private MAE | Private RMSE | Private CC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct MLP | 17,292 | 52.1515 | 68.9838 | 0.648927 | 17,280 | 52.4573 | 75.0656 | 0.703438 |
| ExtraTrees | 17,292 | 49.6350 | 65.2579 | 0.688737 | 17,280 | 52.1130 | 73.9269 | 0.718296 |
| 0.7 MLP + 0.3 ExtraTrees | 17,292 | 50.6616 | 66.9362 | 0.669880 | 17,280 | 51.1626 | 73.3121 | 0.717528 |

The final selected output family remains the requested ensemble:

```text
ensemble_0p7_mlp_0p3_extratrees
```

Note: `ExtraTrees` alone has the best public fixed CC in this particular block evaluation, but the final scheme requested and used for the project is the fixed selected family `0.7 * MLP_h + 0.3 * ExtraTrees_h`.

### Private yearly results for the ensemble

| Year | N | MAE | RMSE | CC |
|---:|---:|---:|---:|---:|
| 2024 | 8,581 | 46.2474 | 65.2816 | 0.577928 |
| 2025 | 8,699 | 56.0112 | 80.4521 | 0.676448 |

### Selected private horizon metrics for the ensemble

| Horizon h | N | MAE | RMSE | CC |
|---:|---:|---:|---:|---:|
| 1 | 238 | 13.5628 | 21.1305 | 0.980410 |
| 6 | 238 | 28.2316 | 44.6105 | 0.902513 |
| 12 | 240 | 36.2506 | 51.7398 | 0.854877 |
| 24 | 242 | 47.7954 | 67.2906 | 0.745239 |
| 36 | 243 | 57.3400 | 84.2726 | 0.671150 |
| 48 | 243 | 60.8979 | 83.0890 | 0.663714 |
| 60 | 240 | 58.6254 | 80.1624 | 0.653500 |
| 72 | 238 | 61.1159 | 85.0333 | 0.567184 |

Performance naturally degrades with horizon. The h=72 private CC is `0.567184`, which is reasonably consistent with the previous single-horizon h=72 private CC around `0.581`. The absolute difference is about `0.0138`, supporting that the multi-horizon block implementation is not leaking future labels.

## 9. Sanity and leakage checks

Saved sanity files:

```text
outputs/multihorizon_72h_blocks/leakage_sanity_report.csv
outputs/multihorizon_72h_blocks/timestamp_sanity_check.csv
outputs/multihorizon_72h_blocks/origin_target_mapping.csv
```

Sanity summary:

| Check | Status | Meaning |
|---|---|---|
| Target coverage | PASS | Private target timestamps from 2024-01-01 00:30 to 2025-12-31 23:30 are all predicted exactly once. |
| Origin spacing | PASS | 244 unique private forecast origins, min/median/max consecutive spacing all 72h. |
| Timestamp causality | PASS | Random 10,000 rows satisfy `target_datetime = forecast_origin + horizon_hour`, horizon is in 1..72, and feature timestamps are at or before origin. |
| Split leakage | PASS | Public fixed training target period ends before validation starts; private diagnostic labels are not used for training/model selection. |
| h=72 comparison | PASS | Previous h=72 private CC ≈ 0.581; new multihorizon h=72 private CC = 0.567184. |

Coverage details:

```text
Expected private hourly target timestamps: 17,544
Predicted private rows: 17,544
Missing timestamps: 0
Extra timestamps: 0
Duplicated target timestamps: 0
Finite observed speed labels: 17,280
Non-finite observed speed labels in source data: 264
```

The 264 non-finite observed labels are present in the source target period and are excluded from metrics. Predictions are still produced for all 17,544 private hourly timestamps.

## 10. Previous experiments and decisions

### 1. Single-horizon h=72 Direct MLP

Early work treated the problem as predicting `Speed(t+72h)` for every hourly origin. This was useful as a baseline and for feature/model exploration, but it is superseded by the corrected block multi-horizon formulation.

### 2. 27-day persistence residual model

Residual formulation:

```text
residual = Speed(t+72h) - Speed(t-576h)
prediction = Speed(t-576h) + alpha * predicted_residual
```

This did not beat the direct model family reliably and was not adopted.

### 3. Tabular ExtraTrees and ensemble

ExtraTrees was consistently useful. The project selected the fixed model family:

```text
0.7 * Direct MLP + 0.3 * ExtraTrees
```

The same model family is used per horizon in the final block forecast.

### 4. Representative CH features

The `representative_mrmr_ch` feature set was adopted. It adds a compact set of CH morphology/intensity features with causal origin-time alignment.

### 5. Expanded CH features

Expanded CH features were evaluated but not adopted. The representative set was kept as the final CH feature set.

### 6. F10.7 features

F10.7 feature experiments did not improve public fixed validation or public CV relative to the current best baseline, so F10.7 was not adopted.

### 7. OMNI-derived features

Additional OMNI-derived feature experiments were not adopted in the final model. The final model uses the current tabular feature family and representative CH features.

### 8. Magnetogram features

Magnetogram feature work was implemented with safeguards, but usable local magnetogram coverage was missing or insufficient. Magnetogram features were therefore not adopted.

### 9. CME residual correction

CME residual correction used `data/donki_cme_catalog.csv` and causal recent-window CME features. It worsened results and was not adopted.

Simplified CME residual experiment results:

| Model | Public MAE | Public RMSE | Public CC | Private MAE | Private RMSE | Private CC |
|---|---:|---:|---:|---:|---:|---:|
| Base only | 64.0717 | 82.0583 | 0.444030 | 61.3793 | 85.9894 | 0.580837 |
| Base + CME residual correction | 71.6180 | 86.7311 | 0.443873 | 74.7879 | 97.5147 | 0.471723 |

Thus CME residual correction is explicitly not part of the final model.

## 11. Repository / file organization

Important scripts:

```text
train_tabular_models_72h.py
run_ch_feature_addition_72h.py
run_multihorizon_72h_blocks.py
run_cme_residual_correction_72h.py
FINAL_REPORT_72H_BLOCK_MULTIHORIZON.md
```

`evaluate_submissions.py` is referenced in the original README text as a possible competition-style workflow, but it is not present in the current code-only workspace snapshot inspected for this report.

Important final outputs:

```text
outputs/multihorizon_72h_blocks/summary.csv
outputs/multihorizon_72h_blocks/fixed_results.csv
outputs/multihorizon_72h_blocks/private_diagnostic.csv
outputs/multihorizon_72h_blocks/private_yearly_diagnostic.csv
outputs/multihorizon_72h_blocks/horizon_metrics.csv
outputs/multihorizon_72h_blocks/horizon_selected_metrics.csv
outputs/multihorizon_72h_blocks/block_metrics.csv
outputs/multihorizon_72h_blocks/best_private_prediction.csv
outputs/multihorizon_72h_blocks/origin_target_mapping.csv
outputs/multihorizon_72h_blocks/timestamp_sanity_check.csv
outputs/multihorizon_72h_blocks/leakage_sanity_report.csv
outputs/multihorizon_72h_blocks/config.json
outputs/multihorizon_72h_blocks/fixed_predictions.csv
outputs/multihorizon_72h_blocks/private_predictions.csv
```

Generated profiles:

```text
outputs/multihorizon_72h_blocks/solar_wind_profile_reference_ensemble_2022.png
outputs/multihorizon_72h_blocks/solar_wind_profile_reference_ensemble_2023.png
outputs/multihorizon_72h_blocks/solar_wind_profile_reference_ensemble_2024.png
outputs/multihorizon_72h_blocks/solar_wind_profile_reference_ensemble_2025.png
```

Large outputs are local artifacts. They should not be committed to the code-only GitHub repository unless they are deliberately selected as lightweight records.

## 12. Reproducibility notes

Required local data placement:

```text
solar_wind_data.csv
solar_wind-public.csv
solar_wind-private.csv
ch_feature/ch_features_by_time.csv
```

Optional data for rejected or diagnostic experiments:

```text
data/donki_cme_catalog.csv
data/magnetograms/
F10.7 input files, if available
```

Python environment:

```text
.venv/
pyproject.toml
uv.lock
requirements-magnetogram-stage2.txt
```

Important dependencies include:

```text
numpy
pandas
scikit-learn
torch
scipy
astropy / drms / sunpy for magnetogram-related scripts
```

Command to reproduce the final multi-horizon experiment:

```bash
.venv/bin/python run_multihorizon_72h_blocks.py
```

Useful debug command for a lightweight smoke run:

```bash
.venv/bin/python run_multihorizon_72h_blocks.py --max-horizon 2 --output-dir outputs/multihorizon_72h_blocks_smoke
```

`run_multihorizon_72h_blocks.py` currently supports:

```text
--output-dir
--skip-private
--max-horizon
```

There is no built-in `--sanity-only` flag. Sanity outputs are produced during the normal run, and the additional leakage report in this workspace was computed from saved output CSVs without retraining.

## 13. Final conclusion

The selected final model is the **72h block multi-horizon ensemble**:

```text
Feature set:
  current tabular features
  + representative_mrmr_ch

For each horizon h = 1..72:
  train MLP_h
  train ExtraTrees_h
  prediction_h = 0.7 * MLP_h + 0.3 * ExtraTrees_h

Forecast:
  issue one forecast every 72 hours
  predict Speed(t+1h) ... Speed(t+72h)
  concatenate non-overlapping 72h blocks
```

Final private diagnostic for the ensemble:

| Metric | Value |
|---|---:|
| MAE | 51.1626 |
| RMSE | 73.3121 |
| CC | 0.717528 |

The earlier h=72-only model is superseded because the corrected task is a 72-hour block multi-horizon forecast, not a single-horizon `Speed(t+72h)` task.
