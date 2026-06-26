# Work Log

This records the modeling work done so far for the 72-hour solar-wind speed
forecasting task.

## Project Goal

Predict near-Earth solar-wind speed 72 hours ahead:

```text
Speed(t + 72h)
```

using past solar-wind and solar-activity inputs from the hourly CSV dataset.

## Data Used

Main data file:

```text
solar_wind_data.csv
```

Public/private split:

```text
Public:  2011-01-01 00:30:00 to 2023-12-31 23:30:00
Private: 2024-01-01 00:30:00 to 2025-12-31 23:30:00
```

Available columns:

```text
datetime
Speed (km/s)
Density (1/cm^3)
Temperature (K)
B (nT)
Sunspot Number
Coronal Hole Area
```

## Feature Strategy

For most models, each sample uses lagged values at:

```text
0, 1, 2, 3, 6, 12, 24, 48, 72, 96, 168, 336, 576, 648 hours before origin t
```

The important physics-related lag for a 72-hour forecast is:

```text
Speed(t + 72h - 648h) = Speed(t - 576h)
```

This is the target-aligned 27-day persistence value.

## Models and Scripts

### 1. First Direct MLP

Script:

```text
train_first_try_72h.py
```

Target:

```text
Speed(t + 72h)
```

Training strategy:

```text
Train:       2011-2019
Validation:  2020-2021
Public test: 2022-2023
Private:     2024-2025
```

Private result:

```text
MAE  = 63.49
RMSE = 89.19
CC   = 0.537
```

Outputs:

```text
outputs/first_try_72h/
```

### 2. Public Ensemble Direct MLP

Script:

```text
train_public_ensemble_72h.py
```

Target:

```text
Speed(t + 72h)
```

Training strategy:

```text
Use 2011-2021 train and 2022-2023 validation for epoch selection.
Retrain final ensemble on all public data, 2011-2023.
Evaluate private 2024-2025 once.
```

Seeds:

```text
11, 42, 77, 101, 123
```

Validation mean over selection models:

```text
MAE  = 64.54
RMSE = 83.39
CC   = 0.432
```

Private result:

```text
MAE  = 62.90
RMSE = 88.85
CC   = 0.548
```

Outputs:

```text
outputs/public_ensemble_72h/
```

### 3. No-Temperature Direct MLP

Script:

```text
train_no_temperature_72h.py
```

Target:

```text
Speed(t + 72h)
```

Input columns:

```text
Speed (km/s)
Density (1/cm^3)
B (nT)
Sunspot Number
Coronal Hole Area
```

`Temperature (K)` was removed.

Private result:

```text
MAE  = 62.88
RMSE = 89.33
CC   = 0.551
```

Conclusion:

```text
Removing Temperature gave almost the same performance.
MAE and CC were slightly better, RMSE slightly worse.
```

Outputs:

```text
outputs/no_temperature_72h/
```

Images:

```text
outputs/no_temperature_72h/sample_private_prediction_no_temperature.png
outputs/no_temperature_72h/best_example_no_temperature.png
outputs/no_temperature_72h/worst_example_no_temperature.png
```

### 4. Residual MLP With Public Validation

Script:

```text
train_residual_72h.py
```

Residual target:

```text
residual = Speed(t + 72h) - Speed(t + 72h - 648h)
```

For the 72-hour forecast:

```text
residual = Speed(t + 72h) - Speed(t - 576h)
```

Final prediction:

```text
prediction = Speed(t - 576h) + alpha * predicted_residual
```

Rows without valid target-aligned persistence are ignored.

Public validation period:

```text
2022-2023
```

Alpha tuning grid:

```text
0.0, 0.1, 0.2, ..., 1.0
```

Alpha selection criterion:

```text
Select alpha by validation CC first.
```

Selected alpha:

```text
alpha = 1.0
```

Best final-prediction CC epoch:

```text
epoch = 1
```

Validation comparison:

```text
27-day persistence:
  MAE  = 85.37
  RMSE = 110.70
  CC   = 0.240

Residual alpha=1.0:
  MAE  = 65.82
  RMSE = 84.87
  CC   = 0.399

Previous direct MLP:
  MAE  = 64.54
  RMSE = 83.39
  CC   = 0.432
```

Conclusion:

```text
Residual learning improved strongly over raw 27-day persistence,
but it did not beat the previous direct MLP on validation.
```

Outputs:

```text
outputs/residual_72h/
```

Images:

```text
outputs/residual_72h/sample_validation_residual.png
outputs/residual_72h/best_validation_residual.png
outputs/residual_72h/worst_validation_residual.png
```

### 5. Residual MLP Private Evaluation

Script:

```text
train_residual_private_72h.py
```

The `.order` file requested:

```text
Do not run private training with only 1 epoch.
Use selected validation alpha and selected epochs, e.g. 50.
Evaluate private only once.
```

Implementation:

```text
Selected validation alpha = 1.0
Validation best CC epoch = 1
Private training default enforces at least 50 epochs.
```

Private residual result with 50 epochs:

```text
MAE  = 70.25
RMSE = 96.01
CC   = 0.444
```

Conclusion:

```text
The 50-epoch residual private model was worse than the previous direct MLP.
It likely overfit the residual target.
```

Outputs:

```text
outputs/residual_private_72h/
```

Images:

```text
outputs/residual_private_72h/sample_private_residual.png
outputs/residual_private_72h/best_private_residual.png
outputs/residual_private_72h/worst_private_residual.png
```

## Best and Worst Examples Observed

The recurring worst-case event was around:

```text
target: 2024-05-12 01:30
observed speed: about 1006 km/s
```

All simple lag-based MLP variants badly underpredicted this event.

Example residual private worst case:

```text
origin:      2024-05-09 01:30
target:      2024-05-12 01:30
persistence: 362.000
residual:      5.750
predicted:   367.750
observed:   1006.000
error:      -638.250 km/s
```

## GPU Note

The machine has four RTX 4090 GPUs. They are visible from the GPU-enabled
execution path:

```text
torch 2.6.0+cu124
torch.cuda.is_available() = True
torch.cuda.device_count() = 4
0 NVIDIA GeForce RTX 4090
1 NVIDIA GeForce RTX 4090
2 NVIDIA GeForce RTX 4090
3 NVIDIA GeForce RTX 4090
```

A local environment was created at:

```text
.venv/
```

Most scikit-learn tree and linear models run on CPU. PyTorch MLP runs use CUDA
when launched through the GPU-enabled execution path.

## Tabular Public-Only Experiments

The residual branch was stopped because it did not beat the direct MLP on
public validation.

Script:

```text
train_tabular_models_72h.py
```

Goal:

```text
Predict Speed(t + 72h) directly.
Use 27-day persistence and recurrence values as input features,
not as a residual target.
```

Feature strategy:

```text
current/recent speed lags
target-aligned 27-day recurrence: Speed(t - 576h)
source-surface recurrence around t - 648h
causal rolling means/std/min/max
speed trend features
density/temperature/B lag and rolling features
coronal-hole area lag and rolling features
sunspot and calendar features
```

Validation:

```text
Fixed public validation:
  train 2011-2021
  validation 2022-2023

Expanding-window CV:
  validation years 2017, 2018, 2019, 2020, 2021, 2022, 2023
```

Best public-CV tabular model from the initial run:

```text
ExtraTreesRegressor
n_estimators = 300
max_depth = 12
min_samples_leaf = 5
max_features = 0.8
sample weighting = no

Mean CV:
  MAE  = 56.49
  RMSE = 72.53
  CC   = 0.552

Fixed 2022-2023:
  MAE  = 65.32
  RMSE = 83.25
  CC   = 0.407
```

The selected ExtraTrees model was worse than the reproduced direct MLP on fixed
2022-2023 validation:

```text
Direct MLP fixed CC:    0.443
ExtraTrees fixed CC:    0.407
```

Therefore both direct MLP and ExtraTrees were kept as candidates.

## Frozen Non-CH Ensemble

Script:

```text
freeze_final_ensemble_72h.py
correct_full_private_outputs_72h.py
```

Public-selected ensemble:

```text
0.70 * Direct MLP + 0.30 * ExtraTrees + 0.00 * 27-day persistence
```

The first frozen output dropped rows with missing 27-day persistence. This was
fixed because the ensemble has zero persistence weight and should not drop
private rows only because persistence is NaN.

Corrected full private files:

```text
outputs/final_private_72h_ensemble_full.csv
outputs/final_private_72h_direct_mlp_full.csv
```

Row diagnostic:

```text
private total rows:                         17544
private finite-target rows:                 17280
old ensemble prediction rows:               17032
corrected ensemble prediction rows:         17544
missing finite-target rows after correction:    0
```

Corrected private diagnostic:

```text
Corrected ensemble full:
  MAE  = 62.55
  RMSE = 89.00
  CC   = 0.565

Corrected direct MLP full:
  MAE  = 62.57
  RMSE = 89.59
  CC   = 0.551
```

## CH Feature Addition

New CH input file:

```text
ch_feature/ch_features_by_time.csv
```

Script:

```text
run_ch_feature_addition_72h.py
```

Excluded target/OMNI diagnostic columns:

```text
solar_wind_speed_kms
omni_V1800
omni_time
omni_matched
omni_speed_source_column
```

CH matching:

```text
merge_asof backward
tolerance = 12 hours
matched CH time <= requested CH time
requested CH time <= forecast origin time
target time = origin time + 72h
```

Representative CH feature set:

```text
I_mean_W_lon7p5_lat15 at target-lag 4d
A_W_lon30_lat30_km2 at target-lag 5d
A_source_if_overlap_W_lon60_lat60_km2 at target-lag 4d
A_W_lon30_lat15_km2 at target-lag 4d
lat_width_eff_10_full_if_overlap_largest_W_lon7p5_lat15 at target-lag 3d
A_grid_4x3_lat1_lon1_km2 at target-lag 4d
```

Expanded CH feature set added:

```text
log_I_mean_W_lon7p5_lat15 at target-lag 4d and 5d
I_mean_W_lon7p5_lat15 at target-lag 5d
A_frac_visible_hemisphere_W_lon30_lat30 at target-lag 5d
```

Comparison for the frozen ensemble family:

```text
Current final features:
  Fixed CC    = 0.401
  CV mean CC  = 0.535
  Private MAE = 62.80
  Private RMSE= 89.71
  Private CC  = 0.556

Current + representative_mrmr_ch:
  Fixed CC    = 0.452
  CV mean CC  = 0.557
  Private MAE = 61.38
  Private RMSE= 85.99
  Private CC  = 0.581

Current + expanded_mrmr_ch:
  Fixed CC    = 0.435
  CV mean CC  = 0.551
  Private MAE = 61.70
  Private RMSE= 87.04
  Private CC  = 0.575
```

The representative CH feature set improved public fixed validation, public CV,
and private diagnostics. The expanded CH set was not selected.

## Current Frozen Best Model

Script:

```text
freeze_final_ch_representative_72h.py
```

Final selected configuration:

```text
0.70 * Direct MLP + 0.30 * ExtraTrees
features = current tabular features + representative_mrmr_ch
expanded_mrmr_ch = excluded
```

Final frozen outputs:

```text
outputs/final_ch_representative_72h_ensemble.csv
outputs/final_ch_representative_72h_comparison.csv
outputs/final_ch_representative_72h_config.json
outputs/final_ch_representative_72h_sanity_check.csv
```

The final prediction CSV has exactly:

```text
datetime,predicted_speed
```

and contains all private timestamps:

```text
private prediction rows:           17544
private finite-target scored rows: 17280
```

Final comparison:

```text
27-day persistence:
  Fixed CC     = 0.240
  CV mean CC   = 0.380
  Private MAE  = 79.88
  Private RMSE = 111.58
  Private CC   = 0.427

Old ensemble without new CH features:
  Fixed CC     = 0.444
  Private MAE  = 62.55
  Private RMSE = 89.00
  Private CC   = 0.565

New representative CH ensemble:
  Fixed CC     = 0.452
  CV mean CC   = 0.557
  Private MAE  = 61.38
  Private RMSE = 85.99
  Private CC   = 0.581

Expanded CH ensemble:
  Fixed CC     = 0.435
  CV mean CC   = 0.551
  Private MAE  = 61.70
  Private RMSE = 87.04
  Private CC   = 0.575
```

## Experiment 2: F10.7 Solar-Cycle Features

Script:

```text
run_f107_features_72h.py
```

Output directory:

```text
outputs/f107_features_72h/
```

Local source used:

```text
OMNI2_H0_MRG1HR_563544.txt
```

The OMNI2 file contains `DAILY_F10.7`; values were merged by forecast
origin date only. Rolling means are trailing/causal, and missing indicators
were added for each F10.7 feature. Rows were not dropped due to missing
F10.7 values; the model pipelines keep train-only imputation.

Compared feature sets:

```text
baseline_current_best = current tabular features + representative_mrmr_ch
f107_basic            = baseline + F10.7 origin, 27-day mean, 81-day mean
f107_full             = baseline + all requested F10.7 means/deltas/trends
```

Ensemble results:

```text
baseline_current_best:
  Fixed CC     = 0.452348
  CV mean CC   = 0.557240
  Private MAE  = 61.379343
  Private RMSE = 85.989385
  Private CC   = 0.580837

f107_basic:
  Fixed CC     = 0.446806
  CV mean CC   = 0.527884
  Private MAE  = 62.452553
  Private RMSE = 87.681191
  Private CC   = 0.572113

f107_full:
  Fixed CC     = 0.435840
  CV mean CC   = 0.549343
  Private MAE  = 62.698153
  Private RMSE = 87.987319
  Private CC   = 0.570641
```

Selection result:

```text
Adopt F10.7 features: False
Best public-selected feature set: baseline_current_best
```

F10.7 did not improve public fixed validation or public CV, so the frozen
best model remains the representative CH ensemble.

## Experiment 3: Magnetogram-Enhanced CH Features

Script:

```text
run_magnetogram_ch_features_72h.py
```

Output directory:

```text
outputs/magnetogram_ch_features_72h/
```

Local data availability:

```text
HMI/GONG/precomputed magnetogram feature file: missing
matched magnetogram rows: 0
feature mode: not_available
missing rate by year: 100%
missing rate by requested magnetic feature: 100%
```

The workspace contains the CH morphology/intensity feature table, but no local
magnetogram maps, CH masks with magnetic pixels, or precomputed magnetic
summary table. Therefore the script did not fabricate magnetic features and
did not evaluate `magnetogram_window_features` or
`magnetogram_ch_mask_features`.

Files written:

```text
outputs/magnetogram_ch_features_72h/summary.csv
outputs/magnetogram_ch_features_72h/fixed_results.csv
outputs/magnetogram_ch_features_72h/cv_results.csv
outputs/magnetogram_ch_features_72h/private_diagnostic.csv
outputs/magnetogram_ch_features_72h/best_private_prediction.csv
outputs/magnetogram_ch_features_72h/timestamp_sanity_check.csv
outputs/magnetogram_ch_features_72h/data_availability_report.csv
```

Baseline ensemble result:

```text
baseline_current_best:
  Fixed CC     = 0.452348
  CV mean CC   = 0.557240
  Private MAE  = 61.379343
  Private RMSE = 85.989385
  Private CC   = 0.580837
```

Selection result:

```text
Adopt magnetogram features: False
Best public-selected feature set: baseline_current_best
```

No magnetogram feature set was adopted because no usable local magnetogram
data source was available.

## Magnetogram Stage 0/1 Feasibility Implementation

Script:

```text
run_magnetogram_stage0_daily.py
```

Purpose:

```text
Stage 0: daily-cadence magnetogram acquisition attempt
Stage 1: approximate disk-window magnetic feature extraction when FITS files exist
```

Implemented safeguards:

```text
- daily cadence only, default 2022-01-01 through 2025-12-31
- target time 00:00 UT
- raw files under data/magnetograms/raw/
- extracted features under data/magnetograms/magnetogram_window_features_daily.csv
- no high-cadence HMI full-disk downloads
- no Carrington synoptic maps
- HMI JSOC path requires JSOC_EMAIL and drms; otherwise it is documented and skipped
- GONG fallback probes public NSO/GONG quick-reduce zero-point-corrected products
- first-pass features are labeled approximate_disk_coordinate
```

Feature windows:

```text
W_lon7p5_lat15
W_lon30_lat30
W_lon30_lat15
W_lon60_lat60
```

Feature summaries:

```text
mean_abs_B
median_abs_B
signed_mean_B
sum_pos_B
sum_neg_B
polarity_imbalance
dominant_polarity
```

Two-day probe command:

```text
.venv/bin/python run_magnetogram_stage0_daily.py --max-days 2 --timeout 20
```

Probe result:

```text
HMI:
  status = not_attempted
  reason = JSOC_EMAIL is not set; JSOC exports usually require registered email

GONG:
  2022-01-01 status = missing
  note = zqs HTTP 500; bqs HTTP 500 from NSO archive

  2022-01-02 status = missing
  note = zqs no FITS found; bqs HTTP 500 from NSO archive

Feature extraction:
  rows = 0
  astropy = missing in current .venv
```

Files written:

```text
data/magnetograms/stage0_download_report.csv
data/magnetograms/magnetogram_window_features_daily.csv
data/magnetograms/raw/
```

The main magnetogram evaluator was also updated to recognize
`data/magnetograms/magnetogram_window_features_daily.csv` once it contains
real rows. Empty or all-missing Stage 0 feature tables are now treated as no
usable magnetogram coverage, so they are not evaluated as fake feature sets.

## Latest Status Check: Magnetogram Stage 0

Timestamp:

```text
2026-06-24 17:33:12 KST
```

Current artifacts:

```text
data/magnetograms/stage0_download_report.csv              522 bytes
data/magnetograms/magnetogram_window_features_daily.csv   919 bytes
data/magnetograms/raw/                                    empty
outputs/magnetogram_ch_features_72h/data_availability_report.csv 5.3K
```

Current blocker:

```text
No raw magnetogram FITS files have been downloaded yet.
The two-day GONG probe hit NSO archive HTTP 500/missing-file responses.
HMI remains unavailable until JSOC_EMAIL and drms are configured.
FITS extraction also needs astropy installed in the active .venv.
```

Current model status:

```text
Frozen best remains:
0.70 * Direct MLP + 0.30 * ExtraTrees
features = current tabular features + representative_mrmr_ch
```

## Current Official 72h Model and CME Residual Check

Timestamp:

```text
2026-06-26 KST
```

Current official best:

```text
Script family:
  train_tabular_models_72h.py
  run_ch_feature_addition_72h.py

Feature set:
  current tabular lag/rolling features
  + representative_mrmr_ch

Model:
  0.70 * Direct MLP
  + 0.30 * ExtraTrees

Target:
  Speed(t + 72h)
```

CME residual correction experiment:

```text
Script:
  run_cme_residual_correction_72h.py

CME catalog:
  data/donki_cme_catalog.csv
  rows = 7,667
  coverage = 2011-01-02 18:00:00 through 2025-12-31 17:12:00

Residual target:
  Speed(t + 72h) - base_pred(t)

CME features:
  causal event windows ending at forecast origin t
  last 24h, last 48h, last 72h, last 96h

Correction model:
  0.70 * residual Direct MLP
  + 0.30 * residual ExtraTrees
```

Causality safeguards:

```text
- CME events are used only when cme_time <= origin time t.
- Public residual labels use out-of-fold base predictions.
- Private diagnostics train residual correction only on public OOF residuals.
- Private labels are not used for model selection.
```

Public fixed validation, 2022-2023:

```text
base_model_only:
  MAE  = 64.071681
  RMSE = 82.058349
  CC   = 0.444030

base_model_plus_cme_residual_correction:
  MAE  = 71.617986
  RMSE = 86.731100
  CC   = 0.443873
```

Private diagnostic, 2024-2025:

```text
base_model_only:
  MAE  = 61.379343
  RMSE = 85.989385
  CC   = 0.580837

base_model_plus_cme_residual_correction:
  MAE  = 74.787928
  RMSE = 97.514671
  CC   = 0.471723
```

Selection result:

```text
Selected model: base_model_only
Adopt CME residual correction: False
Reason: CME residual correction did not improve public fixed CC.
```

Solar-wind speed profile artifacts generated locally:

```text
outputs/cme_residual_correction_72h/solar_wind_speed_profile_reference_base_corrected_2024.png
outputs/cme_residual_correction_72h/solar_wind_speed_profile_reference_base_corrected_2025.png
outputs/cme_residual_correction_72h/solar_wind_speed_profile_reference_base_corrected_2024_2025.csv
```

These generated outputs are recorded here but are not required to be committed
to the code-only GitHub repository.

## Repository Push Policy

The GitHub repository should contain source code, documentation, lightweight
configuration, and experiment records. It should not contain raw data,
downloaded FITS/magnetogram files, large generated outputs, local virtual
environments, caches, or run logs.

## Corrected Task: 72h Block Multihorizon Forecast

Timestamp:

```text
2026-06-26 KST
```

Important problem-definition correction:

```text
The task is not an hourly rolling-origin Speed(t+72h) forecast.

The task is:
  issue one forecast every 72 hours
  predict the full next 72-hour hourly speed profile
  horizons h = 1, 2, ..., 72
  concatenate non-overlapping forecast blocks
```

Implemented script:

```text
run_multihorizon_72h_blocks.py
```

Model family:

```text
For each horizon h:
  MLP_h
  ExtraTrees_h
  ensemble_h = 0.70 * MLP_h + 0.30 * ExtraTrees_h

Features:
  current tabular lag/rolling features
  + representative_mrmr_ch

CME residual correction:
  not used
```

Forecast/evaluation setup:

```text
Fixed public validation:
  train target period = 2011-01-01 00:30 through 2021-12-31 23:30
  validation target period = 2022-01-01 00:30 through 2023-12-31 23:30

Private diagnostic:
  train target period = 2011-01-01 00:30 through 2023-12-31 23:30
  prediction target period = 2024-01-01 00:30 through 2025-12-31 23:30
```

Public fixed validation, 2022-2023:

```text
direct_mlp:
  N    = 17,292
  MAE  = 52.151457
  RMSE = 68.983788
  CC   = 0.648927

extratrees:
  N    = 17,292
  MAE  = 49.634971
  RMSE = 65.257940
  CC   = 0.688737

ensemble_0p7_mlp_0p3_extratrees:
  N    = 17,292
  MAE  = 50.661615
  RMSE = 66.936198
  CC   = 0.669880
```

Private diagnostic, 2024-2025:

```text
direct_mlp:
  N    = 17,280
  MAE  = 52.457317
  RMSE = 75.065633
  CC   = 0.703438

extratrees:
  N    = 17,280
  MAE  = 52.113039
  RMSE = 73.926865
  CC   = 0.718296

ensemble_0p7_mlp_0p3_extratrees:
  N    = 17,280
  MAE  = 51.162601
  RMSE = 73.312121
  CC   = 0.717528
```

Private yearly diagnostics for the ensemble:

```text
2024:
  N    = 8,581
  MAE  = 46.247362
  RMSE = 65.281647
  CC   = 0.577928

2025:
  N    = 8,699
  MAE  = 56.011165
  RMSE = 80.452119
  CC   = 0.676448
```

Output files:

```text
outputs/multihorizon_72h_blocks/summary.csv
outputs/multihorizon_72h_blocks/fixed_results.csv
outputs/multihorizon_72h_blocks/private_diagnostic.csv
outputs/multihorizon_72h_blocks/private_yearly_diagnostic.csv
outputs/multihorizon_72h_blocks/horizon_metrics.csv
outputs/multihorizon_72h_blocks/block_metrics.csv
outputs/multihorizon_72h_blocks/best_private_prediction.csv
outputs/multihorizon_72h_blocks/origin_target_mapping.csv
outputs/multihorizon_72h_blocks/timestamp_sanity_check.csv
outputs/multihorizon_72h_blocks/config.json
```

Sanity-check result:

```text
best_private_prediction.csv rows = 17,544
first target timestamp = 2024-01-01 00:30:00
last target timestamp  = 2025-12-31 23:30:00
duplicate target timestamps = 0
all private target timestamps covered exactly once = True
all feature timestamps <= forecast origin = True
```

## Detailed Final Report Pointer

The comprehensive final documentation for the corrected 72h block
multi-horizon forecasting task is:

```text
FINAL_REPORT_72H_BLOCK_MULTIHORIZON.md
```

That report summarizes the final task definition, data, feature engineering,
model scheme, splits, metrics, final results, sanity/leakage checks, rejected
experiments, and file organization.

## Multihorizon CME/context residual correction retry

Timestamp:

```text
2026-06-26 KST
```

Implemented script:

```text
run_multihorizon_correction_72h_blocks.py
```

Purpose:

```text
Retry the correction model for the corrected final task:
72h block multi-horizon forecast.
```

Base model:

```text
current final multihorizon ensemble
0.70 * MLP_h + 0.30 * ExtraTrees_h for each h=1..72
```

Correction target:

```text
residual(t,h) = true Speed(t+h) - base_pred(t,h)
corrected_pred(t,h) = base_pred(t,h) + residual_pred(t,h)
```

Correction model:

```text
horizon-aware pooled residual model using horizon_hour, base_pred,
current/context features, representative_mrmr_ch, CME recent-window features,
and ETA-to-target CME features
```

CME causality:

```text
only CME events with cme_time <= forecast_origin were used
private labels were not used for correction training or selection
```

Public fixed validation:

```text
base_multihorizon_only:
  N    = 17,292
  MAE  = 52.639729
  RMSE = 69.244519
  CC   = 0.640828

base_plus_horizon_cme_context_correction:
  N    = 17,292
  MAE  = 64.430012
  RMSE = 81.035165
  CC   = 0.564314
```

Private diagnostic:

```text
base_multihorizon_only:
  N    = 17,280
  MAE  = 51.162601
  RMSE = 73.312121
  CC   = 0.717528

base_plus_horizon_cme_context_correction:
  N    = 17,280
  MAE  = 56.958972
  RMSE = 78.298235
  CC   = 0.674677
```

Selection result:

```text
Correction is not adopted.
Reason: it worsened public fixed validation and also worsened private diagnostics.
The final selected model remains the base 72h block multi-horizon ensemble.
```

Sanity checks:

```text
private target rows = 17,544
missing private target timestamps = 0
duplicate private target timestamps = 0
forecast origin spacing = 72h
horizon range = 1..72
target_time = origin_time + horizon_hour
all CME events used have cme_time <= origin_time
private labels were not used for correction training
h=72 base private CC = 0.567183913, matching the previous final h=72 reference
```

Output directory:

```text
outputs/multihorizon_correction_72h_blocks/
```

The final selected model remains the uncorrected 72h block multi-horizon base
ensemble.
