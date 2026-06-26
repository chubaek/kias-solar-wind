# Solar Wind Speed Forecasting

2011–2023년 public 자료로 모델을 개발하고, 2024–2025년 private 자료에서 태양풍 속도를 예측하는 그룹 미니 컴피티션입니다.

> GitHub code release note: this repository is maintained for code,
> documentation, and experiment records. Large raw data, FITS downloads,
> generated intermediate files, and bulk model outputs are intentionally
> excluded from Git. Recreate or download data locally before running scripts.

---

## 1. 데이터

### 파일 구성

| 위치 | 내용 |
|------|------|
| 이 저장소 | 코드, 문서, 실험 기록 |
| 로컬 작업공간 | CSV 데이터, 중간 산출물, 학습/평가 출력 |
| [HuggingFace Dataset](https://huggingface.co/datasets/mingyujeon/kias-solar-wind) | SDO/AIA 193Å FITS (6.1 GB), 시각화 PNG (5.1 GB) |

| 파일 | 기간 | 행 수 | 용도 |
|------|------|------|------|
| `solar_wind-public.csv` | 2011–2023 | 113,952 | 모델 개발용 |
| `solar_wind-private.csv` | 2024–2025 | 17,544 | 최종 holdout 평가용 |
| `solar_wind_data.csv` | 2011–2025 | — | 전체 결합 데이터 |

위 CSV 파일들은 실행에 필요한 로컬 데이터 파일이지만 GitHub code
repository에는 포함하지 않습니다. 원본 자료에서 재생성하거나 별도 데이터
저장소에서 내려받아 repository root에 배치한 뒤 실행합니다.

### 변수 설명

모든 CSV는 아래 7개 열을 공유하며, 1시간 간격의 시계열이다.

| 열 | 단위·의미 | 모델에서의 역할과 주의점 |
|----|------|------|
| `datetime` | UTC, 1시간 간격 | 행 정렬, lag와 target 생성, 시간 split에 사용합니다. |
| `Speed (km/s)` | 지구 근처 태양풍 속도 | **예측 대상.** target horizon h 뒤 값은 target이고, forecast origin까지의 과거값은 autoregressive feature로 사용할 수 있습니다. |
| `Density (1/cm^3)` | 태양풍 양성자 수밀도 | CIR/HSS 압축영역과 관련됩니다. |
| `Temperature (K)` | 태양풍 양성자 온도 | 고속 태양풍에서 함께 상승할 수 있습니다. |
| `B (nT)` | 행성간 자기장 크기 | 방향이 없는 total magnitude입니다. |
| `Sunspot Number` | 일별 흑점 수 | 전체 태양활동 수준의 장기 proxy이며 같은 날짜에 반복됩니다. |
| `Coronal Hole Area` | 0–1 사이의 일별 면적 비율 | 매일 00:00 UT 영상에서 계산해 같은 UTC 날짜의 24개 시간 행에 붙였습니다. CH가 없는 `0.0`과 결측치 `NaN`은 다릅니다. |

### 소스

- **Speed, Density, Temperature, B**: NASA OMNI2 1시간 해상도 데이터. OMNI fill value는 `NaN`으로 대체됨.
- **Sunspot Number**: SIDC/WDC 국제 흑점 수.
- **Coronal Hole Area**: JSOC의 SDO/AIA 193 Å synoptic FITS를 `extract_ch_area.py`로 처리.

### Public 결측률

| 열 | 결측 행 | 결측률 |
|---|---:|---:|
| Speed | 452 | 0.397% |
| Density | 1,014 | 0.890% |
| Temperature | 556 | 0.488% |
| B | 272 | 0.239% |
| Sunspot Number | 0 | 0.000% |
| Coronal Hole Area | 720 | 0.632% |

결측률은 낮지만 처리 방식도 모델의 일부입니다.

- Target Speed가 `NaN`인 표본은 학습·평가에서 제외합니다.
- Predictor는 causal forward fill, train-only imputation 또는 NaN을 지원하는 모델을 사용할 수 있습니다.
- 양방향 interpolation처럼 미래값을 이용하는 처리는 피합니다.
- CH가 결측인 날짜에 전날 값을 무기한 전달하지 않습니다.

---

## 2. 예측 목표

**2024–2025년 태양풍 속도(`Speed (km/s)`)를 예측**한다. 예측값은 아래 형식의 CSV로 제출한다.

```csv
datetime,predicted_speed
2024-01-01 00:30:00,401.2
2024-01-01 01:30:00,398.7
```

`submissions/` 폴더에 CSV를 저장한 뒤 `evaluate_submissions.py`의 `SUBMISSION_FILES`에 경로를 추가하고 실행합니다.

```bash
uv run evaluate_submissions.py
```

---

## 3. 왜 public과 private로 나누었나?

이 분할의 의도는 **과거 자료로 모델을 개발하고, 개발 과정에서 사용하지 않은 연속된 미래 기간에서 일반화 성능을 확인하는 것**입니다. 태양풍은 인접 시간의 상관이 강하므로 전체 자료를 무작위로 나누면 비슷한 시점이 train과 test에 함께 들어가 성능이 낙관적으로 보일 수 있습니다. 따라서 2011–2023은 모델과 전처리를 결정하는 development 기간으로, 그보다 미래인 2024–2025는 확정된 모델을 평가하는 holdout 기간으로 분리했습니다. 즉, 파일을 숨기기 위한 구분이 아니라 **모델 선택과 최종 평가를 분리하기 위한 실험 설계**입니다.

Kaggle competition에 비유하면 public 기간은 모델을 개발하고 public score를 확인하는 **public leaderboard 단계**, private 기간은 최종 순위를 결정하는 **private leaderboard 단계**와 비슷합니다.

여기서 private는 파일이 비밀이라는 뜻이 아니라 **development에 사용하지 않는 holdout 기간**이라는 뜻입니다. 파일을 가지고 있어도 다음 작업에는 사용하지 않습니다.

- Private 평균·중앙값·분포를 보고 feature나 모델 결정
- Private score를 보고 hyperparameter 수정
- Private 성능이 좋아질 때까지 반복 실행

Private를 본 뒤 모델을 수정하면 private는 test가 아니라 또 하나의 validation이 됩니다. Private 평가 직전에 코드와 설정을 저장하고, 평가 후에는 결과를 해석만 하는 것을 원칙으로 합니다.

Public은 약 13년으로 모델 개발에 충분히 길고, private는 최근 2년의 연속된 미래 구간입니다. Public에도 Solar Cycle 24와 Cycle 25 상승기 일부가 들어 있고, private는 Cycle 25의 높은 활동 수준을 포함하므로 시간 변화와 solar-cycle-related distribution shift를 함께 시험할 수 있습니다.

---

## 4. Coronal Hole Area 계산법

이 값은 태양 전체의 물리적 면적(km²)이 아니라, **중앙자오선 주변 띠에서 CH로 판정된 픽셀의 비율**입니다.

1. 매일 00:00 UT의 SDO/AIA 193 Å synoptic FITS를 사용합니다.
2. 픽셀을 Heliographic Stonyhurst 경도·위도로 변환합니다.
3. 태양 원반 안에서 경도 ±7.5°, 위도 ±60°인 띠를 선택합니다.
4. 태양 원반 전체 밝기의 중앙값을 계산합니다.
5. 중앙값의 30%보다 어두운 띠 안의 픽셀을 CH로 판정합니다.
6. `CH 픽셀 수 / 띠 전체 픽셀 수`를 일별 면적 비율로 저장합니다.
7. 계산하지 못한 날은 `NaN`으로 둡니다.

따라서 현재 CH 변수는 위치, 형상, 위도별 분포, 자기장 극성을 모두 표현하지 않는 1차원 지표입니다.

---

## 5. Time Series Forecasting 접근법

### Single-horizon model

forecast origin `t`마다 `Speed(t + h)` 하나만 출력한다.

```text
[x(t-L), ..., x(t)] → Speed(t + h)
```

구조가 단순하고 h별로 모델을 독립적으로 최적화할 수 있다. 여러 horizon을 예측하려면 h마다 별도 모델을 학습해야 한다. 특정 horizon(4일)만 요구할 때 유리하다.

### Multi-horizon model (Direct multi-step)

단일 모델이 여러 horizon을 동시에 출력한다.

```text
[x(t-L), ..., x(t)] → [Speed(t+1h), Speed(t+2h), ..., Speed(t+H)]
```

한 번의 추론으로 여러 미래 시점을 얻을 수 있어 효율적이다. 각 horizon마다 독립 head를 두거나 sequence-to-sequence 구조를 사용한다. 모든 horizon에 동시에 최적화되므로 특정 horizon에서 single-horizon 모델보다 성능이 낮을 수 있다.

### Recursive / Autoregressive

1시간 또는 다음 타임스텝 예측 모델을 반복 호출해 원하는 horizon까지 이어간다.

```text
Speed(t+1h) → Speed(t+2h) → ... → Speed(t+H)
```

예측 오차가 누적(error accumulation)되므로 horizon이 길수록 성능이 떨어지는 경향이 있다. 태양풍처럼 horizon이 수일(수십~수백 스텝)인 경우 직접 예측(direct)이 유리한 경우가 많다.

---

## 6. ESWF 참고 모델

ESWF(Empirical Solar Wind Forecast)는 중앙자오선 주변 CH 면적과 약 4일 뒤 지구 태양풍 속도의 경험적 관계를 사용합니다.

```text
Coronal Hole Area(t) → Speed(t + 4 days)
```

- [Vršnak et al. (2007)](https://link.springer.com/article/10.1007/s11207-007-0285-8): CH 면적·위치와 HSS 특성
- [Rotter et al. (2015)](https://arxiv.org/abs/1501.06697): 평균 지연 `4.02 ± 0.5일`
- [Reiss et al. (2016)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016SW001390): Operational ESWF 검증
- [Milošić et al. (2023)](https://link.springer.com/article/10.1007/s11207-022-02102-5): 개선된 ESWF 2.0

현재 CSV는 ESWF 아이디어를 참고했지만 operational ESWF를 완전히 복제한 것은 아닙니다. ESWF는 시간에 따라 변하는 CH 자료와 최근 Carrington rotation을 사용해 경험식의 계수를 갱신합니다.

---

## 7. Public에서 모델을 개발하는 방법

### Train, validation, test

- **Train**: 모델 계수를 학습합니다.
- **Validation**: feature, 모델, hyperparameter를 선택합니다.
- **Public test**: 선택한 방법을 public의 미래 구간에서 확인합니다.
- **Private test**: public에서 확정한 방법을 2024–2025에서 최종 확인합니다.

### Cross-validation의 의미와 목적

Cross-validation(CV)은 public을 여러 train·validation 조합으로 나누어 반복 평가하는 선택적 방법입니다. 이 프로젝트에서 반드시 사용해야 하는 것은 아닙니다.

- 특정 기간에서 우연히 잘 나온 모델을 고를 위험을 줄입니다.
- 서로 다른 계절과 태양활동 상태에서 모델을 확인합니다.
- Fold별 평균과 표준편차로 성능과 안정성을 함께 비교합니다.

CV는 새로운 데이터를 만드는 data augmentation이 아니며 private test를 대체하지 않습니다. 일반 K-fold처럼 시간을 무작위로 섞으면 미래 자료가 train에 들어갈 수 있으므로 시계열에서는 시간 순서를 고려해야 합니다.

---

## 8. 사용할 수 있는 split 예시

아래 방법은 가능한 선택지를 설명하기 위한 예시입니다. 모든 split을 반드시 구현하거나 성능을 비교할 필요는 없습니다. 목적과 계산시간에 맞는 방법 하나를 선택하고, 선택 이유와 미래정보 누출 여부를 설명하면 됩니다. 가장 단순한 방법은 S4이고, 조금 더 안정적인 평가가 필요하면 S5를 사용할 수 있습니다.

### S1. Random hourly split

시간 행을 무작위로 나눕니다. 구현은 쉽지만 인접 시점과 동일 날짜의 CH가 train/test에 섞이므로 실제 시계열 forecasting에는 권장하지 않습니다.

### S2. Fixed 8/2/2-month split

모든 연도에서 같은 8개월을 train, 2개월을 validation, 2개월을 test로 사용합니다. Test 계절이 고정되고 같은 연도의 미래 월이 train에 포함될 수 있습니다.

### S3. Shifted 8/2/2-month split

S2의 월 배치를 매년 4개월씩 이동합니다. 계절 편향은 줄지만 같은 연도의 미래 자료로 과거를 평가할 수 있어 실제 미래 예보와는 차이가 있습니다.

### S4. Fixed chronological split

```text
Train:       2011–2019
Validation:  2020–2021
Public test: 2022–2023
```

과거로 학습해 미래를 평가하므로 구현과 설명이 쉽습니다. 한 번의 기간 분할에 결과가 민감할 수 있습니다.

### S5. Expanding-window time-series cross-validation

```text
Fold 1: Train 2011–2016 / Validation 2017 / Test 2018
Fold 2: Train 2011–2017 / Validation 2018 / Test 2019
...
Fold 6: Train 2011–2021 / Validation 2022 / Test 2023
```

학습기간을 점차 늘리면서 항상 미래 연도를 평가합니다. Rolling-origin evaluation 또는 forward-chaining CV라고도 합니다. 시간 순서를 지키면서 여러 미래 구간의 성능 변동을 확인하고 싶을 때 사용할 수 있는 비교적 안정적인 방법입니다.

### Split 경계 주의

target을 만들 때 train 입력에 연결된 정답이 validation/test 기간으로 넘어가면 안 됩니다. Target timestamp 기준으로 표본을 나누거나 각 경계에서 최소 h 시간을 제거합니다.

Scaler, imputer, feature selection은 각 fold의 train에서만 `fit`하고 validation/test에는 `transform`만 적용합니다.

---

## 9. Baseline

### Baseline 설명

**Public mean (Climatology)**

public 기간(2011–2023)의 평균 `417.20 km/s`를 모든 private 시점에 상수로 예측한다. 어떤 시간 변화도 추적하지 않는 가장 단순한 기준이다.

**27-day persistence**

태양이 약 27일마다 같은 면을 지구를 향하는 자전 주기를 이용한다.

```text
prediction(t) = Speed(t - 648 hours)
```

태양풍의 가장 강력한 물리적 baseline이다. 고정 학습 모델이 아니라 관측 스트림이 순차적으로 들어오는 walk-forward model로, 각 시점까지 이미 관측된 Speed만 사용한다.

### Private baseline evaluation (2024–2025, N=17,280)

| Baseline | MAE (km/s) | RMSE (km/s) | CC | MAE Skill vs 27-day |
|---|---:|---:|---:|---:|
| Public mean (`417.20`) | **77.82** | **108.85** | — | **+0.025** |
| 27-day persistence | 79.85 | 111.39 | **0.427** | 0.000 |

Public mean은 MAE·RMSE 기준으로 27-day persistence보다 약간 좋지만 CC가 정의되지 않아 HSS 도착 시점을 전혀 맞히지 못한다. **개발 모델은 MAE Skill > 0이면서 CC > 0.427을 동시에 달성하는 것을 목표로 한다.**

### 평가 지표

```text
MAE   = mean(|y_i - p_i|)
RMSE  = sqrt(mean((y_i - p_i)^2))
CC    = Pearson correlation(y_i, p_i)
MAE Skill = 1 - (Model MAE / 27-day persistence MAE)
```

- `Skill > 0`: 27-day persistence보다 좋음
- `Skill = 0`: 27-day persistence와 같음
- `Skill < 0`: 27-day persistence보다 나쁨

Target Speed가 `NaN`인 시점은 평가에서 제외합니다. 모든 모델과 baseline은 같은 target 시점에서 비교합니다.

---

## 10. Current 72h Modeling Status

중요한 문제 정의 수정 후 현재 task는 매시간 origin을 굴리는
`Speed(t+72h)` 단일 예측이 아니라, 72시간마다 한 번 forecast를 발행하고
다음 72시간의 hourly profile을 직접 예측하는 block forecast입니다.

```text
Forecast schedule:
  one forecast origin every 72 hours
  horizons h = 1, 2, ..., 72
  concatenate non-overlapping forecast blocks

Feature set:
  current tabular lag/rolling features
  + representative_mrmr_ch coronal-hole morphology/intensity features

Model:
  one Direct MLP and one ExtraTrees model per horizon
  ensemble_h = 0.70 * MLP_h + 0.30 * ExtraTrees_h

Target:
  Speed(t + h), for h = 1..72
```

Latest script:

```text
run_multihorizon_72h_blocks.py
```

Public fixed validation, 2022–2023:

| Model | MAE | RMSE | CC |
|---|---:|---:|---:|
| Direct MLP | 52.1515 | 68.9838 | 0.648927 |
| ExtraTrees | 49.6350 | 65.2579 | 0.688737 |
| 0.7 MLP + 0.3 ExtraTrees | 50.6616 | 66.9362 | 0.669880 |

Private diagnostic, 2024–2025:

| Model | MAE | RMSE | CC |
|---|---:|---:|---:|
| Direct MLP | 52.4573 | 75.0656 | 0.703438 |
| ExtraTrees | 52.1130 | 73.9269 | 0.718296 |
| 0.7 MLP + 0.3 ExtraTrees | 51.1626 | 73.3121 | 0.717528 |

The private prediction file covers `2024-01-01 00:30:00` through
`2025-12-31 23:30:00` exactly once per target timestamp. CME residual
correction is not used in this corrected block-forecast task.

---

## 11. 권장 조별 진행 순서

1. Public 데이터와 결측치를 탐색합니다.
2. target과 causal lag feature를 만듭니다.
3. Baseline을 구현합니다.
4. Public 안에서 사용할 validation split 하나를 정합니다.
5. Public에서 모델과 hyperparameter를 결정합니다.
6. 선택 이유와 public 결과를 기록하고 코드를 저장합니다.
7. 확정 모델을 public 전체로 재학습합니다.
8. Private를 한 번 평가합니다.
9. Public–private 성능 차이, leakage, 계절 및 Solar Cycle 의존성을 토론합니다.

---

## 12. 최종 결과물

- 재현 가능한 전처리·학습·예측 코드
- 사용한 split, feature, 모델과 선택 이유
- Public baseline 및 모델 비교표
- Private 최종 성능
- Public과 private 성능 차이에 대한 해석
- 한계와 후속 개선 아이디어

---

## 13. 시작하기

```bash
git clone https://github.com/chubaek/kias-solar-wind
cd kias-solar-wind
uv sync
```

FITS/PNG가 필요한 경우 HuggingFace에서 추가로 다운로드합니다.

```bash
# 전체 다운로드
uvx hf download mingyujeon/kias-solar-wind --repo-type dataset --local-dir .

# FITS만 필요한 경우
uvx hf download mingyujeon/kias-solar-wind --repo-type dataset --local-dir . --include "aia193_download/*"

# PNG만 필요한 경우
uvx hf download mingyujeon/kias-solar-wind --repo-type dataset --local-dir . --include "aia193_plot/*"
```

---

## 14. 데이터 생성 파이프라인

GitHub code repository에는 생성된 CSV가 포함되지 않으므로, 모델 재현 시
아래 과정을 실행하거나 동일한 형식의 CSV를 별도로 배치해야 합니다.

```text
OMNI 원본 TXT
  └─ omni.py
       └─ sw_data.csv

SDO/AIA 193 Å FITS
  └─ extract_ch_area.py
       └─ ch_area_output.csv

sw_data.csv + ch_area_output.csv
  └─ prepare_competition_data.py
       ├─ solar_wind_data.csv
       ├─ solar_wind-public.csv
       └─ solar_wind-private.csv

solar_wind-public.csv + solar_wind-private.csv
  ├─ make_baseline_submissions.py → submissions/baseline_*.csv
  └─ evaluate_submissions.py → submissions/evaluation_ranking.csv
```

```bash
uv run omni.py                       # OMNI2 TXT → sw_data.csv
uv run extract_ch_area.py            # AIA FITS → ch_area_output.csv
uv run prepare_competition_data.py   # 결합 및 public/private 분할
uv run omni_plot.py                  # 전체 기간 시각화 → solar_wind_overview.png
```

---

## 15. 유사 사례

NOAA/NASA의 [MagNet competition](https://www.drivendata.org/competitions/73/noaa-magnetic-forecasting/)은 실시간 태양풍 자료로 Dst를 예측하고 미래 기간에서 최종 평가했습니다. NASA CCMC의 [SIML-HSS](https://ccmc.gsfc.nasa.gov/models/SIML-HSS~1/)는 코로날 홀과 과거 태양풍을 이용해 4-day solar-wind forecast를 수행합니다.
