# mRMR 상위 10개 CH feature 해석

이 문서는 `mrmr_selected_features.csv`의 상위 10개 feature를 물리적으로 해석한 것이다. 현재 결과는 다음 설정으로 생성되었다.

- Target: `solar_wind_speed_kms`
- Lag: CH parameter가 solar wind speed보다 1-7일 선행
- Matching: `match_mode = ffill`, `ffill_limit_hours = 6`
- mRMR relevance: `abs(Spearman(feature, Vsw))`
- mRMR redundancy: 이미 선택된 feature들과의 평균 `abs(Spearman)`

중요한 주의점은 Spearman correlation과 mRMR이 인과성을 증명하지 않는다는 점이다. 아래 해석은 feature 정의와 알려진 CH-HSS 연결 물리를 바탕으로 한 물리적 독해다.

## Window 표기

`W_lonL_latB`는 다음 joint window를 뜻한다.

```text
|Stonyhurst longitude from central meridian| < L deg
|heliographic latitude - B0| < B deg
```

즉 `W_lon30_lat30`은 중앙자오선 기준 경도 ±30도, 지구가 보는 태양 위도 기준 ±30도 안에 들어온 코로나홀 영역이다.

## 상위 10개 요약

| Rank | Feature | Lag | rho | 의미 요약 |
|---:|---|---:|---:|---|
| 1 | `I_mean_W_lon7p5_lat15` | 4d | -0.478 | 중앙자오선 극근처, 저위도 CH의 평균 어두움 |
| 2 | `A_W_lon30_lat30_km2` | 5d | +0.396 | 지구방향 저위도 CH 면적 |
| 3 | `A_source_if_overlap_W_lon60_lat60_km2` | 4d | +0.242 | 넓은 Earth-facing window와 겹치는 CH 전체 source 면적 |
| 4 | `I_mean_W_lon7p5_lat15` | 5d | -0.472 | Rank 1과 같은 물리량의 5일 lag |
| 5 | `A_W_lon30_lat15_km2` | 4d | +0.323 | 더 엄격한 저위도 중앙부 CH 면적 |
| 6 | `log_I_mean_W_lon7p5_lat15` | 4d | -0.478 | Rank 1의 로그 변환 |
| 7 | `lat_width_eff_10_full_if_overlap_largest_W_lon7p5_lat15` | 3d | +0.235 | 중앙 core와 겹친 dominant CH의 latitudinal extent |
| 8 | `A_frac_visible_hemisphere_W_lon30_lat30` | 5d | +0.396 | Rank 2의 visible hemisphere 정규화 비율 |
| 9 | `log_I_mean_W_lon7p5_lat15` | 5d | -0.472 | Rank 4의 로그 변환 |
| 10 | `A_grid_4x3_lat1_lon1_km2` | 4d | +0.311 | 중앙 경도, 남반구 저위도 grid area |

## Feature별 물리적 의미

### 1. `I_mean_W_lon7p5_lat15__lag_4d`

`W_lon7p5_lat15`는 중앙자오선 주변 ±7.5도, 위도 ±15도의 매우 좁은 Earth-facing core다. 이 영역은 ESWF 계열의 central meridional slice와 가장 가까운 위치 정보다.

`I_mean`은 그 window 안에 들어온 CH 영역의 평균 AIA 193 intensity다. Spearman rho가 음수라는 것은 평균 intensity가 낮을수록, 즉 CH가 더 어두울수록 4일 뒤 solar wind speed가 높아지는 경향이 있다는 뜻이다. 물리적으로는 더 어두운 EUV CH가 더 낮은 밀도 또는 더 강한 open-field CH 성격을 가질 가능성을 반영하는 proxy로 볼 수 있다.

해석 주의: intensity는 AIA calibration, exposure/preprocessing, solar cycle background, SPoCA segmentation 품질의 영향을 받는다. 따라서 raw intensity를 절대 물리량으로 해석하기보다 darkness/quality proxy로 보는 것이 안전하다.

### 2. `A_W_lon30_lat30_km2__lag_5d`

`W_lon30_lat30` 안의 CH 면적이다. 중앙자오선 ±30도, 위도 ±30도는 지구방향 HSS source로 가장 해석하기 쉬운 넓은 geoeffective window다.

rho가 양수이므로 이 window 안의 CH 면적이 클수록 약 5일 뒤 solar wind speed가 높아지는 경향이 있다. 이는 CH 면적과 HSS speed 사이의 경험적 연결과 잘 맞는다. 4일이 아니라 5일 lag가 선택된 것은 이 데이터셋의 segmentation cadence, CH 중심 위치, HSS propagation time, stream interaction region 형성 위치가 합쳐진 경험적 최적점으로 해석해야 한다.

### 3. `A_source_if_overlap_W_lon60_lat60_km2__lag_4d`

이 feature는 `W_lon60_lat60` 안에 들어온 조각의 면적이 아니라, 그 window와 조금이라도 겹친 CH event의 전체 polygon 면적 합이다.

물리적 의미는 “지구방향 넓은 영역과 연결될 가능성이 있는 source CH 자체가 얼마나 큰가”다. `clipped area`가 지구와 직접 연결될 가능성이 큰 부분을 재는 반면, `source_if_overlap`은 CH 전체 source reservoir의 크기를 보존한다. HSS peak speed와 duration에는 window 안쪽 조각뿐 아니라 전체 CH의 longitudinal/latitudinal extent가 영향을 줄 수 있으므로 이 feature가 선택된 것은 타당하다.

### 4. `I_mean_W_lon7p5_lat15__lag_5d`

Rank 1과 같은 물리량이지만 lag가 5일이다. 4일과 5일 lag가 모두 선택된 것은 중앙 core CH darkness 신호가 단일 시점이 아니라 4-5일 전 기간에 걸쳐 유효하다는 뜻일 수 있다.

다만 같은 feature의 인접 lag는 서로 강하게 상관될 가능성이 크다. 모델링 단계에서는 4일 lag와 5일 lag를 둘 다 넣는 모델과 하나만 넣는 모델을 비교하는 ablation이 필요하다.

### 5. `A_W_lon30_lat15_km2__lag_4d`

`W_lon30_lat15`는 경도 ±30도는 유지하되 위도를 ±15도로 더 엄격하게 제한한 저위도 중심 면적이다.

이 feature는 “지구 위도와 매우 가까운 CH area”를 나타낸다. 같은 면적이라도 고위도 CH보다 저위도 CH가 지구에서 관측되는 HSS와 더 직접적으로 연결될 가능성이 크다. 따라서 이 feature는 단순 central area보다 geoeffective source area에 더 가깝다.

Rank 2의 `W_lon30_lat30`과 비교하면, 이 feature는 위도 연결성을 더 강하게 본다. Rank 2가 넓은 geoeffective area라면 Rank 5는 저위도 core area다.

### 6. `log_I_mean_W_lon7p5_lat15__lag_4d`

Rank 1의 `I_mean`에 로그를 취한 feature다. Spearman correlation에서는 단조 변환이므로 `I_mean`과 거의 같은 순위 정보를 갖는다.

물리적 의미는 Rank 1과 동일하게 중앙 core CH darkness proxy다. 로그 변환은 intensity scale의 long-tail 또는 multiplicative drift를 완화하는 데 도움이 될 수 있지만, Spearman 기반 ranking에서는 raw `I_mean`과 정보가 거의 중복된다.

모델링 권장: 선형 모델이나 tree 모델에 둘 다 넣기보다 `I_mean` 또는 `log_I_mean` 중 하나를 선택하는 것이 더 해석적이다.

### 7. `lat_width_eff_10_full_if_overlap_largest_W_lon7p5_lat15__lag_3d`

이 feature는 `W_lon7p5_lat15`와 겹친 dominant CH의 전체 polygon에서 계산한 latitudinal effective width다. `eff_10`은 latitude별 area profile이 최대값의 10% 이상인 구간의 폭을 뜻한다.

`full_if_overlap`이므로 좁은 중앙 window 안에서 잘린 조각의 폭이 아니라, 중앙 core와 연결된 CH source 전체의 남북 폭이다.

물리적 의미는 dominant source CH가 얼마나 넓은 위도 범위에 걸쳐 있는가다. 위도 폭이 클수록 지구가 HSS 구조의 중심 또는 flank를 만날 가능성이 바뀔 수 있다. 특히 B0와 CH 중심 위도의 상대 위치에 따라 hit/miss와 speed profile이 달라질 수 있다.

주의: latitudinal width는 직접적으로 speed를 올리는 원인이라기보다 지구와 CH source의 연결 확률, flank encounter, duration 구조를 나타내는 형태학적 proxy에 가깝다.

### 8. `A_frac_visible_hemisphere_W_lon30_lat30__lag_5d`

Rank 2의 `A_W_lon30_lat30_km2`를 visible hemisphere 면적으로 나눈 정규화 feature다. 따라서 물리적 의미는 Rank 2와 거의 같다.

단위 면적이 아니라 fractional area이므로 서로 다른 기간이나 태양 반경 scale을 명시적으로 정규화한 값이다. 하지만 현재 구현에서는 같은 HEK area scale을 쓰므로 `A_W_lon30_lat30_km2`와 거의 완전히 같은 정보를 담는다.

모델링 권장: `A_W_lon30_lat30_km2`와 `A_frac_visible_hemisphere_W_lon30_lat30` 중 하나만 선택해도 충분할 가능성이 크다.

### 9. `log_I_mean_W_lon7p5_lat15__lag_5d`

Rank 4의 로그 변환이다. 물리적 의미는 중앙 core CH darkness의 5일 lag 신호다.

Rank 1, 4, 6, 9는 모두 같은 central-core intensity family다. mRMR이 이들을 여러 개 선택한 이유는 relevance가 매우 강하고, redundancy penalty가 완전히 배제 규칙은 아니기 때문이다. 최종 모델을 만들 때는 이 family에서 대표 feature를 1-2개로 줄이는 것이 좋다.

### 10. `A_grid_4x3_lat1_lon1_km2__lag_4d`

`A_grid_4x3`의 grid 정의는 다음과 같다.

```text
lat edges: [-60, -30, 0, 30, 60]
lon edges: [-60, -20, 20, 60]
```

따라서 `lat1_lon1`은 다음 grid cell이다.

```text
-30 deg < lat < 0 deg
-20 deg < lon < 20 deg
```

즉 중앙자오선 근처의 남반구 저위도 CH 면적이다. 이 feature가 선택된 것은 넓은 joint window보다 더 위치 분해된 central low-latitude area가 solar wind speed와 관련 있음을 시사한다.

물리적으로는 지구 연결성이 높은 중앙부 CH area를 더 세밀하게 분리한 feature다. 남반구 cell이 선택된 것은 분석 기간의 B0 변화, CH 분포 비대칭, 특정 solar cycle phase의 catalog 특성, 또는 실제 지구 연결 geometry가 반영된 결과일 수 있다. 북반구/남반구 비대칭은 추가 검증이 필요하다.

## 모델링 시 권장 정리

상위 10개를 그대로 모두 넣는 것은 가능하지만, 해석 모델에서는 중복 family를 정리하는 것이 좋다.

권장 대표 feature:

1. Central darkness: `I_mean_W_lon7p5_lat15__lag_4d` 또는 `log_I_mean_W_lon7p5_lat15__lag_4d`
2. Main geoeffective area: `A_W_lon30_lat30_km2__lag_5d`
3. Low-latitude core area: `A_W_lon30_lat15_km2__lag_4d`
4. Full source area: `A_source_if_overlap_W_lon60_lat60_km2__lag_4d`
5. Source morphology/width: `lat_width_eff_10_full_if_overlap_largest_W_lon7p5_lat15__lag_3d`
6. Grid-local area: `A_grid_4x3_lat1_lon1_km2__lag_4d`

중복성이 큰 feature:

- `I_mean_W_lon7p5_lat15` vs `log_I_mean_W_lon7p5_lat15`
- 4일 lag vs 5일 lag의 같은 intensity feature
- `A_W_lon30_lat30_km2` vs `A_frac_visible_hemisphere_W_lon30_lat30`

이 중복 feature들은 최종 모델에서 ablation으로 선택하는 것이 좋다.

