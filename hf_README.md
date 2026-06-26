---
license: cc-by-4.0
language:
  - ko
task_categories:
  - time-series-forecasting
tags:
  - solar-wind
  - space-weather
  - coronal-hole
  - SDO
  - AIA
  - OMNI
  - astronomy
size_categories:
  - 10K<n<100K
---

# KIAS Solar Wind Dataset

SDO/AIA 193Å FITS 이미지와 시각화 PNG를 포함한 태양풍 예측 데이터셋입니다.
코드, CSV 데이터, 문서는 GitHub 저장소를 참고하세요.

GitHub: [mgjeon/kias-solar-wind](https://github.com/mgjeon/kias-solar-wind)

---

## 데이터셋 구성

| 폴더 | 내용 | 크기 |
|------|------|------|
| `aia193_download/` | SDO/AIA 193Å synoptic FITS (일별, 2011–2025) | 6.1 GB |
| `aia193_plot/` | 코로날 홀 탐지 시각화 PNG (일별, 2011–2025) | 5.1 GB |

CSV 데이터(`solar_wind-public.csv`, `solar_wind-private.csv`, `solar_wind_data.csv`)는 GitHub 저장소에 포함되어 있습니다.

---

## 다운로드

```bash
# 전체 다운로드
uvx hf download mingyujeon/kias-solar-wind --repo-type dataset --local-dir .

# FITS만 필요한 경우
uvx hf download mingyujeon/kias-solar-wind --repo-type dataset --local-dir . --include "aia193_download/*"

# PNG만 필요한 경우
uvx hf download mingyujeon/kias-solar-wind --repo-type dataset --local-dir . --include "aia193_plot/*"
```

---

## FITS 파일

- 출처: [JSOC SDO/AIA Synoptic Archive](https://jsoc1.stanford.edu/data/aia/synoptic)
- 파일명 형식: `AIA{YYYYMMDD}_{HHMM}_0193.fits`
- 매일 00:00 UT 193Å synoptic 이미지
- 기간: 2011-01-01 – 2025-12-31

## PNG 파일

- 파일명 형식: `ch_area_{YYYY-MM-DD}_{HHMM}.png`
- FITS에서 코로날 홀(CH) 영역을 탐지한 결과 시각화
- `extract_ch_area.py`로 생성 ([GitHub 참고](https://github.com/mgjeon/kias-solar-wind))

---

## 코로날 홀 면적 계산법

1. 매일 00:00 UT SDO/AIA 193Å synoptic FITS 사용
2. 픽셀을 Heliographic Stonyhurst 경도·위도로 변환
3. 경도 ±7.5°, 위도 ±60° 중앙자오선 띠 선택
4. 태양 원반 전체 밝기 중앙값의 30%보다 어두운 픽셀을 CH로 판정
5. `CH 픽셀 수 / 띠 전체 픽셀 수` → 일별 면적 비율 (`Coronal Hole Area`)

---

## 데이터 출처

- **태양풍 데이터**: [NASA OMNI](https://omniweb.gsfc.nasa.gov/) (1시간 해상도)
- **태양 이미지**: [SDO/AIA](https://aia.lmsal.com/) via [JSOC](https://jsoc1.stanford.edu/)
- **흑점 수**: OMNI 데이터에 포함
