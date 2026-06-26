"""태양풍과 CH 자료를 합치고 public/private CSV로 나눈다."""

from pathlib import Path

import pandas as pd


# -----------------------------------------------------------------------------
# 1. 입력·출력 파일
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent

SW_CSV = HERE / "sw_data.csv"
CH_CSV = HERE / "ch_area_output.csv"

FULL_CSV = HERE / "solar_wind_data.csv"
PUBLIC_CSV = HERE / "solar_wind-public.csv"
PRIVATE_CSV = HERE / "solar_wind-private.csv"

PRIVATE_START = pd.Timestamp("2024-01-01")


# -----------------------------------------------------------------------------
# 2. 시간별 태양풍과 일별 CH 면적을 읽는다
# -----------------------------------------------------------------------------
solar_wind = pd.read_csv(SW_CSV, parse_dates=["datetime"])

coronal_hole = pd.read_csv(
    CH_CSV,
    usecols=["date", "ch_area"],
    parse_dates=["date"],
).rename(columns={"ch_area": "Coronal Hole Area"})

if solar_wind["datetime"].duplicated().any():
    raise ValueError("sw_data.csv에 중복 datetime이 있습니다.")

if coronal_hole["date"].duplicated().any():
    raise ValueError("ch_area_output.csv에 중복 date가 있습니다.")


# -----------------------------------------------------------------------------
# 3. 같은 UTC 날짜의 CH 면적을 시간별 태양풍 행에 붙인다
# -----------------------------------------------------------------------------
solar_wind["date"] = solar_wind["datetime"].dt.normalize()

merged = solar_wind.merge(
    coronal_hole,
    on="date",
    how="left",
    sort=False,
    validate="many_to_one",
)

merged = merged.drop(columns="date").sort_values("datetime").reset_index(drop=True)

# CH 자료가 없는 날짜는 이전 날짜 값으로 채우지 않고 NaN으로 유지한다.


# -----------------------------------------------------------------------------
# 4. 전체 자료를 2024-01-01 경계에서 public/private로 나눈다
# -----------------------------------------------------------------------------
public = merged[merged["datetime"] < PRIVATE_START].copy()
private = merged[merged["datetime"] >= PRIVATE_START].copy()


# -----------------------------------------------------------------------------
# 5. 세 CSV를 저장한다
# -----------------------------------------------------------------------------
outputs = {
    FULL_CSV: merged,
    PUBLIC_CSV: public,
    PRIVATE_CSV: private,
}

for output_path, data in outputs.items():
    data.to_csv(
        output_path,
        index=False,
        na_rep="NaN",
        date_format="%Y-%m-%d %H:%M:%S",
    )
    print(
        f"저장: {output_path} | {len(data):,} rows | "
        f"{data['datetime'].min()} ~ {data['datetime'].max()}"
    )
