# =============================================================================
#  AIA 193 Å synoptic 이미지에서 ESWF 방식 코로날 홀(CH) "area"를 뽑아 CSV로 저장
# -----------------------------------------------------------------------------
#  사용법:
#     uv run extract_ch_area.py
#     uv run extract_ch_area.py --plot-only
#     uv run extract_ch_area.py --cnn-export png
#     uv run extract_ch_area.py --cnn-export npy
#
#  아래 [설정] 부분의 값만 바꾸면 됩니다.
#     - 기간(START_DATE ~ END_DATE)
#     - 임계값(CH_FRAC)
#     - 자오선 띠 경도/위도 범위(LON_HALF, LAT_BAND)
#     - 하루 중 사용할 프레임 시각(FRAME_HHMM)
#
#  원리(한 문장): 매일 한 장의 193Å 디스크 이미지에서, 중앙자오선 근처 "띠" 안의
#  어두운 픽셀(=코로날 홀)이 차지하는 면적 비율을 계산한다.
# =============================================================================

# --- 표준 라이브러리 -------------------------------------------------------
import argparse               # 실행 모드(--plot-only) 선택
import glob                   # 다운로드된 FITS 파일 목록 찾기
import os                     # 폴더 생성, 경로 합치기
import re                     # FITS 파일명에서 날짜·시각 읽기
import urllib.request         # 인터넷에서 FITS 파일 내려받기

# --- 외부 라이브러리 -------------------------------------------------------
import numpy as np            # 배열 계산 (픽셀 통계)
import pandas as pd           # 표(날짜별 결과) 만들고 CSV로 저장
import astropy.units as u     # 각도 단위(deg) 등 물리 단위
import sunpy.map              # FITS를 "태양 지도(Map)" 객체로 읽기
from sunpy.coordinates import HeliographicStonyhurst   # 픽셀 -> 태양 경위도 변환
import matplotlib.pyplot as plt   # 결과를 그림(PNG)으로 그리기
from matplotlib import colors      # 로그 스케일 등 색 보정
from PIL import Image          # CNN용 단일채널 16-bit PNG 저장


# =============================================================================
#  [설정]  ←←← 여기 값만 바꾸세요
# =============================================================================
START_DATE = "2011-01-01"     # 시작일 (포함)
END_DATE   = "2026-01-01"     # 종료일 (포함)

CH_FRAC  = 0.3               # 코로날 홀 임계값: 디스크 밝기 중앙값의 몇 배보다 어두우면 CH
LON_HALF = 7.5               # 중앙자오선 기준 경도 ±몇 도까지를 "띠"로 볼지 (deg)
LAT_BAND = 60.0              # 적도 기준 위도 ±몇 도까지를 "띠"로 볼지 (deg)

FRAME_HHMM = "0000"           # 하루 중 사용할 프레임 시각 (HHMM). 예: "0000", "0030"
                              #  (30분 차이는 결과에 거의 영향 없음 — 검증 완료)

# 다운로드 받은 FITS를 저장할 폴더 / 결과 CSV 경로
from pathlib import Path
_HERE = Path(__file__).parent

DOWNLOAD_DIR = str(_HERE / "aia193_download")
OUTPUT_CSV   = str(_HERE / "ch_area_output.csv")

# 기존 FITS 전체의 날짜별 PNG 저장 폴더 (--plot-only에서만 사용)
PLOT_ONLY_DIR = str(_HERE / "aia193_plot")

# CNN 입력 저장 폴더 (--cnn-export에서만 사용)
CNN_PNG_DIR = str(_HERE / "aia193_cnn_png")
CNN_NPY_DIR = str(_HERE / "aia193_cnn_npy")

# AIA synoptic 아카이브 주소 (바꿀 필요 없음)
BASE_URL = "https://jsoc1.stanford.edu/data/aia/synoptic"


# =============================================================================
#  함수 1) 하루치 FITS 파일을 내려받아 로컬 경로를 돌려준다
# =============================================================================
def download_one_day(date):
    """date(Timestamp)에 해당하는 193Å FITS를 내려받고 저장 경로를 반환."""
    y, mo, d = date.year, date.month, date.day

    # 아카이브의 파일 이름 규칙: AIA{YYYYMMDD}_{HHMM}_0193.fits
    filename = f"AIA{y}{mo:02d}{d:02d}_{FRAME_HHMM}_0193.fits"

    # 전체 URL:  .../YYYY/MM/DD/H0000/AIA...._0193.fits
    #  (synoptic은 매시 H0000~H2300 폴더가 있고, 분 단위 프레임은 H0000 안에 모여 있음)
    url = f"{BASE_URL}/{y}/{mo:02d}/{d:02d}/H0000/{filename}"

    # 로컬 저장 경로
    local_path = os.path.join(DOWNLOAD_DIR, filename)

    # 이미 받아둔 파일이면 다시 받지 않는다 (시간 절약)
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(url, local_path)

    return local_path


# =============================================================================
#  함수 2) FITS 한 장에서 CH area(면적 비율) 하나를 계산한다
# =============================================================================
def compute_ch_area(fits_path):
    """FITS 경로를 받아 ESWF 방식 fractional CH area(0~1)와 디스크 중앙값을 반환."""

    # (a) FITS를 태양 지도 객체로 읽는다 — 헤더의 좌표 정보까지 함께 들어온다
    smap = sunpy.map.Map(fits_path)

    # (b) 모든 픽셀의 하늘 좌표를 구한 뒤, 태양 경위도(Stonyhurst)로 변환한다
    pixel_coords = sunpy.map.all_coordinates_from_map(smap)
    helio = pixel_coords.transform_to(HeliographicStonyhurst(obstime=smap.date))
    lon = helio.lon.to_value(u.deg)   # 각 픽셀의 경도(deg), 디스크 밖은 NaN
    lat = helio.lat.to_value(u.deg)   # 각 픽셀의 위도(deg), 디스크 밖은 NaN

    # (c) "태양 원반 안"인 픽셀만 True (경위도가 숫자로 나오는 곳)
    on_disk = np.isfinite(lon) & np.isfinite(lat)

    # (d) 그중에서도 "중앙자오선 띠"에 드는 픽셀만 True
    #     - 경도가 ±LON_HALF 안   AND   위도가 ±LAT_BAND 안
    in_slice = on_disk & (np.abs(lon) < LON_HALF) & (np.abs(lat) < LAT_BAND)

    # (e) 디스크 전체 밝기의 중앙값 — 임계값의 기준이 된다
    #     (중앙값 대비 비율을 쓰므로 노출시간/센서열화가 자동 상쇄된다)
    disk_median = np.nanmedian(smap.data[on_disk])

    # (f) 코로날 홀 픽셀: 띠 안에 있으면서 밝기가 임계값보다 어두운 곳
    is_coronal_hole = (smap.data < CH_FRAC * disk_median) & in_slice

    # (g) 면적 비율 = (띠 안 CH 픽셀 수) / (띠 전체 픽셀 수)
    ch_area = is_coronal_hole.sum() / in_slice.sum()

    return ch_area, disk_median


# =============================================================================
#  메인: 기간 안의 매일에 대해 다운로드 → 계산 → 표에 누적 → CSV 저장
# =============================================================================
def main():
    # 저장 폴더가 없으면 만든다
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # 시작일~종료일을 하루 간격 날짜 리스트로 만든다
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    results = []   # 날짜별 결과를 모을 리스트

    # 화면 출력 머리말
    print(f"기간 {START_DATE} ~ {END_DATE}  |  임계 {CH_FRAC}×median  "
          f"|  띠 경도±{LON_HALF}° 위도±{LAT_BAND}°  |  프레임 {FRAME_HHMM}")
    print(f"{'date':12} {'disk_median':>12} {'ch_area':>10}")
    print("-" * 36)

    # 날짜를 하나씩 돌면서 처리
    for date in dates:
        try:
            fits_path = download_one_day(date)            # 1) 다운로드
            ch_area, disk_median = compute_ch_area(fits_path)  # 2) 계산
            print(f"{date.date()!s:12} {disk_median:12.2f} {ch_area:10.4f}")
            results.append(dict(date=date.date(),
                                ch_area=round(ch_area, 5),
                                disk_median=round(disk_median, 2)))
        except Exception as e:
            # 어떤 날 파일이 없거나 오류가 나도 멈추지 않고 NaN으로 기록
            print(f"{date.date()!s:12}  실패: {e}")
            results.append(dict(date=date.date(), ch_area=np.nan, disk_median=np.nan))

    # 모은 결과를 표로 만들어 CSV로 저장
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    # 마무리 요약
    ok = df["ch_area"].notna().sum()
    print("-" * 36)
    print(f"완료: {ok}/{len(df)}일 성공  →  저장: {OUTPUT_CSV}")


# =============================================================================
#  함수 3) FITS 한 장에서 시각화에 필요한 것들(이미지·마스크)을 한꺼번에 돌려준다
#          - compute_ch_area()와 같은 계산이지만, 그림용으로 마스크까지 반환
# =============================================================================
def compute_masks(fits_path):
    """FITS 경로 -> (태양지도, 띠 마스크, CH 마스크, ch_area, disk_median) 반환."""
    smap = sunpy.map.Map(fits_path)
    pixel_coords = sunpy.map.all_coordinates_from_map(smap)
    helio = pixel_coords.transform_to(HeliographicStonyhurst(obstime=smap.date))
    lon = helio.lon.to_value(u.deg)
    lat = helio.lat.to_value(u.deg)

    on_disk = np.isfinite(lon) & np.isfinite(lat)
    in_slice = on_disk & (np.abs(lon) < LON_HALF) & (np.abs(lat) < LAT_BAND)
    disk_median = np.nanmedian(smap.data[on_disk])
    is_ch = (smap.data < CH_FRAC * disk_median) & in_slice
    ch_area = is_ch.sum() / in_slice.sum()
    return smap, in_slice, is_ch, ch_area, disk_median


# =============================================================================
#  함수 4) FITS 한 장을 날짜별 PNG 한 장으로 저장한다
# =============================================================================
def save_visualization(fits_path, timestamp, out):
    """기존 FITS 한 장을 마스크와 함께 PNG로 저장하고 성공 여부를 반환."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xticks([])
    ax.set_yticks([])
    success = False
    try:
        smap, in_slice, is_ch, ch_area, disk_median = compute_masks(fits_path)

        # 태양 이미지 (시각화 전용 정규화: 디스크 median -> 100 고정)
        #   -> 노출/열화로 밝기가 달라도 모든 날이 같은 밝기로 보인다 (계산엔 무관)
        data = smap.data / disk_median * 100.0
        data = np.clip(data, 1, None)
        norm = colors.LogNorm(vmin=10, vmax=1000)           # 모든 날짜 동일 스케일
        ax.imshow(data, origin="lower", cmap="sdoaia193", norm=norm)

        # 경위도 띠(파랑 반투명)
        blue = np.zeros((*in_slice.shape, 4)); blue[in_slice] = (0.2, 0.6, 1.0, 0.25)
        ax.imshow(blue, origin="lower")

        # CH 판정 픽셀(빨강)
        red = np.zeros((*is_ch.shape, 4)); red[is_ch] = (1.0, 0.1, 0.1, 0.9)
        ax.imshow(red, origin="lower")

        ax.set_title(
            f"{timestamp:%Y-%m-%d %H:%M}  A={ch_area:.5f}",
            fontsize=12,
        )
        success = True
    except Exception as e:
        ax.set_title(f"{timestamp:%Y-%m-%d %H:%M}  no data", fontsize=12)
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        print(f"[시각화] {timestamp:%Y-%m-%d %H:%M} 실패: {e}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[시각화] 저장: {out}")
    return success


# =============================================================================
#  --plot-only: 다운로드 폴더에 이미 있는 모든 FITS만 날짜별 PNG로 저장한다
# =============================================================================
def visualize_downloaded():
    """기존 FITS 전체를 그린다. 다운로드하거나 CSV를 읽고 쓰지 않는다."""
    pattern = os.path.join(DOWNLOAD_DIR, "AIA????????_????_0193.fits")
    fits_paths = sorted(glob.glob(pattern))
    filename_pattern = re.compile(r"AIA(\d{8})_(\d{4})_0193\.fits$")

    if not fits_paths:
        print(f"[시각화] 기존 FITS가 없습니다: {DOWNLOAD_DIR}")
        return

    ok = 0
    skipped = 0
    for fits_path in fits_paths:
        match = filename_pattern.fullmatch(os.path.basename(fits_path))
        if match is None:
            continue

        date_text, hhmm = match.groups()
        timestamp = pd.to_datetime(date_text + hhmm, format="%Y%m%d%H%M")
        out = os.path.join(
            PLOT_ONLY_DIR,
            f"ch_area_{timestamp:%Y-%m-%d}_{hhmm}.png",
        )

        if os.path.exists(out):
            print(f"[시각화] 건너뜀(이미 존재): {out}")
            skipped += 1
            continue

        ok += save_visualization(fits_path, timestamp, out)

    print(
        f"[시각화] 완료: 신규 {ok}개, 기존 {skipped}개 건너뜀, "
        f"전체 FITS {len(fits_paths)}개 → {PLOT_ONLY_DIR}"
    )


# =============================================================================
#  --cnn-export: 기존 FITS 전체를 CNN 입력용 PNG 또는 NPY로 저장한다
# =============================================================================
def export_cnn_inputs(export_format):
    """기존 FITS의 원본 크기 단일채널 배열을 PNG 또는 NPY로 저장한다."""
    pattern = os.path.join(DOWNLOAD_DIR, "AIA????????_????_0193.fits")
    fits_paths = sorted(glob.glob(pattern))
    out_dir = CNN_PNG_DIR if export_format == "png" else CNN_NPY_DIR
    os.makedirs(out_dir, exist_ok=True)

    if not fits_paths:
        print(f"[CNN] 기존 FITS가 없습니다: {DOWNLOAD_DIR}")
        return

    ok = 0
    for fits_path in fits_paths:
        stem = os.path.splitext(os.path.basename(fits_path))[0]
        out = os.path.join(out_dir, f"{stem}.{export_format}")
        try:
            # FITS의 2차원 픽셀 배열을 그대로 사용한다. 좌표축·제목·마스크·리사이즈 없음.
            data = np.asarray(sunpy.map.Map(fits_path).data, dtype=np.float32)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            if export_format == "npy":
                np.save(out, data, allow_pickle=False)
            else:
                # PNG는 단일채널 uint16이므로 음수는 0, 65535 초과값은 65535로 제한한다.
                png_data = np.rint(np.clip(data, 0, 65535)).astype(np.uint16)
                Image.fromarray(png_data).save(out)

            ok += 1
            print(f"[CNN] 저장: {out}  shape={data.shape}")
        except Exception as e:
            print(f"[CNN] 실패: {fits_path}  ({e})")

    print(f"[CNN] 완료: {ok}/{len(fits_paths)}개 성공 → {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="AIA 193Å 코로날 홀 면적 계산·시각화")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plot-only",
        action="store_true",
        help="기존에 다운로드된 모든 FITS의 PNG만 생성(CSV·다운로드 미사용)",
    )
    mode.add_argument(
        "--cnn-export",
        choices=("png", "npy"),
        help="기존 FITS 전체를 CNN 입력용 PNG 또는 NPY로 저장",
    )
    return parser.parse_args()


# 이 파일을 직접 실행했을 때만 돌린다 (import 시에는 안 돌아감)
if __name__ == "__main__":
    args = parse_args()
    if args.plot_only:
        visualize_downloaded()
    elif args.cnn_export:
        export_cnn_inputs(args.cnn_export)
    else:
        main()                 # 다운로드 + 계산 + CSV 저장
