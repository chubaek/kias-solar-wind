"""Limited 2022-2025 HMI magnetogram feasibility diagnostic for 72h speed.

This script intentionally stays scoped to daily HMI full-disk line-of-sight
magnetograms from 2022-01-01 through 2025-12-31. It uses monthly JSOC exports,
extracts approximate geometric disk-center window features, then compares the
current representative CH feature set against the same set plus HMI window
features for a limited public/private diagnostic.
"""

from __future__ import annotations

import argparse
import calendar
import importlib.util
import math
import os
import re
import shutil
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

import run_ch_feature_addition_72h as chrun
import train_tabular_models_72h as tab


HERE = Path(__file__).resolve().parent
MAG_DIR = HERE / "data" / "magnetograms"
RAW_DIR = MAG_DIR / "hmi_raw"
DOWNLOAD_REPORT_CSV = MAG_DIR / "stage1_2022_2025_download_report.csv"
FEATURE_CSV = MAG_DIR / "magnetogram_window_features_daily_2022_2025.csv"
OUT_DIR = HERE / "outputs" / "magnetogram_ch_features_72h"

HMI_SERIES = "hmi.M_720s"
HMI_SEGMENT = "magnetogram"
INSTALL_HINT = "uv add astropy drms sunpy"
USER_AGENT = "kias-solar-wind-hmi-stage1/1.0"

START_DATE = "2022-01-01"
END_DATE = "2025-12-31"
DAILY_TARGET_HHMM = "0000"

WINDOWS = {
    "W_lon7p5_lat15": (7.5, 15.0),
    "W_lon30_lat15": (30.0, 15.0),
    "W_lon30_lat30": (30.0, 30.0),
    "W_lon60_lat60": (60.0, 60.0),
}
MAG_METRICS = [
    "mean_abs_B",
    "median_abs_B",
    "signed_mean_B",
    "sum_pos_B",
    "sum_neg_B",
    "sum_abs_B",
    "polarity_imbalance",
    "dominant_polarity",
]
MAG_OFFSETS = [
    (3, 0),
    (4, -1),
    (5, -2),
]
ENSEMBLE_MODEL = "ensemble_0p7_mlp_0p3_extratrees"
ENSEMBLE_WEIGHTS = {"direct_mlp": 0.70, "extratrees": 0.30}


@dataclass(frozen=True)
class RecordInfo:
    month: str
    requested_time: pd.Timestamp
    matched_time: pd.Timestamp
    matched_time_le_requested: bool


def require_environment() -> str:
    missing = [name for name in ("astropy", "drms") if importlib.util.find_spec(name) is None]
    if missing:
        print(f"missing_required_dependencies={','.join(missing)}", flush=True)
        print(INSTALL_HINT, flush=True)
        raise SystemExit(2)
    email = os.environ.get("JSOC_EMAIL", "").strip()
    if not email:
        print('export JSOC_EMAIL="[your_email@example.com](mailto:your_email@example.com)"', flush=True)
        raise SystemExit(2)
    return email


def month_starts(start: str, end: str) -> list[pd.Timestamp]:
    starts = pd.date_range(pd.Timestamp(start).replace(day=1), pd.Timestamp(end), freq="MS")
    return [pd.Timestamp(ts) for ts in starts]


def month_days(month_start: pd.Timestamp, end: str) -> list[pd.Timestamp]:
    _, n_days = calendar.monthrange(month_start.year, month_start.month)
    month_end = month_start + pd.Timedelta(days=n_days - 1)
    last = min(month_end, pd.Timestamp(end))
    return list(pd.date_range(month_start, last, freq="D"))


def jsoc_time_token(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y.%m.%d_%H:%M:%S_TAI")


def month_query(month_start: pd.Timestamp, n_days: int) -> str:
    return f"{HMI_SERIES}[{jsoc_time_token(month_start)}/{n_days}d@1d]"


def export_query_for_records(records: list[RecordInfo]) -> str:
    if not records:
        raise ValueError("Cannot build an export query for zero records.")
    first = records[0].matched_time
    return f"{HMI_SERIES}[{jsoc_time_token(first)}/{len(records)}d@1d]{{{HMI_SEGMENT}}}"


def single_export_query(record: RecordInfo) -> str:
    return f"{HMI_SERIES}[{jsoc_time_token(record.matched_time)}]{{{HMI_SEGMENT}}}"


def parse_jsoc_time(value: Any) -> pd.Timestamp | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    for suffix in ("_TAI", "_UTC"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    for fmt in ("%Y.%m.%d_%H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y.%m.%d_%H:%M"):
        try:
            return pd.to_datetime(text, format=fmt)
        except ValueError:
            pass
    try:
        return pd.to_datetime(text)
    except Exception:
        return None


def hmi_glob_for_time(ts: pd.Timestamp) -> str:
    return f"hmi.m_720s.{ts.strftime('%Y%m%d_%H%M%S')}_TAI*.magnetogram.fits"


def existing_path_for_time(ts: pd.Timestamp) -> Path | None:
    matches = [path for path in sorted(RAW_DIR.glob(hmi_glob_for_time(ts))) if is_probable_fits(path)]
    return matches[0] if matches else None


def local_hmi_path(download_url: str, matched_time: pd.Timestamp) -> Path:
    name = Path(urllib.parse.urlparse(download_url).path).name
    if not name or "." not in name:
        name = f"hmi.m_720s.{matched_time.strftime('%Y%m%d_%H%M%S')}_TAI.magnetogram.fits"
    return RAW_DIR / name


def url_time(url: str) -> pd.Timestamp | None:
    name = Path(urllib.parse.urlparse(url).path).name
    match = re.search(r"(\d{8})_(\d{6})_TAI", name)
    if not match:
        return None
    return pd.to_datetime("".join(match.groups()), format="%Y%m%d%H%M%S")


def dataframe_urls(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        for col in ("url", "URL"):
            if col in value.columns:
                return [str(item) for item in value[col].dropna().tolist()]
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]
    return []


def export_status_text(export_request: Any) -> str:
    parts: list[str] = []
    for attr in ("status", "requestid", "protocol", "method", "dirurl"):
        if hasattr(export_request, attr):
            try:
                parts.append(f"{attr}={getattr(export_request, attr)}")
            except Exception as exc:
                parts.append(f"{attr}=<error {type(exc).__name__}: {exc}>")
    try:
        urls = dataframe_urls(export_request.urls)
    except Exception as exc:
        urls = []
        parts.append(f"urls_error={type(exc).__name__}: {exc}")
    if urls:
        parts.append(f"urls={';'.join(urls)}")
    return " | ".join(parts)


def download_url(url: str, path: Path, timeout: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    if not is_probable_fits(tmp):
        preview = tmp.read_text(errors="replace")[:300]
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded response is not a FITS file: {preview}")
    tmp.replace(path)


def query_month_records(client: Any, month_start: pd.Timestamp, end: str) -> list[RecordInfo]:
    days = month_days(month_start, end)
    query = month_query(month_start, len(days))
    records = client.query(query, key="T_REC,QUALITY")
    found: dict[str, pd.Timestamp] = {}
    if records is not None and len(records) > 0 and "T_REC" in records.columns:
        for value in records["T_REC"].tolist():
            ts = parse_jsoc_time(value)
            if ts is not None:
                found[ts.strftime("%Y-%m-%d")] = ts

    infos: list[RecordInfo] = []
    for day in days:
        requested = day.replace(hour=0, minute=0, second=0, microsecond=0)
        matched = found.get(day.strftime("%Y-%m-%d"))
        if matched is None:
            matched = requested
        infos.append(
            RecordInfo(
                month=month_start.strftime("%Y-%m"),
                requested_time=requested,
                matched_time=matched,
                matched_time_le_requested=bool(matched <= requested),
            )
        )
    return infos


def report_row(
    record: RecordInfo,
    *,
    requested_records: int,
    downloaded_records: int,
    skipped_existing_files: int,
    failed_records: int,
    export_method: str,
    status: str,
    jsoc_query: str,
    url: str = "",
    export_status: str = "",
    local_path: str = "",
    error_message: str = "",
) -> dict[str, Any]:
    return {
        "month": record.month,
        "requested_records": requested_records,
        "downloaded_records": downloaded_records,
        "skipped_existing_files": skipped_existing_files,
        "failed_records": failed_records,
        "export_method": export_method,
        "requested_time": record.requested_time.strftime("%Y-%m-%d %H:%M:%S"),
        "matched_hmi_time": record.matched_time.strftime("%Y-%m-%d %H:%M:%S"),
        "matched_time_le_requested_time": record.matched_time_le_requested,
        "status": status,
        "jsoc_query": jsoc_query,
        "url": url,
        "export_status": export_status,
        "local_path": local_path,
        "error_message": error_message,
    }


def download_record_urls(
    records: list[RecordInfo],
    urls: list[str],
    timeout: int,
) -> tuple[list[dict[str, Any]], int, int, int]:
    by_time = {record.matched_time.strftime("%Y%m%d_%H%M%S"): record for record in records}
    downloaded = 0
    skipped = 0
    failed = 0
    rows: list[dict[str, Any]] = []

    for url in urls:
        matched = url_time(url)
        record = by_time.get(matched.strftime("%Y%m%d_%H%M%S")) if matched is not None else None
        if record is None and records:
            record = records[min(len(rows), len(records) - 1)]
            matched = record.matched_time
        if record is None or matched is None:
            continue
        existing = existing_path_for_time(matched)
        path = existing or local_hmi_path(url, matched)
        if existing is not None:
            skipped += 1
            rows.append(
                report_row(
                    record,
                    requested_records=len(records),
                    downloaded_records=0,
                    skipped_existing_files=1,
                    failed_records=0,
                    export_method="monthly_batch",
                    status="skipped_existing",
                    jsoc_query="",
                    url=url,
                    local_path=str(existing),
                )
            )
            continue
        try:
            download_url(url, path, timeout)
            downloaded += 1
            rows.append(
                report_row(
                    record,
                    requested_records=len(records),
                    downloaded_records=1,
                    skipped_existing_files=0,
                    failed_records=0,
                    export_method="monthly_batch",
                    status="downloaded",
                    jsoc_query="",
                    url=url,
                    local_path=str(path),
                )
            )
        except Exception as exc:
            failed += 1
            rows.append(
                report_row(
                    record,
                    requested_records=len(records),
                    downloaded_records=0,
                    skipped_existing_files=0,
                    failed_records=1,
                    export_method="monthly_batch",
                    status="download_failed",
                    jsoc_query="",
                    url=url,
                    local_path=str(path),
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            )
    return rows, downloaded, skipped, failed


def export_records(client: Any, query: str, timeout: int, export_timeout: int) -> tuple[list[str], str]:
    export_request = client.export(query, method="url", protocol="fits")
    export_request.wait(timeout=export_timeout)
    status = export_status_text(export_request)
    urls = dataframe_urls(export_request.urls)
    if not urls:
        raise RuntimeError(f"JSOC export returned no URLs. {status}")
    return urls, status


def daily_fallback(
    client: Any,
    records: list[RecordInfo],
    timeout: int,
    export_timeout: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        existing = existing_path_for_time(record.matched_time)
        query = single_export_query(record)
        if existing is not None:
            rows.append(
                report_row(
                    record,
                    requested_records=1,
                    downloaded_records=0,
                    skipped_existing_files=1,
                    failed_records=0,
                    export_method="daily_fallback",
                    status="skipped_existing",
                    jsoc_query=query,
                    local_path=str(existing),
                )
            )
            continue
        try:
            urls, export_status = export_records(client, query, timeout, export_timeout)
            url = urls[0]
            path = local_hmi_path(url, record.matched_time)
            download_url(url, path, timeout)
            rows.append(
                report_row(
                    record,
                    requested_records=1,
                    downloaded_records=1,
                    skipped_existing_files=0,
                    failed_records=0,
                    export_method="daily_fallback",
                    status="downloaded",
                    jsoc_query=query,
                    url=url,
                    export_status=export_status,
                    local_path=str(path),
                )
            )
        except Exception as exc:
            rows.append(
                report_row(
                    record,
                    requested_records=1,
                    downloaded_records=0,
                    skipped_existing_files=0,
                    failed_records=1,
                    export_method="daily_fallback",
                    status="failed",
                    jsoc_query=query,
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return rows


def download_hmi_monthly(args: argparse.Namespace) -> pd.DataFrame:
    email = require_environment()
    import drms

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = drms.Client(email=email)
    rows: list[dict[str, Any]] = []
    for month_start in month_starts(args.start, args.end):
        month = month_start.strftime("%Y-%m")
        records = query_month_records(client, month_start, args.end)
        missing = [record for record in records if existing_path_for_time(record.matched_time) is None]
        skipped = len(records) - len(missing)
        print(f"month={month} requested={len(records)} existing={skipped} missing={len(missing)}", flush=True)

        if not missing:
            for record in records:
                rows.append(
                    report_row(
                        record,
                        requested_records=len(records),
                        downloaded_records=0,
                        skipped_existing_files=1,
                        failed_records=0,
                        export_method="monthly_batch",
                        status="skipped_existing",
                        jsoc_query=single_export_query(record),
                        local_path=str(existing_path_for_time(record.matched_time) or ""),
                    )
                )
            continue

        query = export_query_for_records(records)
        try:
            urls, export_status = export_records(client, query, args.timeout, args.export_timeout)
            month_rows, downloaded, skipped_from_urls, failed = download_record_urls(records, urls, args.timeout)
            for row in month_rows:
                row["jsoc_query"] = query
                row["export_status"] = row["export_status"] or export_status
            covered_keys = {
                pd.Timestamp(row["matched_hmi_time"]).strftime("%Y%m%d_%H%M%S")
                for row in month_rows
                if row["status"] in {"downloaded", "skipped_existing"}
            }
            for record in records:
                key = record.matched_time.strftime("%Y%m%d_%H%M%S")
                if key not in covered_keys:
                    failed += 1
                    month_rows.append(
                        report_row(
                            record,
                            requested_records=len(records),
                            downloaded_records=0,
                            skipped_existing_files=0,
                            failed_records=1,
                            export_method="monthly_batch",
                            status="missing_from_export_urls",
                            jsoc_query=query,
                            export_status=export_status,
                            error_message="No URL returned for this requested record.",
                        )
                    )
            rows.extend(month_rows)
            print(
                f"month={month} monthly_batch downloaded={downloaded} skipped={skipped + skipped_from_urls} failed={failed}",
                flush=True,
            )
        except Exception as exc:
            print(f"month={month} monthly_batch_failed={type(exc).__name__}: {exc}", flush=True)
            rows.extend(
                daily_fallback(client, records, args.timeout, args.export_timeout, args.sleep_seconds)
            )

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    report = pd.DataFrame(rows)
    DOWNLOAD_REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(DOWNLOAD_REPORT_CSV, index=False)
    return report


def raw_files() -> list[Path]:
    return [path for path in sorted(RAW_DIR.glob("hmi.m_720s.*_TAI*.magnetogram.fits")) if is_probable_fits(path)]


def is_probable_fits(path: Path) -> bool:
    try:
        if path.stat().st_size < 1_000_000:
            return False
        with path.open("rb") as handle:
            return handle.read(6) == b"SIMPLE"
    except OSError:
        return False


def fits_time(path: Path) -> pd.Timestamp | None:
    match = re.search(r"(\d{8})_(\d{6})_TAI", path.name)
    if match:
        return pd.to_datetime("".join(match.groups()), format="%Y%m%d%H%M%S")
    return None


def open_fits_array(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        hdu = next((item for item in hdul if getattr(item, "data", None) is not None), None)
        if hdu is None:
            raise RuntimeError("No image HDU found.")
        data = np.asarray(hdu.data, dtype=np.float64)
        header = dict(hdu.header)
    if data.ndim > 2:
        data = np.squeeze(data)
    if data.ndim != 2:
        raise RuntimeError(f"Expected 2D FITS image, got shape {data.shape}.")
    return data, header


def approximate_hmi_disk_geometry(
    shape: tuple[int, int],
    header: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = shape
    y, x = np.indices((ny, nx), dtype=np.float64)
    cx = float(header.get("CRPIX1", (nx + 1) / 2.0)) - 1.0
    cy = float(header.get("CRPIX2", (ny + 1) / 2.0)) - 1.0
    cdelt1 = abs(float(header.get("CDELT1", 1.0)))
    cdelt2 = abs(float(header.get("CDELT2", cdelt1)))
    if "RSUN_OBS" in header:
        rsun_x = abs(float(header["RSUN_OBS"]) / cdelt1)
        rsun_y = abs(float(header["RSUN_OBS"]) / cdelt2)
    elif "R_SUN" in header:
        rsun_x = rsun_y = float(header["R_SUN"])
    else:
        rsun_x = rsun_y = min(nx, ny) / 2.0
    x_norm = (x - cx) / rsun_x
    y_norm = (y - cy) / rsun_y
    rho2 = x_norm**2 + y_norm**2
    mu = np.sqrt(np.clip(1.0 - rho2, 0.0, 1.0))
    lon = np.degrees(np.arcsin(np.clip(x_norm, -1.0, 1.0)))
    lat = np.degrees(np.arcsin(np.clip(y_norm, -1.0, 1.0)))
    valid_disk = rho2 <= 1.0
    return lon, lat, mu, valid_disk


def summarize_field(values: np.ndarray) -> dict[str, float]:
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return {
            "n_pix": 0,
            "mean_abs_B": np.nan,
            "median_abs_B": np.nan,
            "signed_mean_B": np.nan,
            "sum_pos_B": np.nan,
            "sum_neg_B": np.nan,
            "sum_abs_B": np.nan,
            "polarity_imbalance": np.nan,
            "dominant_polarity": np.nan,
        }
    sum_b = float(np.sum(valid))
    sum_abs = float(np.sum(np.abs(valid)))
    return {
        "n_pix": int(len(valid)),
        "mean_abs_B": float(np.mean(np.abs(valid))),
        "median_abs_B": float(np.median(np.abs(valid))),
        "signed_mean_B": float(np.mean(valid)),
        "sum_pos_B": float(np.sum(valid[valid > 0])),
        "sum_neg_B": float(np.sum(valid[valid < 0])),
        "sum_abs_B": sum_abs,
        "polarity_imbalance": float(abs(sum_b) / sum_abs) if sum_abs > 0 else np.nan,
        "dominant_polarity": float(np.sign(sum_b)) if sum_abs > 0 else np.nan,
    }


def extract_features_for_file(path: Path) -> dict[str, Any]:
    mag_time = fits_time(path)
    if mag_time is None:
        raise RuntimeError(f"Cannot parse HMI time from {path.name}")
    data, header = open_fits_array(path)
    lon, lat, mu, valid_disk = approximate_hmi_disk_geometry(data.shape, header)
    good_mu = valid_disk & (mu >= 0.2)
    radial_field = np.full_like(data, np.nan, dtype=np.float64)
    radial_field[good_mu] = data[good_mu] / mu[good_mu]

    row: dict[str, Any] = {
        "magnetogram_time": mag_time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "HMI",
        "product": HMI_SERIES,
        "local_path": str(path),
        "feature_label": "approximate_window_magnetogram_features",
        "coordinate_mode": "approximate_geometric_disk_center",
        "field_mode": "Br_approx_equals_B_los_over_mu_mu_ge_0p2",
    }
    for window, (lon_half, lat_half) in WINDOWS.items():
        mask = good_mu & (np.abs(lon) <= lon_half) & (np.abs(lat) <= lat_half)
        for key, value in summarize_field(radial_field[mask]).items():
            row[f"{window}_{key}"] = value
    return row


def extract_features(force: bool = False) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = raw_files()
    if FEATURE_CSV.exists() and not force:
        existing = pd.read_csv(FEATURE_CSV)
        done = set(existing.get("local_path", pd.Series(dtype=str)).astype(str))
        rows = existing.to_dict("records")
    else:
        done = set()
        rows = []
    for idx, path in enumerate(files, start=1):
        if str(path) in done:
            continue
        try:
            rows.append(extract_features_for_file(path))
        except Exception as exc:
            mag_time = fits_time(path)
            rows.append(
                {
                    "magnetogram_time": mag_time.strftime("%Y-%m-%d %H:%M:%S") if mag_time is not None else "",
                    "source": "HMI",
                    "product": HMI_SERIES,
                    "local_path": str(path),
                    "feature_label": "approximate_window_magnetogram_features",
                    "coordinate_mode": "approximate_geometric_disk_center",
                    "field_mode": "Br_approx_equals_B_los_over_mu_mu_ge_0p2",
                    "extraction_error": f"{type(exc).__name__}: {exc}",
                }
            )
        if idx % 50 == 0:
            print(f"feature_extraction_processed_files={idx}/{len(files)}", flush=True)
    features = pd.DataFrame(rows).sort_values("magnetogram_time").reset_index(drop=True)
    FEATURE_CSV.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURE_CSV, index=False)
    return features


def raw_disk_usage_bytes() -> int:
    return int(sum(path.stat().st_size for path in raw_files()))


def build_download_availability_report(download_report: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    days = pd.date_range(START_DATE, END_DATE, freq="D")
    valid_features = features
    if "extraction_error" in valid_features.columns:
        valid_features = valid_features[valid_features["extraction_error"].isna()]
    downloaded_times = set(pd.to_datetime(valid_features["magnetogram_time"]).dt.strftime("%Y-%m-%d")) if not valid_features.empty else set()
    requested_day_set = set(days.strftime("%Y-%m-%d"))
    failed_days = requested_day_set - downloaded_times

    rows: list[dict[str, Any]] = [
        {
            "report_type": "overall",
            "item": "requested_days",
            "value": len(days),
            "note": "",
        },
        {
            "report_type": "overall",
            "item": "downloaded_days",
            "value": len(downloaded_times),
            "note": "",
        },
        {
            "report_type": "overall",
            "item": "failed_days",
            "value": len(failed_days),
            "note": "",
        },
        {
            "report_type": "overall",
            "item": "total_raw_fits_disk_usage_bytes",
            "value": raw_disk_usage_bytes(),
            "note": "",
        },
    ]

    by_year = pd.DataFrame({"day": days})
    by_year["year"] = by_year["day"].dt.year
    by_year["downloaded"] = by_year["day"].dt.strftime("%Y-%m-%d").isin(downloaded_times)
    for year, group in by_year.groupby("year"):
        rows.append(
            {
                "report_type": "missing_rate_by_year",
                "item": int(year),
                "value": float(1.0 - group["downloaded"].mean()),
                "note": f"downloaded_days={int(group['downloaded'].sum())} requested_days={len(group)}",
            }
        )

    for window in WINDOWS:
        cols = [f"{window}_{metric}" for metric in MAG_METRICS if f"{window}_{metric}" in features.columns]
        if cols and not valid_features.empty:
            nan_rate = float(valid_features[cols].isna().mean().mean())
        else:
            nan_rate = 1.0
        rows.append(
            {
                "report_type": "feature_nan_rate_by_window",
                "item": window,
                "value": nan_rate,
                "note": "approximate_window_magnetogram_features",
            }
        )
    return pd.DataFrame(rows)


def load_stage1_features() -> pd.DataFrame:
    mag = pd.read_csv(FEATURE_CSV)
    if "extraction_error" in mag.columns:
        mag = mag[mag["extraction_error"].isna()].copy()
    mag["mag_time"] = pd.to_datetime(mag["magnetogram_time"], utc=True).dt.tz_convert(None)
    rename = {}
    for window in WINDOWS:
        for metric in MAG_METRICS:
            col = f"{window}_{metric}"
            if col in mag.columns:
                rename[col] = f"mag_{window}_{metric}"
    return mag.rename(columns=rename).sort_values("mag_time").reset_index(drop=True)


def mag_feature_columns(mag: pd.DataFrame) -> list[str]:
    cols = []
    for window in WINDOWS:
        for metric in MAG_METRICS:
            col = f"mag_{window}_{metric}"
            if col in mag.columns:
                cols.append(col)
    return cols


def build_baseline_all() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    ch = chrun.load_ch()
    table_all_base = tab.build_feature_table(
        tab.FULL_CSV,
        min_target_year=2011,
        max_target_year=2025,
        require_finite_target=False,
        require_finite_persistence=False,
    )
    tables_all_ch, features_ch, sanity_ch = chrun.feature_sets(table_all_base, ch)
    name = "current_plus_representative_mrmr_ch"
    return tables_all_ch[name], features_ch[name], sanity_ch


def add_magnetogram_features(
    table: pd.DataFrame,
    mag: pd.DataFrame,
    tolerance_hours: int,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    out = table.copy()
    base = out[["origin_datetime", "target_datetime"]].copy()
    base["origin_datetime"] = pd.to_datetime(base["origin_datetime"])
    base["target_datetime"] = pd.to_datetime(base["target_datetime"])
    base["_row_id"] = np.arange(len(base))
    source_cols = mag_feature_columns(mag)
    new_cols: list[str] = []
    sanity_parts: list[pd.DataFrame] = []
    added_columns: dict[str, np.ndarray] = {}

    for target_lag, origin_offset in MAG_OFFSETS:
        request = base.copy()
        request["requested_magnetogram_time"] = request["origin_datetime"] + pd.to_timedelta(origin_offset, unit="D")
        request = request.sort_values("requested_magnetogram_time")
        merged = pd.merge_asof(
            request,
            mag[["mag_time", *source_cols]].sort_values("mag_time"),
            left_on="requested_magnetogram_time",
            right_on="mag_time",
            direction="backward",
            tolerance=pd.Timedelta(hours=tolerance_hours),
        ).sort_values("_row_id")

        for feature in source_cols:
            out_col = f"{feature}__target_lag_{target_lag}d__origin_offset_{origin_offset}d"
            values = merged[feature].to_numpy()
            added_columns[out_col] = values
            missing_col = f"{out_col}__missing"
            added_columns[missing_col] = pd.isna(values).astype(np.float32)
            new_cols.extend([out_col, missing_col])

        sanity = merged[["_row_id", "origin_datetime", "target_datetime", "requested_magnetogram_time", "mag_time"]].copy()
        sanity["feature_set_name"] = "magnetogram_window_features"
        sanity["target_lag_days"] = target_lag
        sanity["origin_offset_days"] = origin_offset
        sanity["matched_magnetogram_time_le_requested"] = sanity["mag_time"].isna() | (
            sanity["mag_time"] <= sanity["requested_magnetogram_time"]
        )
        sanity["requested_time_le_origin"] = sanity["requested_magnetogram_time"] <= sanity["origin_datetime"]
        sanity["target_is_origin_plus_72h"] = (sanity["target_datetime"] - sanity["origin_datetime"]) == pd.Timedelta(hours=72)
        sanity_parts.append(sanity)
    if added_columns:
        out = pd.concat([out, pd.DataFrame(added_columns, index=out.index)], axis=1)
    return out, new_cols, pd.concat(sanity_parts, ignore_index=True)


def finite_target_idx(table: pd.DataFrame, years: list[int]) -> np.ndarray:
    mask = table["target_year"].isin(years) & table["target_speed"].notna() & table["persistence_27day_target_aligned"].notna()
    return np.flatnonzero(mask.to_numpy())


def model_configs() -> dict[str, Any]:
    configs = tab.candidate_models("initial")
    mlp = next(c for c in configs if c["name"].startswith("mlp_hidden128x64"))
    extra = next(c for c in configs if c["name"] == "extratrees_n300_depth12_min5_feat0.8")
    return {"direct_mlp": mlp["estimator"], "extratrees": extra["estimator"]}


def fit_predict(estimator: Any, X_train: pd.DataFrame, y_train: np.ndarray, X_eval: pd.DataFrame) -> np.ndarray:
    model = clone(estimator)
    model.fit(X_train, y_train)
    return model.predict(X_eval)


def evaluate_indices(
    table: pd.DataFrame,
    features: list[str],
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    feature_set_name: str,
    scheme: str,
    fold: str,
    models: dict[str, Any],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    X_train = table.iloc[train_idx][features]
    X_eval = table.iloc[eval_idx][features]
    y_train = table.iloc[train_idx]["target_speed"].to_numpy(dtype=np.float32)
    y_eval = table.iloc[eval_idx]["target_speed"].to_numpy(dtype=np.float32)
    persistence = table.iloc[eval_idx]["persistence_27day_target_aligned"].to_numpy(dtype=np.float32)
    preds = {
        "direct_mlp": fit_predict(models["direct_mlp"], X_train, y_train, X_eval),
        "extratrees": fit_predict(models["extratrees"], X_train, y_train, X_eval),
    }
    preds[ENSEMBLE_MODEL] = (
        ENSEMBLE_WEIGHTS["direct_mlp"] * preds["direct_mlp"]
        + ENSEMBLE_WEIGHTS["extratrees"] * preds["extratrees"]
    )
    rows = []
    for model_name, pred in preds.items():
        rows.append(
            {
                "feature_set_name": feature_set_name,
                "model_name": model_name,
                "validation_scheme": scheme,
                "fold": fold,
                **tab.metrics(y_eval, pred, persistence),
                "n_prediction_rows": int(len(eval_idx)),
                "n_scored_finite_target_rows": int(np.isfinite(y_eval).sum()),
            }
        )
    pred_frame = table.iloc[eval_idx][["origin_datetime", "target_datetime", "target_speed"]].copy()
    pred_frame["feature_set_name"] = feature_set_name
    pred_frame["predicted_speed"] = preds[ENSEMBLE_MODEL]
    return rows, pred_frame


def stage1_timestamp_sanity(mag_sanity: pd.DataFrame) -> pd.DataFrame:
    candidate = mag_sanity.copy()
    origin_year = pd.to_datetime(candidate["origin_datetime"]).dt.year
    candidate = candidate[(origin_year >= 2022) & (origin_year <= 2025)]
    matched = candidate[candidate["mag_time"].notna()]
    if len(matched) >= 10:
        candidate = matched
    if candidate.empty:
        candidate = mag_sanity
    sample = candidate.sample(n=min(10, len(candidate)), random_state=20260624).copy()
    sample = sample.rename(
        columns={
            "origin_datetime": "origin_time",
            "target_datetime": "target_time",
            "mag_time": "matched_magnetogram_time",
        }
    )
    cols = [
        "origin_time",
        "target_time",
        "requested_magnetogram_time",
        "matched_magnetogram_time",
        "matched_magnetogram_time_le_requested",
        "requested_time_le_origin",
        "target_is_origin_plus_72h",
        "target_lag_days",
        "origin_offset_days",
    ]
    return sample[cols].sort_values("origin_time")


def evaluate_stage1(features_df: pd.DataFrame, availability: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mag = load_stage1_features()
    baseline_table, baseline_features, _ = build_baseline_all()
    mag_table, mag_cols, mag_sanity = add_magnetogram_features(baseline_table, mag, tolerance_hours=36)
    models = model_configs()

    tables = {
        "baseline_current_best": (baseline_table, baseline_features),
        "magnetogram_window_features": (mag_table, baseline_features + mag_cols),
    }

    fixed_train = finite_target_idx(baseline_table, [2022])
    fixed_eval = finite_target_idx(baseline_table, [2023])
    private_train = finite_target_idx(baseline_table, [2022, 2023])
    private_eval = finite_target_idx(baseline_table, [2024, 2025])

    fixed_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    private_predictions: list[pd.DataFrame] = []
    for name, (table, cols) in tables.items():
        rows, _ = evaluate_indices(
            table,
            cols,
            fixed_train,
            fixed_eval,
            name,
            "fixed_validation_2022_train_2023_validate",
            "fixed_2022_2023",
            models,
        )
        fixed_rows.extend(rows)
        rows, pred = evaluate_indices(
            table,
            cols,
            private_train,
            private_eval,
            name,
            "private_diagnostic_2024_2025",
            "private_2024_2025",
            models,
        )
        private_rows.extend(rows)
        private_predictions.append(pred)

    fixed_df = pd.DataFrame(fixed_rows)
    private_df = pd.DataFrame(private_rows)
    summary = fixed_df[fixed_df["model_name"] == ENSEMBLE_MODEL][
        ["feature_set_name", "mae", "rmse", "cc", "n_prediction_rows", "n_scored_finite_target_rows"]
    ].rename(
        columns={
            "feature_set_name": "feature_set",
            "mae": "fixed_MAE",
            "rmse": "fixed_RMSE",
            "cc": "fixed_CC",
            "n_prediction_rows": "fixed_prediction_rows",
            "n_scored_finite_target_rows": "fixed_scored_finite_target_rows",
        }
    )
    private_summary = private_df[private_df["model_name"] == ENSEMBLE_MODEL][
        ["feature_set_name", "mae", "rmse", "cc", "n_prediction_rows", "n_scored_finite_target_rows"]
    ].rename(
        columns={
            "feature_set_name": "feature_set",
            "mae": "private_MAE",
            "rmse": "private_RMSE",
            "cc": "private_CC",
            "n_prediction_rows": "number_of_private_prediction_rows",
            "n_scored_finite_target_rows": "number_of_scored_finite_target_rows",
        }
    )
    summary = summary.merge(private_summary, on="feature_set", how="left")

    baseline = summary[summary["feature_set"] == "baseline_current_best"].iloc[0]
    mag_row = summary[summary["feature_set"] == "magnetogram_window_features"].iloc[0]
    fixed_improves = bool(
        (mag_row["fixed_CC"] > baseline["fixed_CC"] + 0.01)
        and (mag_row["fixed_MAE"] < baseline["fixed_MAE"])
    )
    summary["selection_rule"] = np.where(
        summary["feature_set"] == "magnetogram_window_features",
        "adopt_if_fixed_CC_improves_by_gt_0p01_and_fixed_MAE_decreases",
        "baseline_current_best",
    )
    summary["adopt_magnetogram_features"] = summary["feature_set"].eq("magnetogram_window_features") & fixed_improves
    summary["next_step"] = np.where(
        summary["adopt_magnetogram_features"],
        "prepare_full_2011_2025_HMI_daily_feature_generation_plan",
        "do_not_adopt_approximate_window_magnetogram_features",
    )

    fixed_df.to_csv(OUT_DIR / "stage1_fixed_results.csv", index=False)
    private_df.to_csv(OUT_DIR / "stage1_private_diagnostic.csv", index=False)
    summary.to_csv(OUT_DIR / "stage1_summary.csv", index=False)
    timestamp = stage1_timestamp_sanity(mag_sanity)
    timestamp.to_csv(OUT_DIR / "stage1_timestamp_sanity_check.csv", index=False)
    availability.to_csv(OUT_DIR / "stage1_data_availability_report.csv", index=False)

    best_name = "magnetogram_window_features" if fixed_improves else "baseline_current_best"
    pred = pd.concat(private_predictions, ignore_index=True)
    pred = pred[pred["feature_set_name"] == best_name][["target_datetime", "predicted_speed"]].rename(
        columns={"target_datetime": "datetime"}
    )
    pred.to_csv(OUT_DIR / "stage1_best_private_prediction.csv", index=False)

    print("\nData availability report", flush=True)
    print(availability.to_string(index=False), flush=True)
    print("\nTimestamp sanity check", flush=True)
    print(timestamp.to_string(index=False), flush=True)
    print("\nConcise comparison", flush=True)
    cols = [
        "feature_set",
        "fixed_MAE",
        "fixed_RMSE",
        "fixed_CC",
        "private_MAE",
        "private_RMSE",
        "private_CC",
        "number_of_private_prediction_rows",
        "number_of_scored_finite_target_rows",
    ]
    print(summary[cols].to_string(index=False), flush=True)
    print(f"\nAdopt magnetogram features: {fixed_improves}", flush=True)
    if fixed_improves:
        print("Next step: prepare full 2011-2025 HMI daily feature generation plan.", flush=True)
    else:
        print("Do not adopt these approximate magnetogram window features.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--export-timeout", type=int, default=900)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-feature-extract", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    require_environment()
    MAG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.skip_download and DOWNLOAD_REPORT_CSV.exists():
        download_report = pd.read_csv(DOWNLOAD_REPORT_CSV)
    elif args.skip_download:
        download_report = pd.DataFrame()
    else:
        download_report = download_hmi_monthly(args)

    features = extract_features(force=args.force_feature_extract)
    availability = build_download_availability_report(download_report, features)
    availability.to_csv(OUT_DIR / "stage1_data_availability_report.csv", index=False)
    print("\nDownload/feature availability", flush=True)
    print(availability.to_string(index=False), flush=True)

    if not args.skip_eval:
        evaluate_stage1(features, availability)


if __name__ == "__main__":
    main()
