"""Full 2011-2025 HMI daily magnetogram feature/evaluation pipeline.

The script is action-gated: pass --download, --extract, and/or --evaluate to do
work. Pass --dry-run to print the planned monthly/daily JSOC queries without
downloading, extracting, or fitting models.
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
DOWNLOAD_REPORT_CSV = MAG_DIR / "stage2_2011_2025_download_report.csv"
DOWNLOAD_EVENTS_CSV = MAG_DIR / "stage2_2011_2025_download_events.csv"
FEATURE_CSV = MAG_DIR / "magnetogram_window_features_daily_2011_2025.csv"
RAW_VALIDATION_CSV = MAG_DIR / "stage2_raw_validation_report.csv"
OUT_DIR = HERE / "outputs" / "magnetogram_ch_features_72h"

HMI_SERIES = "hmi.M_720s"
HMI_SEGMENT = "magnetogram"
INSTALL_HINT = ".venv/bin/python -m pip install astropy drms sunpy"
USER_AGENT = "kias-solar-wind-hmi-stage2/1.0"

START_DATE = "2011-01-01"
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


def require_modules(names: tuple[str, ...]) -> None:
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        print(f"missing_required_dependencies={','.join(missing)}", flush=True)
        print(INSTALL_HINT, flush=True)
        raise SystemExit(2)


def require_download_environment() -> str:
    require_modules(("astropy", "drms"))
    email = os.environ.get("JSOC_EMAIL", "").strip()
    if not email:
        print('export JSOC_EMAIL="your_email@example.com"', flush=True)
        raise SystemExit(2)
    return email


def require_extract_environment() -> None:
    require_modules(("astropy",))


def month_starts(start: str, end: str) -> list[pd.Timestamp]:
    starts = pd.date_range(pd.Timestamp(start).replace(day=1), pd.Timestamp(end), freq="MS")
    return [pd.Timestamp(ts) for ts in starts]


def selected_month_starts(start: str, end: str, max_months: int | None = None) -> list[pd.Timestamp]:
    months = month_starts(start, end)
    if max_months is not None:
        if max_months < 1:
            raise ValueError("--max-months must be >= 1 when provided.")
        months = months[:max_months]
    return months


def effective_end_date(start: str, end: str, max_months: int | None = None) -> str:
    months = selected_month_starts(start, end, max_months)
    if not months:
        return end
    last_month = months[-1]
    _, n_days = calendar.monthrange(last_month.year, last_month.month)
    return min(last_month + pd.Timedelta(days=n_days - 1), pd.Timestamp(end)).strftime("%Y-%m-%d")


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


def planned_month_records(month_start: pd.Timestamp, end: str) -> list[RecordInfo]:
    records: list[RecordInfo] = []
    for day in month_days(month_start, end):
        requested = day.replace(hour=0, minute=0, second=0, microsecond=0)
        records.append(
            RecordInfo(
                month=month_start.strftime("%Y-%m"),
                requested_time=requested,
                matched_time=requested,
                matched_time_le_requested=True,
            )
        )
    return records


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
    email = require_download_environment()
    import drms

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = drms.Client(email=email)
    rows: list[dict[str, Any]] = []
    for month_start in selected_month_starts(args.start_date, args.end_date, args.max_months):
        month = month_start.strftime("%Y-%m")
        try:
            records = query_month_records(client, month_start, args.end_date)
        except Exception as exc:
            print(f"month={month} metadata_query_failed={type(exc).__name__}: {exc}", flush=True)
            records = planned_month_records(month_start, args.end_date)
            rows.extend(
                report_row(
                    record,
                    requested_records=len(records),
                    downloaded_records=0,
                    skipped_existing_files=0,
                    failed_records=1,
                    export_method="metadata_query",
                    status="metadata_query_failed_using_planned_daily_times",
                    jsoc_query=month_query(month_start, len(records)),
                    error_message=f"{type(exc).__name__}: {exc}",
                )
                for record in records
            )
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
            month_rows, _, _, _ = download_record_urls(records, urls, args.timeout)
            for row in month_rows:
                row["jsoc_query"] = query
                row["export_status"] = row["export_status"] or export_status
            covered_keys = {
                pd.Timestamp(row["matched_hmi_time"]).strftime("%Y%m%d_%H%M%S")
                for row in month_rows
                if row["status"] in {"downloaded", "skipped_existing"}
            }
            fallback_records = [
                record
                for record in records
                if record.matched_time.strftime("%Y%m%d_%H%M%S") not in covered_keys
            ]
            for record in fallback_records:
                month_rows.append(
                    report_row(
                        record,
                        requested_records=len(records),
                        downloaded_records=0,
                        skipped_existing_files=0,
                        failed_records=1,
                        export_method="monthly_batch",
                        status="monthly_batch_uncovered_retrying_daily",
                        jsoc_query=query,
                        export_status=export_status,
                        error_message="No successful FITS file from monthly export for this requested record.",
                    )
                )
            if fallback_records:
                print(f"month={month} daily_fallback_records={len(fallback_records)}", flush=True)
                fallback_rows = daily_fallback(
                    client,
                    fallback_records,
                    args.timeout,
                    args.export_timeout,
                    args.sleep_seconds,
                )
                month_rows.extend(fallback_rows)
            final_success_keys = {
                pd.Timestamp(row["matched_hmi_time"]).strftime("%Y%m%d_%H%M%S")
                for row in month_rows
                if row["status"] in {"downloaded", "skipped_existing"}
            }
            final_downloaded = sum(row["status"] == "downloaded" for row in month_rows)
            final_skipped = sum(row["status"] == "skipped_existing" for row in month_rows)
            final_failed = len(records) - len(final_success_keys)
            rows.extend(month_rows)
            print(
                f"month={month} final downloaded={final_downloaded} skipped={final_skipped} failed={final_failed}",
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
    DOWNLOAD_EVENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(DOWNLOAD_EVENTS_CSV, index=False)
    return report


def raw_files(
    start: str = START_DATE,
    end: str = END_DATE,
    max_months: int | None = None,
) -> list[Path]:
    first = pd.Timestamp(start)
    last = pd.Timestamp(effective_end_date(start, end, max_months))
    files: list[Path] = []
    for path in sorted(RAW_DIR.glob("hmi.m_720s.*_TAI*.magnetogram.fits")):
        if not is_probable_fits(path):
            continue
        ts = fits_time(path)
        if ts is None or ts < first or ts > last + pd.Timedelta(days=1) - pd.Timedelta(seconds=1):
            continue
        files.append(path)
    return files


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


def sniff_raw_file_kind(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            head = handle.read(512)
    except OSError as exc:
        return f"read_error:{type(exc).__name__}"
    stripped = head.lstrip().lower()
    if head.startswith(b"SIMPLE"):
        return "fits"
    if stripped.startswith(b"<!doctype html") or stripped.startswith(b"<html") or b"<html" in stripped[:128]:
        return "html_or_error_page"
    if b"error" in stripped[:256] or b"exception" in stripped[:256]:
        return "text_error_response"
    return "not_fits_unknown"


def validate_raw_file(path: Path) -> dict[str, Any]:
    kind = sniff_raw_file_kind(path)
    row: dict[str, Any] = {
        "local_path": str(path),
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size if path.exists() else np.nan,
        "parsed_time": fits_time(path),
        "file_kind": kind,
        "is_real_fits": False,
        "image_hdu_index": np.nan,
        "has_image_data_hdu": False,
        "data_shape": "",
        "finite_pixel_count": np.nan,
        "DATE_OBS": "",
        "T_REC": "",
        "error_message": "",
    }
    if kind != "fits":
        return row
    require_extract_environment()
    from astropy.io import fits

    try:
        with fits.open(path, memmap=False) as hdul:
            row["is_real_fits"] = True
            for idx, hdu in enumerate(hdul):
                data = getattr(hdu, "data", None)
                if data is None:
                    continue
                arr = np.asarray(data)
                row["image_hdu_index"] = idx
                row["has_image_data_hdu"] = True
                row["data_shape"] = "x".join(str(dim) for dim in arr.shape)
                row["finite_pixel_count"] = int(np.isfinite(arr).sum())
                header = hdu.header
                row["DATE_OBS"] = str(header.get("DATE-OBS", ""))
                row["T_REC"] = str(header.get("T_REC", ""))
                break
    except Exception as exc:
        row["error_message"] = f"{type(exc).__name__}: {exc}"
    return row


def validate_raw_files(start: str, end: str, max_months: int | None = None) -> pd.DataFrame:
    rows = [validate_raw_file(path) for path in raw_candidate_files(start, end, max_months)]
    report = pd.DataFrame(rows)
    RAW_VALIDATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(RAW_VALIDATION_CSV, index=False)
    return report


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


def extract_features(
    start: str,
    end: str,
    max_months: int | None = None,
    force: bool = False,
    delete_raw_after_extract: bool = False,
) -> pd.DataFrame:
    require_extract_environment()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = raw_files(start, end, max_months)
    if FEATURE_CSV.exists() and not force:
        existing = pd.read_csv(FEATURE_CSV)
        done = set(existing.get("local_path", pd.Series(dtype=str)).astype(str))
        rows = existing.to_dict("records")
    else:
        done = set()
        rows = []
    for idx, path in enumerate(files, start=1):
        if str(path) in done:
            if delete_raw_after_extract:
                path.unlink(missing_ok=True)
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
        else:
            if delete_raw_after_extract:
                path.unlink(missing_ok=True)
        if idx % 50 == 0:
            print(f"feature_extraction_processed_files={idx}/{len(files)}", flush=True)
    features = pd.DataFrame(rows).sort_values("magnetogram_time").reset_index(drop=True)
    FEATURE_CSV.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURE_CSV, index=False)
    return features


def raw_disk_usage_bytes(start: str = START_DATE, end: str = END_DATE, max_months: int | None = None) -> int:
    return int(sum(path.stat().st_size for path in raw_files(start, end, max_months)))


def raw_candidate_files(
    start: str = START_DATE,
    end: str = END_DATE,
    max_months: int | None = None,
) -> list[Path]:
    first = pd.Timestamp(start)
    last = pd.Timestamp(effective_end_date(start, end, max_months))
    candidates: list[Path] = []
    for path in sorted(RAW_DIR.glob("hmi.m_720s.*")):
        if not path.is_file():
            continue
        ts = fits_time(path)
        if ts is None or ts < first or ts > last + pd.Timedelta(days=1) - pd.Timedelta(seconds=1):
            continue
        candidates.append(path)
    return candidates


def raw_file_day(path: Path) -> str | None:
    ts = fits_time(path)
    return ts.strftime("%Y-%m-%d") if ts is not None else None


def invalid_raw_files(
    start: str = START_DATE,
    end: str = END_DATE,
    max_months: int | None = None,
) -> list[Path]:
    return [path for path in raw_candidate_files(start, end, max_months) if not is_probable_fits(path)]


def build_download_availability_report(
    download_report: pd.DataFrame,
    features: pd.DataFrame,
    start: str,
    end: str,
    max_months: int | None = None,
    extraction_run: bool = False,
) -> pd.DataFrame:
    effective_end = effective_end_date(start, end, max_months)
    days = pd.date_range(start, effective_end, freq="D")
    requested_day_set = set(days.strftime("%Y-%m-%d"))
    valid_raw = raw_files(start, end, max_months)
    valid_raw_days = {day for path in valid_raw if (day := raw_file_day(path)) is not None}
    invalid_raw = invalid_raw_files(start, end, max_months)
    newly_downloaded_files = 0
    if not download_report.empty and {"status", "local_path", "matched_hmi_time"}.issubset(download_report.columns):
        downloaded = download_report[download_report["status"].eq("downloaded")].copy()
        if not downloaded.empty:
            downloaded["day"] = pd.to_datetime(downloaded["matched_hmi_time"], errors="coerce").dt.strftime("%Y-%m-%d")
            downloaded = downloaded[downloaded["day"].isin(requested_day_set)]
            newly_downloaded_files = int(downloaded["local_path"].dropna().nunique())
    missing_after_download = requested_day_set - valid_raw_days

    rows: list[dict[str, Any]] = [
        {
            "report_type": "download_status",
            "item": "requested_days",
            "value": len(days),
            "note": "",
        },
        {
            "report_type": "download_status",
            "item": "existing_raw_files",
            "value": len(valid_raw),
            "note": "Valid local FITS files in the requested date range after the download step.",
        },
        {
            "report_type": "download_status",
            "item": "newly_downloaded_files",
            "value": newly_downloaded_files,
            "note": "Unique local FITS paths newly downloaded in this run/report.",
        },
        {
            "report_type": "download_status",
            "item": "missing_after_download",
            "value": len(missing_after_download),
            "note": "Requested days with no valid local raw FITS file.",
        },
        {
            "report_type": "download_status",
            "item": "invalid_raw_files",
            "value": len(invalid_raw),
            "note": "Local raw candidates in the requested date range that are not valid FITS files.",
        },
        {
            "report_type": "download_status",
            "item": "raw_fits_disk_usage_bytes",
            "value": raw_disk_usage_bytes(start, end, max_months),
            "note": "",
        },
    ]

    by_year = pd.DataFrame({"day": days})
    by_year["year"] = by_year["day"].dt.year
    by_year["valid_raw"] = by_year["day"].dt.strftime("%Y-%m-%d").isin(valid_raw_days)
    for year, group in by_year.groupby("year"):
        rows.append(
            {
                "report_type": "download_status_by_year",
                "item": int(year),
                "value": int(group["valid_raw"].sum()),
                "note": f"valid_raw_days={int(group['valid_raw'].sum())} requested_days={len(group)}",
            }
        )

    if not extraction_run:
        rows.append(
            {
                "report_type": "feature_availability",
                "item": "extraction_not_run",
                "value": True,
                "note": "Feature availability was not computed because --extract was not run.",
            }
        )
        return pd.DataFrame(rows)

    valid_features = features
    if "extraction_error" in valid_features.columns:
        valid_features = valid_features[valid_features["extraction_error"].isna()]
    if not valid_features.empty:
        mag_times = pd.to_datetime(valid_features["magnetogram_time"])
        valid_features = valid_features[(mag_times >= days.min()) & (mag_times <= days.max() + pd.Timedelta(days=1))]
    for window in WINDOWS:
        cols = [f"{window}_{metric}" for metric in MAG_METRICS if f"{window}_{metric}" in features.columns]
        if cols and not valid_features.empty:
            nan_rate = float(valid_features[cols].isna().mean().mean())
        else:
            nan_rate = 1.0
        rows.append(
            {
                "report_type": "feature_availability",
                "item": window,
                "value": nan_rate,
                "note": "feature_nan_rate_for_approximate_window_magnetogram_features",
            }
        )
    return pd.DataFrame(rows)


def load_stage2_features() -> pd.DataFrame:
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


def stage2_timestamp_sanity(mag_sanity: pd.DataFrame) -> pd.DataFrame:
    candidate = mag_sanity.copy()
    origin_year = pd.to_datetime(candidate["origin_datetime"]).dt.year
    candidate = candidate[(origin_year >= 2011) & (origin_year <= 2025)]
    matched = candidate[candidate["mag_time"].notna()]
    if len(matched) >= 20:
        candidate = matched
    if candidate.empty:
        candidate = mag_sanity
    sample = candidate.sample(n=min(20, len(candidate)), random_state=20260624).copy()
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


def summarize_stage2(fixed: pd.DataFrame, cv: pd.DataFrame, private: pd.DataFrame) -> pd.DataFrame:
    cv_mean = (
        cv.groupby(["feature_set_name", "model_name"], dropna=False)
        .agg(
            cv_mean_mae=("mae", "mean"),
            cv_mean_rmse=("rmse", "mean"),
            cv_mean_cc=("cc", "mean"),
            cv_mean_skill=("mae_skill_vs_27day", "mean"),
        )
        .reset_index()
    )
    fixed_s = fixed[["feature_set_name", "model_name", "mae", "rmse", "cc", "mae_skill_vs_27day"]].rename(
        columns={"mae": "fixed_mae", "rmse": "fixed_rmse", "cc": "fixed_cc", "mae_skill_vs_27day": "fixed_skill"}
    )
    private_s = private[["feature_set_name", "model_name", "mae", "rmse", "cc", "mae_skill_vs_27day"]].rename(
        columns={"mae": "private_mae", "rmse": "private_rmse", "cc": "private_cc", "mae_skill_vs_27day": "private_skill"}
    )
    return fixed_s.merge(cv_mean, on=["feature_set_name", "model_name"], how="left").merge(
        private_s, on=["feature_set_name", "model_name"], how="left"
    )


def select_public_best(fixed_df: pd.DataFrame, cv_df: pd.DataFrame) -> str:
    ensemble = ENSEMBLE_MODEL
    cv_rank = (
        cv_df[cv_df["model_name"] == ensemble]
        .groupby("feature_set_name", dropna=False)
        .agg(cv_mean_cc=("cc", "mean"), cv_mean_mae=("mae", "mean"))
        .reset_index()
    )
    fixed_rank = fixed_df[fixed_df["model_name"] == ensemble][["feature_set_name", "cc", "mae"]].rename(
        columns={"cc": "fixed_cc", "mae": "fixed_mae"}
    )
    public = cv_rank.merge(fixed_rank, on="feature_set_name", how="left")
    if public.empty:
        return "current_best"
    best = public.sort_values(["cv_mean_cc", "fixed_cc", "cv_mean_mae"], ascending=[False, False, True]).iloc[0]
    return str(best["feature_set_name"])


def evaluate_stage2(features_df: pd.DataFrame, availability: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if features_df.empty:
        raise RuntimeError(f"No magnetogram features available at {FEATURE_CSV}. Run --extract first.")
    mag = load_stage2_features()
    baseline_table, baseline_features, _ = build_baseline_all()
    mag_table, mag_cols, mag_sanity = add_magnetogram_features(baseline_table, mag, tolerance_hours=36)
    models = model_configs()

    tables = {
        "current_best": (baseline_table, baseline_features),
        "current_best_plus_magnetogram_window_features": (mag_table, baseline_features + mag_cols),
    }

    fixed = tab.fixed_split()
    fixed_train = finite_target_idx(baseline_table, list(range(fixed.train_start, fixed.train_end + 1)))
    fixed_eval = finite_target_idx(baseline_table, list(range(fixed.val_start, fixed.val_end + 1)))
    private_train = finite_target_idx(baseline_table, list(range(2011, 2024)))
    private_eval = finite_target_idx(baseline_table, [2024, 2025])

    fixed_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    private_predictions: list[pd.DataFrame] = []
    for name, (table, cols) in tables.items():
        rows, _ = evaluate_indices(
            table,
            cols,
            fixed_train,
            fixed_eval,
            name,
            fixed.scheme,
            fixed.fold,
            models,
        )
        fixed_rows.extend(rows)
        for split in tab.cv_splits():
            cv_train = finite_target_idx(baseline_table, list(range(split.train_start, split.train_end + 1)))
            cv_eval = finite_target_idx(baseline_table, list(range(split.val_start, split.val_end + 1)))
            rows, _ = evaluate_indices(
                table,
                cols,
                cv_train,
                cv_eval,
                name,
                split.scheme,
                split.fold,
                models,
            )
            cv_rows.extend(rows)
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
    cv_df = pd.DataFrame(cv_rows)
    private_df = pd.DataFrame(private_rows)
    summary = summarize_stage2(fixed_df, cv_df, private_df)
    best_name = select_public_best(fixed_df, cv_df)

    fixed_df.to_csv(OUT_DIR / "stage2_fixed_results.csv", index=False)
    cv_df.to_csv(OUT_DIR / "stage2_cv_results.csv", index=False)
    private_df.to_csv(OUT_DIR / "stage2_private_diagnostic.csv", index=False)
    summary.to_csv(OUT_DIR / "stage2_summary.csv", index=False)
    timestamp = stage2_timestamp_sanity(mag_sanity)
    timestamp.to_csv(OUT_DIR / "stage2_timestamp_sanity_check.csv", index=False)
    availability.to_csv(OUT_DIR / "stage2_data_availability_report.csv", index=False)

    pred = pd.concat(private_predictions, ignore_index=True)
    pred = pred[pred["feature_set_name"] == best_name][["target_datetime", "predicted_speed"]].rename(
        columns={"target_datetime": "datetime"}
    )
    pred.to_csv(OUT_DIR / "stage2_best_private_prediction.csv", index=False)

    print("\nData availability report", flush=True)
    print(availability.to_string(index=False), flush=True)
    print("\nTimestamp sanity check", flush=True)
    print(timestamp.to_string(index=False), flush=True)
    print("\nConcise comparison", flush=True)
    cols = [
        "feature_set_name",
        "model_name",
        "fixed_mae",
        "fixed_rmse",
        "fixed_cc",
        "cv_mean_mae",
        "cv_mean_cc",
        "private_mae",
        "private_rmse",
        "private_cc",
    ]
    print(summary[cols].to_string(index=False), flush=True)
    print(f"\nBest public-selected feature set: {best_name}", flush=True)
    print("Private rows are diagnostic only.", flush=True)


def print_dry_run(args: argparse.Namespace) -> None:
    months = selected_month_starts(args.start_date, args.end_date, args.max_months)
    effective_end = effective_end_date(args.start_date, args.end_date, args.max_months)
    print("Dry run: no downloads, feature extraction, or model fitting will be run.", flush=True)
    print(f"date_range={args.start_date}..{effective_end}", flush=True)
    print(f"months={len(months)} raw_dir={RAW_DIR}", flush=True)
    for month_start in months:
        days = month_days(month_start, args.end_date)
        monthly = month_query(month_start, len(days))
        batch = f"{monthly}{{{HMI_SEGMENT}}}"
        print(f"month={month_start.strftime('%Y-%m')} monthly_query={batch}", flush=True)
        for day in days:
            record = RecordInfo(
                month=month_start.strftime("%Y-%m"),
                requested_time=day,
                matched_time=day,
                matched_time_le_requested=True,
            )
            print(f"month={record.month} daily_fallback_query={single_export_query(record)}", flush=True)
    print_manual_commands()


def print_manual_commands() -> None:
    print("\nManual commands", flush=True)
    print('export JSOC_EMAIL="your_email@example.com"', flush=True)
    print("", flush=True)
    print(
        ".venv/bin/python run_magnetogram_stage2_full.py --dry-run "
        "--start-date 2011-01-01 --end-date 2025-12-31",
        flush=True,
    )
    print(
        ".venv/bin/python run_magnetogram_stage2_full.py --download "
        "--start-date 2011-01-01 --end-date 2025-12-31 --max-months 1",
        flush=True,
    )
    print(
        ".venv/bin/python run_magnetogram_stage2_full.py --validate-raw "
        "--start-date 2011-01-01 --end-date 2025-12-31 --max-months 1",
        flush=True,
    )
    print(
        ".venv/bin/python run_magnetogram_stage2_full.py --download --extract "
        "--start-date 2011-01-01 --end-date 2025-12-31",
        flush=True,
    )
    print(
        ".venv/bin/python run_magnetogram_stage2_full.py --evaluate "
        "--start-date 2011-01-01 --end-date 2025-12-31",
        flush=True,
    )


def print_recommended_after_download(args: argparse.Namespace) -> None:
    print("\nRecommended next command", flush=True)
    print(
        ".venv/bin/python run_magnetogram_stage2_full.py --extract "
        f"--start-date {args.start_date} --end-date {effective_end_date(args.start_date, args.end_date, args.max_months)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--validate-raw", action="store_true")
    parser.add_argument("--delete-raw-after-extract", action="store_true")
    parser.add_argument("--max-months", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--export-timeout", type=int, default=900)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--force-feature-extract", action="store_true")
    args = parser.parse_args()

    if pd.Timestamp(args.start_date) > pd.Timestamp(args.end_date):
        raise SystemExit("--start-date must be <= --end-date")
    if args.max_months is not None and args.max_months < 1:
        raise SystemExit("--max-months must be >= 1")

    if args.dry_run:
        print_dry_run(args)
        return

    if not (args.download or args.extract or args.evaluate or args.validate_raw):
        print("No action selected. Use --download, --extract, --evaluate, --validate-raw, or --dry-run.", flush=True)
        print_manual_commands()
        return

    MAG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.download:
        download_report = download_hmi_monthly(args)
    elif DOWNLOAD_EVENTS_CSV.exists():
        download_report = pd.read_csv(DOWNLOAD_EVENTS_CSV)
    elif DOWNLOAD_REPORT_CSV.exists():
        download_report = pd.read_csv(DOWNLOAD_REPORT_CSV)
    else:
        download_report = pd.DataFrame()

    if args.extract:
        features = extract_features(
            args.start_date,
            args.end_date,
            max_months=args.max_months,
            force=args.force_feature_extract,
            delete_raw_after_extract=args.delete_raw_after_extract,
        )
    elif FEATURE_CSV.exists():
        features = pd.read_csv(FEATURE_CSV)
    else:
        features = pd.DataFrame()

    availability = build_download_availability_report(
        download_report,
        features,
        args.start_date,
        args.end_date,
        args.max_months,
        extraction_run=args.extract,
    )
    availability.to_csv(OUT_DIR / "stage2_data_availability_report.csv", index=False)
    availability[availability["report_type"].str.startswith("download_status")].to_csv(
        DOWNLOAD_REPORT_CSV,
        index=False,
    )
    if args.download or args.extract:
        print("\nDownload/feature availability", flush=True)
        print(availability.to_string(index=False), flush=True)

    if args.validate_raw:
        raw_validation = validate_raw_files(args.start_date, args.end_date, args.max_months)
        print("\nRaw validation summary", flush=True)
        if raw_validation.empty:
            print("No raw candidate files found in the requested date range.", flush=True)
        else:
            summary = (
                raw_validation.groupby(["file_kind", "has_image_data_hdu"], dropna=False)
                .size()
                .reset_index(name="count")
            )
            print(summary.to_string(index=False), flush=True)
            print(f"Saved raw validation report to {RAW_VALIDATION_CSV}", flush=True)

    if args.evaluate:
        evaluate_stage2(features, availability)

    if args.download and not args.extract and not args.evaluate:
        print_recommended_after_download(args)

    print_manual_commands()


if __name__ == "__main__":
    main()
