"""Stage 0 HMI daily magnetogram probe and simple window feature extraction.

This is a feasibility probe only. It uses SDO/HMI line-of-sight magnetograms
from JSOC, downloads a small two-day daily sample near 00:00, stores raw FITS
files, and extracts approximate central-window radial-field summaries.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data" / "magnetograms"
RAW_DIR = OUT_DIR / "hmi_raw"
FEATURE_CSV = OUT_DIR / "magnetogram_window_features_daily.csv"
REPORT_CSV = OUT_DIR / "stage0_download_report.csv"

HMI_SERIES = "hmi.M_720s"
HMI_SEGMENT = "magnetogram"

DEFAULT_START = "2022-01-01"
DEFAULT_END = "2022-01-03"
DEFAULT_TARGET_HHMM = "0000"
USER_AGENT = "kias-solar-wind-hmi-stage0/1.0"

WINDOWS = {
    "W_lon7p5_lat15": (7.5, 15.0),
    "W_lon30_lat15": (30.0, 15.0),
    "W_lon30_lat30": (30.0, 30.0),
    "W_lon60_lat60": (60.0, 60.0),
}

REQUIRED_DEPENDENCIES = ("astropy", "drms")
OPTIONAL_DEPENDENCIES = ("sunpy",)
INSTALL_HINT = "uv add astropy drms sunpy"


@dataclass(frozen=True)
class DependencyStatus:
    missing_required: tuple[str, ...]
    missing_optional: tuple[str, ...]

    @property
    def ready_for_download(self) -> bool:
        return not self.missing_required

    @property
    def any_missing(self) -> bool:
        return bool(self.missing_required or self.missing_optional)


@dataclass(frozen=True)
class DownloadResult:
    requested_date: str
    requested_time: str
    matched_magnetogram_time: str
    matched_time_le_requested: bool | None
    source: str
    product: str
    status: str
    jsoc_query: str
    url: str
    export_status: str
    local_path: str
    exception_message: str
    note: str


class JsocQueryError(RuntimeError):
    def __init__(self, query: str, exc: Exception) -> None:
        self.query = query
        self.original_exception = exc
        super().__init__(f"{type(exc).__name__}: {exc}")


def dependency_status() -> DependencyStatus:
    missing_required = tuple(
        name for name in REQUIRED_DEPENDENCIES if importlib.util.find_spec(name) is None
    )
    missing_optional = tuple(
        name for name in OPTIONAL_DEPENDENCIES if importlib.util.find_spec(name) is None
    )
    return DependencyStatus(missing_required, missing_optional)


def print_dependency_status(status: DependencyStatus) -> None:
    if status.missing_required:
        print(f"missing_required_dependencies={','.join(status.missing_required)}", flush=True)
    if status.missing_optional:
        print(f"missing_optional_dependencies={','.join(status.missing_optional)}", flush=True)
    if status.any_missing:
        print(INSTALL_HINT, flush=True)


def empty_feature_frame() -> pd.DataFrame:
    cols = [
        "requested_time",
        "magnetogram_time",
        "matched_time_le_requested",
        "source",
        "product",
        "local_path",
        "coordinate_mode",
        "field_mode",
    ]
    metrics = [
        "n_pix",
        "mean_abs_B",
        "median_abs_B",
        "signed_mean_B",
        "sum_pos_B",
        "sum_neg_B",
        "polarity_imbalance",
        "dominant_polarity",
    ]
    for window in WINDOWS:
        cols.extend([f"{window}_{metric}" for metric in metrics])
    return pd.DataFrame(columns=cols)


def write_report(results: list[DownloadResult]) -> pd.DataFrame:
    report = pd.DataFrame([result.__dict__ for result in results])
    report.to_csv(REPORT_CSV, index=False)
    return report


def write_empty_outputs(results: list[DownloadResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = write_report(results)
    features = empty_feature_frame()
    features.to_csv(FEATURE_CSV, index=False)
    return report, features


def iter_daily_requested_times(start: str, end: str, target_hhmm: str) -> list[pd.Timestamp]:
    """Return a half-open daily cadence: start <= day < end."""
    start_day = pd.Timestamp(start)
    end_day = pd.Timestamp(end)
    hour = int(target_hhmm[:2])
    minute = int(target_hhmm[2:])
    days: list[pd.Timestamp] = []
    current = start_day
    while current < end_day:
        days.append(current.replace(hour=hour, minute=minute, second=0, microsecond=0))
        current += pd.Timedelta(days=1)
    return days


def jsoc_time_token(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y.%m.%d_%H:%M:%S_TAI")


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


def make_probe_query(requested_time: pd.Timestamp, lookback_hours: int) -> str:
    start = requested_time - pd.Timedelta(hours=lookback_hours)
    return f"{HMI_SERIES}[{jsoc_time_token(start)}/{lookback_hours}h@720s]"


def make_record_query(magnetogram_time: pd.Timestamp) -> str:
    return f"{HMI_SERIES}[{jsoc_time_token(magnetogram_time)}]"


def make_export_query(magnetogram_time: pd.Timestamp) -> str:
    return f"{make_record_query(magnetogram_time)}{{{HMI_SEGMENT}}}"


def request_row(
    requested_time: pd.Timestamp,
    status: str,
    jsoc_query: str = "",
    matched_magnetogram_time: str = "",
    matched_time_le_requested: bool | None = None,
    url: str = "",
    export_status: str = "",
    local_path: str = "",
    exception_message: str = "",
    note: str = "",
) -> DownloadResult:
    return DownloadResult(
        requested_date=requested_time.strftime("%Y-%m-%d"),
        requested_time=requested_time.strftime("%Y-%m-%d %H:%M:%S"),
        matched_magnetogram_time=matched_magnetogram_time,
        matched_time_le_requested=matched_time_le_requested,
        source="HMI",
        product=HMI_SERIES,
        status=status,
        jsoc_query=jsoc_query,
        url=url,
        export_status=export_status,
        local_path=local_path,
        exception_message=exception_message,
        note=note,
    )


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


def choose_download_url(export_request: Any) -> str:
    urls = dataframe_urls(getattr(export_request, "urls", None))
    fits_urls = [
        url for url in urls if urllib.parse.urlparse(url).path.lower().endswith((".fits", ".fit", ".fits.gz"))
    ]
    if fits_urls:
        return fits_urls[0]
    if urls:
        return urls[0]
    raise RuntimeError("JSOC export completed but no downloadable URL was returned.")


def local_hmi_path(download_url: str, magnetogram_time: pd.Timestamp) -> Path:
    name = Path(urllib.parse.urlparse(download_url).path).name
    if not name or "." not in name:
        name = f"hmi_M_720s_{magnetogram_time.strftime('%Y%m%d_%H%M%S')}_TAI_magnetogram.fits"
    return RAW_DIR / name


def download_url(url: str, path: Path, timeout: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(path)


def find_latest_hmi_record_at_or_before(
    client: Any,
    requested_time: pd.Timestamp,
    lookback_hours: int,
) -> tuple[pd.Timestamp | None, str]:
    exact_query = make_record_query(requested_time)
    try:
        exact_records = client.query(exact_query, key="T_REC,QUALITY")
    except Exception as exc:
        raise JsocQueryError(exact_query, exc) from exc
    if exact_records is not None and len(exact_records) > 0:
        if "T_REC" not in exact_records.columns:
            raise RuntimeError(f"JSOC query returned columns {list(exact_records.columns)}, missing T_REC.")
        exact_time = parse_jsoc_time(exact_records.iloc[0]["T_REC"])
        if exact_time is not None and exact_time <= requested_time:
            return exact_time, exact_query

    probe_query = make_probe_query(requested_time, lookback_hours)
    try:
        records = client.query(probe_query, key="T_REC,QUALITY")
    except Exception as exc:
        raise JsocQueryError(probe_query, exc) from exc
    if records is None or len(records) == 0:
        return None, probe_query
    if "T_REC" not in records.columns:
        raise RuntimeError(f"JSOC query returned columns {list(records.columns)}, missing T_REC.")

    candidates: list[pd.Timestamp] = []
    for value in records["T_REC"].tolist():
        record_time = parse_jsoc_time(value)
        if record_time is not None and record_time <= requested_time:
            candidates.append(record_time)
    if not candidates:
        return None, probe_query
    return max(candidates), probe_query


def try_hmi_download_for_time(
    client: Any,
    requested_time: pd.Timestamp,
    lookback_hours: int,
    timeout: int,
    export_timeout: int,
    dry_run: bool,
) -> DownloadResult:
    try:
        matched_time, probe_query = find_latest_hmi_record_at_or_before(
            client, requested_time, lookback_hours
        )
    except Exception as exc:
        failed_query = getattr(exc, "query", make_probe_query(requested_time, lookback_hours))
        return request_row(
            requested_time,
            status="query_failed",
            jsoc_query=failed_query,
            exception_message=f"{type(exc).__name__}: {exc}",
        )

    if matched_time is None:
        return request_row(
            requested_time,
            status="no_record_at_or_before_requested_time",
            jsoc_query=probe_query,
            note="No hmi.M_720s T_REC was found at or before the requested time in the lookback window.",
        )

    matched_text = matched_time.strftime("%Y-%m-%d %H:%M:%S")
    le_requested = bool(matched_time <= requested_time)
    export_query = make_export_query(matched_time)
    print(
        "matched_magnetogram_time="
        f"{matched_text} requested_time={requested_time:%Y-%m-%d %H:%M:%S} "
        f"matched_time_le_requested={le_requested}",
        flush=True,
    )

    path = RAW_DIR / f"hmi_M_720s_{matched_time.strftime('%Y%m%d_%H%M%S')}_TAI_magnetogram.fits"
    if path.exists():
        return request_row(
            requested_time,
            status="cached",
            jsoc_query=export_query,
            matched_magnetogram_time=matched_text,
            matched_time_le_requested=le_requested,
            local_path=str(path),
        )
    if dry_run:
        return request_row(
            requested_time,
            status="dry_run_matched",
            jsoc_query=export_query,
            matched_magnetogram_time=matched_text,
            matched_time_le_requested=le_requested,
            local_path=str(path),
        )

    export_request: Any | None = None
    try:
        export_request = client.export(export_query, method="url", protocol="fits")
        export_request.wait(timeout=export_timeout)
        export_status = export_status_text(export_request)
        url = choose_download_url(export_request)
        path = local_hmi_path(url, matched_time)
        download_url(url, path, timeout)
        return request_row(
            requested_time,
            status="downloaded",
            jsoc_query=export_query,
            matched_magnetogram_time=matched_text,
            matched_time_le_requested=le_requested,
            url=url,
            export_status=export_status,
            local_path=str(path),
        )
    except Exception as exc:
        export_status = export_status_text(export_request) if export_request is not None else ""
        url = ""
        if export_request is not None:
            try:
                urls = dataframe_urls(export_request.urls)
                url = ";".join(urls)
            except Exception:
                url = ""
        return request_row(
            requested_time,
            status="download_failed",
            jsoc_query=export_query,
            matched_magnetogram_time=matched_text,
            matched_time_le_requested=le_requested,
            url=url,
            export_status=export_status,
            exception_message=f"{type(exc).__name__}: {exc}",
        )


def get_fits_module() -> Any:
    from astropy.io import fits

    return fits


def open_fits_array(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    fits = get_fits_module()
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
        "polarity_imbalance": float(abs(sum_b) / sum_abs) if sum_abs > 0 else np.nan,
        "dominant_polarity": float(np.sign(sum_b)) if sum_abs > 0 else np.nan,
    }


def extract_features_for_file(item: pd.Series) -> dict[str, Any]:
    path = Path(str(item["local_path"]))
    data, header = open_fits_array(path)
    lon, lat, mu, valid_disk = approximate_hmi_disk_geometry(data.shape, header)
    good_mu = valid_disk & (mu >= 0.2)
    radial_field = np.full_like(data, np.nan, dtype=np.float64)
    radial_field[good_mu] = data[good_mu] / mu[good_mu]

    row: dict[str, Any] = {
        "requested_time": item["requested_time"],
        "magnetogram_time": item["matched_magnetogram_time"],
        "matched_time_le_requested": item["matched_time_le_requested"],
        "source": item["source"],
        "product": item["product"],
        "local_path": str(path),
        "coordinate_mode": "approximate_geometric_disk_center",
        "field_mode": "Br_approx_equals_B_los_over_mu_mu_ge_0p2",
    }
    for window, (lon_half, lat_half) in WINDOWS.items():
        mask = good_mu & (np.abs(lon) <= lon_half) & (np.abs(lat) <= lat_half)
        stats = summarize_field(radial_field[mask])
        for key, value in stats.items():
            row[f"{window}_{key}"] = value
    return row


def extract_features(report: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, item in report.iterrows():
        if item["status"] not in {"cached", "downloaded"}:
            continue
        local_path = str(item.get("local_path", ""))
        if not local_path or not Path(local_path).exists():
            continue
        try:
            rows.append(extract_features_for_file(item))
        except Exception as exc:
            rows.append(
                {
                    "requested_time": item["requested_time"],
                    "magnetogram_time": item["matched_magnetogram_time"],
                    "matched_time_le_requested": item["matched_time_le_requested"],
                    "source": item["source"],
                    "product": item["product"],
                    "local_path": local_path,
                    "coordinate_mode": "approximate_geometric_disk_center",
                    "field_mode": "Br_approx_equals_B_los_over_mu_mu_ge_0p2",
                    "extraction_error": f"{type(exc).__name__}: {exc}",
                }
            )
    if rows:
        return pd.DataFrame(rows)
    return empty_feature_frame()


def run_stage0(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    requested_times = iter_daily_requested_times(args.start, args.end, args.target_hhmm)
    deps = dependency_status()
    print_dependency_status(deps)

    if not deps.ready_for_download:
        results = [
            request_row(
                requested_time,
                status="missing_required_dependencies",
                note=f"Missing required dependencies: {','.join(deps.missing_required)}",
            )
            for requested_time in requested_times
        ]
        return write_empty_outputs(results)

    jsoc_email = os.environ.get("JSOC_EMAIL", "").strip()
    if not jsoc_email:
        results = [
            request_row(
                requested_time,
                status="missing_jsoc_email",
                note="JSOC_EMAIL is required for HMI download.",
            )
            for requested_time in requested_times
        ]
        return write_empty_outputs(results)

    import drms

    client = drms.Client(email=jsoc_email)
    results: list[DownloadResult] = []
    for idx, requested_time in enumerate(requested_times, start=1):
        result = try_hmi_download_for_time(
            client=client,
            requested_time=requested_time,
            lookback_hours=args.lookback_hours,
            timeout=args.timeout,
            export_timeout=args.export_timeout,
            dry_run=args.dry_run,
        )
        results.append(result)
        print(
            f"processed_days={idx} requested_time={result.requested_time} status={result.status}",
            flush=True,
        )
        if args.sleep_seconds > 0 and not args.dry_run:
            time.sleep(args.sleep_seconds)

    report = write_report(results)
    features = extract_features(report)
    features.to_csv(FEATURE_CSV, index=False)
    return report, features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--target-hhmm", default=DEFAULT_TARGET_HHMM)
    parser.add_argument("--lookback-hours", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--export-timeout", type=int, default=600)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report, features = run_stage0(args)
    status_counts = report["status"].value_counts(dropna=False).to_dict()

    print("\nStage 0 HMI download/query report", flush=True)
    print(
        f"date_range_half_open={args.start}..{args.end} "
        f"target_hhmm={args.target_hhmm} product={HMI_SERIES}",
        flush=True,
    )
    print(f"raw_dir={RAW_DIR}", flush=True)
    print(f"report_csv={REPORT_CSV}", flush=True)
    print(f"status_counts={status_counts}", flush=True)
    print("\nStage 1 feature extraction", flush=True)
    print(f"feature_csv={FEATURE_CSV}", flush=True)
    print(f"feature_rows={len(features)}", flush=True)
    if not features.empty:
        print(features.head().to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
