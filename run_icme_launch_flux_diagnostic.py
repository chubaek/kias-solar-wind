"""Post-event ICME launch-time HMI flux diagnostic for large 72h errors.

This script is not a causal forecasting feature.  It identifies bad private
prediction intervals, estimates launch times from observed near-Earth speed,
samples nearby HMI line-of-sight magnetograms, and computes simple unsigned
magnetic flux proxy summaries.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import shutil
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OFFICIAL_PREDICTION = HERE / "outputs" / "final_ch_representative_72h_ensemble.csv"
PRIVATE_CSV = HERE / "solar_wind-private.csv"
FULL_CSV = HERE / "solar_wind_data.csv"
OUT_DIR = HERE / "outputs" / "icme_launch_flux_diagnostic_72h"
ICME_RAW_DIR = HERE / "data" / "icme_diagnostics" / "hmi_raw"
LEGACY_RAW_DIR = HERE / "data" / "magnetograms" / "hmi_raw"

HMI_SERIES = "hmi.M_720s"
HMI_SEGMENT = "magnetogram"
AU_KM = 149_597_870.7
USER_AGENT = "kias-solar-wind-icme-diagnostic/1.0"


@dataclass(frozen=True)
class MatchedHMI:
    requested_time: pd.Timestamp
    matched_time: pd.Timestamp | None
    local_path: Path | None
    match_method: str
    time_offset_minutes: float
    status: str
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--speed-threshold", type=float, default=800.0)
    parser.add_argument("--abs-error-threshold", type=float, default=300.0)
    parser.add_argument("--merge-gap-hours", type=float, default=6.0)
    parser.add_argument("--expand-hours", type=float, default=12.0)
    parser.add_argument("--hmi-match", choices=["nearest", "backward"], default="nearest")
    parser.add_argument("--hmi-tolerance-minutes", type=float, default=90.0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--export-timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def cc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) <= 1e-8 or np.std(y_pred) <= 1e-8:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


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


def load_prediction_frame(year: int) -> pd.DataFrame:
    private = pd.read_csv(PRIVATE_CSV, parse_dates=["datetime"])
    pred = pd.read_csv(OFFICIAL_PREDICTION, parse_dates=["datetime"])
    pred = pred[["datetime", "predicted_speed"]].drop_duplicates("datetime", keep="last")
    frame = private[["datetime", "Speed (km/s)"]].merge(pred, on="datetime", how="left")
    frame = frame[frame["datetime"].dt.year == year].copy()
    frame["error"] = frame["predicted_speed"] - frame["Speed (km/s)"]
    frame["abs_error"] = frame["error"].abs()
    return frame.sort_values("datetime").reset_index(drop=True)


def merge_intervals(intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def identify_events(
    frame: pd.DataFrame,
    speed_threshold: float,
    abs_error_threshold: float,
    merge_gap_hours: float,
    expand_hours: float,
) -> pd.DataFrame:
    bad = frame[
        (frame["Speed (km/s)"] > speed_threshold)
        | (frame["abs_error"] > abs_error_threshold)
    ].copy()
    if bad.empty:
        return pd.DataFrame()

    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current_start = pd.Timestamp(bad.iloc[0]["datetime"])
    current_end = current_start
    max_gap = pd.Timedelta(hours=merge_gap_hours)
    for ts in pd.to_datetime(bad["datetime"]).iloc[1:]:
        if ts - current_end <= max_gap:
            current_end = ts
        else:
            windows.append((current_start, current_end))
            current_start = current_end = ts
    windows.append((current_start, current_end))

    expanded = [
        (start - pd.Timedelta(hours=expand_hours), end + pd.Timedelta(hours=expand_hours))
        for start, end in windows
    ]
    expanded = merge_intervals(expanded)

    rows: list[dict[str, Any]] = []
    for event_id, (start, end) in enumerate(expanded, start=1):
        part = frame[(frame["datetime"] >= start) & (frame["datetime"] <= end)].copy()
        scored = part[np.isfinite(part["Speed (km/s)"]) & np.isfinite(part["predicted_speed"])]
        if scored.empty:
            continue
        peak_idx = scored["Speed (km/s)"].idxmax()
        rows.append(
            {
                "event_id": event_id,
                "event_start": start,
                "event_end": end,
                "peak_time": pd.Timestamp(scored.loc[peak_idx, "datetime"]),
                "peak_observed_speed": float(scored.loc[peak_idx, "Speed (km/s)"]),
                "mean_observed_speed": float(scored["Speed (km/s)"].mean()),
                "median_observed_speed": float(scored["Speed (km/s)"].median()),
                "p90_observed_speed": float(scored["Speed (km/s)"].quantile(0.90)),
                "mean_predicted_speed": float(scored["predicted_speed"].mean()),
                "max_abs_error": float(scored["abs_error"].max()),
                "mean_abs_error": float(scored["abs_error"].mean()),
                "scored_rows": int(len(scored)),
            }
        )
    return pd.DataFrame(rows)


def estimate_launch_times(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in events.to_dict("records"):
        peak_time = pd.Timestamp(row["peak_time"])
        speed_defs = {
            "V_peak": float(row["peak_observed_speed"]),
            "V_mean": float(row["mean_observed_speed"]),
            "V_p90": float(row["p90_observed_speed"]),
        }
        for speed_name, speed in speed_defs.items():
            travel_time_hours = AU_KM / speed / 3600.0 if speed > 0 else float("nan")
            launch_time = peak_time - pd.Timedelta(hours=travel_time_hours)
            rows.append(
                {
                    "event_id": int(row["event_id"]),
                    "speed_definition": speed_name,
                    "speed_km_s": speed,
                    "peak_time": peak_time,
                    "travel_time_hours": travel_time_hours,
                    "launch_time_est": launch_time,
                }
            )
    return pd.DataFrame(rows)


def hmi_sampling_requests(launches: pd.DataFrame) -> pd.DataFrame:
    offsets = [0, -1, 1, -3, 3, -5, 5, -10, 10]
    rows: list[dict[str, Any]] = []
    for row in launches.to_dict("records"):
        launch_time = pd.Timestamp(row["launch_time_est"])
        for offset in offsets:
            rows.append(
                {
                    "event_id": int(row["event_id"]),
                    "speed_definition": row["speed_definition"],
                    "launch_time_est": launch_time,
                    "sample_offset_hours": offset,
                    "requested_hmi_time": launch_time + pd.Timedelta(hours=offset),
                }
            )
    return pd.DataFrame(rows)


def local_hmi_files() -> list[tuple[pd.Timestamp, Path]]:
    files: list[tuple[pd.Timestamp, Path]] = []
    for raw_dir in [ICME_RAW_DIR, LEGACY_RAW_DIR]:
        if not raw_dir.exists():
            continue
        for path in sorted(raw_dir.glob("hmi.m_720s.*_TAI*.magnetogram.fits")):
            if not is_probable_fits(path):
                continue
            ts = fits_time(path)
            if ts is not None:
                files.append((ts, path))
    files.sort(key=lambda item: item[0])
    return files


def match_local_hmi(
    requested_time: pd.Timestamp,
    files: list[tuple[pd.Timestamp, Path]],
    method: str,
    tolerance_minutes: float,
) -> MatchedHMI:
    if not files:
        return MatchedHMI(requested_time, None, None, method, float("nan"), "missing", "no local HMI FITS files found")
    times = np.array([item[0].value for item in files], dtype=np.int64)
    req_value = requested_time.value
    if method == "backward":
        candidates = np.flatnonzero(times <= req_value)
        if len(candidates) == 0:
            return MatchedHMI(requested_time, None, None, method, float("nan"), "missing", "no backward local HMI record")
        idx = int(candidates[-1])
    else:
        idx = int(np.argmin(np.abs(times - req_value)))
    matched_time, path = files[idx]
    offset = (matched_time - requested_time).total_seconds() / 60.0
    if abs(offset) > tolerance_minutes:
        return MatchedHMI(
            requested_time,
            matched_time,
            None,
            method,
            offset,
            "missing",
            f"local file outside {tolerance_minutes:g} minute tolerance",
        )
    return MatchedHMI(requested_time, matched_time, path, method, offset, "matched_local")


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


def choose_download_url(export_request: Any) -> str:
    urls = dataframe_urls(export_request.urls)
    fits_urls = [
        url for url in urls
        if urllib.parse.urlparse(url).path.lower().endswith((".fits", ".fit", ".fits.gz"))
    ]
    if fits_urls:
        return fits_urls[0]
    if urls:
        return urls[0]
    raise RuntimeError("JSOC export completed but no downloadable URL was returned.")


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


def query_hmi_record(client: Any, requested_time: pd.Timestamp, method: str, tolerance_minutes: float) -> pd.Timestamp | None:
    start = requested_time - pd.Timedelta(minutes=tolerance_minutes)
    duration_minutes = int(math.ceil(2 * tolerance_minutes))
    query = f"{HMI_SERIES}[{jsoc_time_token(start)}/{duration_minutes}m]"
    records = client.query(query, key="T_REC,QUALITY")
    if records is None or len(records) == 0 or "T_REC" not in records.columns:
        return None
    times = [parse_jsoc_time(value) for value in records["T_REC"].tolist()]
    times = [ts for ts in times if ts is not None]
    if method == "backward":
        times = [ts for ts in times if ts <= requested_time]
        return max(times) if times else None
    return min(times, key=lambda ts: abs((ts - requested_time).total_seconds())) if times else None


def local_path_for_hmi(url: str, matched_time: pd.Timestamp) -> Path:
    name = Path(urllib.parse.urlparse(url).path).name
    if not name or "." not in name:
        name = f"hmi.m_720s.{matched_time.strftime('%Y%m%d_%H%M%S')}_TAI.magnetogram.fits"
    return ICME_RAW_DIR / name


def try_download_hmi(
    requested_time: pd.Timestamp,
    method: str,
    tolerance_minutes: float,
    timeout: int,
    export_timeout: int,
) -> tuple[MatchedHMI, dict[str, Any]]:
    if importlib.util.find_spec("drms") is None:
        matched = MatchedHMI(requested_time, None, None, method, float("nan"), "missing", "drms is not installed")
        return matched, {"status": "missing_dependency", "error_message": "drms is not installed"}
    email = os.environ.get("JSOC_EMAIL", "").strip()
    if not email:
        matched = MatchedHMI(requested_time, None, None, method, float("nan"), "missing", "JSOC_EMAIL is not set")
        return matched, {"status": "missing_jsoc_email", "error_message": "JSOC_EMAIL is required for HMI download"}

    import drms

    client = drms.Client(email=email)
    try:
        matched_time = query_hmi_record(client, requested_time, method, tolerance_minutes)
        if matched_time is None:
            matched = MatchedHMI(requested_time, None, None, method, float("nan"), "missing", "no hmi.M_720s record in tolerance window")
            return matched, {"status": "no_record", "error_message": matched.note}
        offset = (matched_time - requested_time).total_seconds() / 60.0
        export_query = f"{HMI_SERIES}[{jsoc_time_token(matched_time)}]{{{HMI_SEGMENT}}}"
        export_request = client.export(export_query, method="url", protocol="fits")
        export_request.wait(timeout=export_timeout)
        url = choose_download_url(export_request)
        path = local_path_for_hmi(url, matched_time)
        if path.exists() and is_probable_fits(path):
            status = "skipped_existing"
        else:
            download_url(url, path, timeout)
            status = "downloaded"
        matched = MatchedHMI(requested_time, matched_time, path, method, offset, "matched_downloaded")
        return matched, {
            "status": status,
            "matched_hmi_time": matched_time,
            "time_offset_minutes": offset,
            "jsoc_query": export_query,
            "url": url,
            "local_path": str(path),
            "error_message": "",
        }
    except Exception as exc:
        matched = MatchedHMI(requested_time, None, None, method, float("nan"), "missing", f"{type(exc).__name__}: {exc}")
        return matched, {"status": "download_failed", "error_message": matched.note}


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


def region_metrics(prefix: str, values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    selected = values[mask & np.isfinite(values)]
    if len(selected) == 0:
        return {
            f"unsigned_flux_proxy_{prefix}": float("nan"),
            f"mean_abs_B_{prefix}": float("nan"),
            f"max_abs_B_{prefix}": float("nan"),
            f"strong_field_area_{prefix}": 0,
            f"strong_field_unsigned_flux_{prefix}": float("nan"),
        }
    abs_selected = np.abs(selected)
    strong = abs_selected > 100.0
    return {
        f"unsigned_flux_proxy_{prefix}": float(np.sum(abs_selected)),
        f"mean_abs_B_{prefix}": float(np.mean(abs_selected)),
        f"max_abs_B_{prefix}": float(np.max(abs_selected)),
        f"strong_field_area_{prefix}": int(np.sum(strong)),
        f"strong_field_unsigned_flux_{prefix}": float(np.sum(abs_selected[strong])),
    }


def largest_component(mask: np.ndarray) -> np.ndarray | None:
    if not mask.any():
        return None
    if importlib.util.find_spec("scipy") is not None:
        from scipy import ndimage

        labels, n_labels = ndimage.label(mask)
        if n_labels == 0:
            return None
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        largest = int(np.argmax(counts))
        return labels == largest
    return None


def roi_mask(br: np.ndarray, lon: np.ndarray, lat: np.ndarray, good_mu: np.ndarray) -> tuple[np.ndarray, str]:
    candidate = good_mu & (np.abs(br) > 300.0) & (np.abs(lon) < 70.0) & (np.abs(lat) < 50.0)
    component = largest_component(candidate)
    if component is not None and component.any():
        return component, "largest_connected_component_abs_B_gt_300G"
    central = good_mu & (np.abs(lon) < 70.0) & (np.abs(lat) < 50.0) & np.isfinite(br)
    if not central.any():
        return np.zeros(br.shape, dtype=bool), "no_central_valid_pixels"
    central_abs = np.abs(br[central])
    threshold = float(np.nanpercentile(central_abs, 99.0))
    top = central & (np.abs(br) >= threshold)
    if not top.any():
        return np.zeros(br.shape, dtype=bool), "no_top_1_percent_pixels"
    yy, xx = np.where(top)
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    box = np.zeros(br.shape, dtype=bool)
    box[y0:y1, x0:x1] = central[y0:y1, x0:x1]
    return box, "bounding_box_around_top_1_percent_abs_B"


def compute_flux_metrics(path: Path) -> dict[str, Any]:
    data, header = open_fits_array(path)
    lon, lat, mu, valid_disk = approximate_hmi_disk_geometry(data.shape, header)
    good_mu = valid_disk & (mu >= 0.2) & np.isfinite(data)
    br = np.full_like(data, np.nan, dtype=np.float64)
    br[good_mu] = data[good_mu] / mu[good_mu]

    full_disk = good_mu
    central_disk = good_mu & (np.abs(lon) < 60.0) & (np.abs(lat) < 60.0)
    roi, roi_method = roi_mask(br, lon, lat, good_mu)

    row: dict[str, Any] = {
        "local_path": str(path),
        "matched_hmi_time_from_filename": fits_time(path),
        "flux_units": "proxy_sum_G_pixels",
        "field_mode": "Br_approx_equals_B_los_over_mu_mu_ge_0p2",
        "roi_method": roi_method,
    }
    row.update(region_metrics("full_disk", br, full_disk))
    row.update(region_metrics("central_disk", br, central_disk))

    roi_values = br[roi & np.isfinite(br)]
    if len(roi_values) == 0:
        roi_y = roi_x = np.array([], dtype=int)
        row.update(
            {
                "roi_unsigned_flux_proxy": float("nan"),
                "roi_area_pixels": 0,
                "roi_mean_abs_B": float("nan"),
                "roi_max_abs_B": float("nan"),
                "roi_centroid_x": float("nan"),
                "roi_centroid_y": float("nan"),
                "roi_centroid_lon_approx": float("nan"),
                "roi_centroid_lat_approx": float("nan"),
                "roi_positive_flux_proxy": float("nan"),
                "roi_negative_flux_proxy": float("nan"),
                "roi_flux_imbalance": float("nan"),
            }
        )
    else:
        roi_y, roi_x = np.where(roi & np.isfinite(br))
        pos = float(np.sum(roi_values[roi_values > 0]))
        neg = float(np.sum(roi_values[roi_values < 0]))
        denom = abs(pos) + abs(neg)
        weights = np.abs(roi_values)
        centroid_x = float(np.average(roi_x, weights=weights)) if np.sum(weights) > 0 else float(np.mean(roi_x))
        centroid_y = float(np.average(roi_y, weights=weights)) if np.sum(weights) > 0 else float(np.mean(roi_y))
        iy = np.clip(np.rint(centroid_y).astype(int), 0, lon.shape[0] - 1)
        ix = np.clip(np.rint(centroid_x).astype(int), 0, lon.shape[1] - 1)
        row.update(
            {
                "roi_unsigned_flux_proxy": float(np.sum(np.abs(roi_values))),
                "roi_area_pixels": int(len(roi_values)),
                "roi_mean_abs_B": float(np.mean(np.abs(roi_values))),
                "roi_max_abs_B": float(np.max(np.abs(roi_values))),
                "roi_centroid_x": centroid_x,
                "roi_centroid_y": centroid_y,
                "roi_centroid_lon_approx": float(lon[iy, ix]),
                "roi_centroid_lat_approx": float(lat[iy, ix]),
                "roi_positive_flux_proxy": pos,
                "roi_negative_flux_proxy": neg,
                "roi_flux_imbalance": float(abs(pos + neg) / denom) if denom > 0 else float("nan"),
            }
        )
    return row


def build_hmi_matches(
    requests: pd.DataFrame,
    method: str,
    tolerance_minutes: float,
    download: bool,
    timeout: int,
    export_timeout: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    local_files = local_hmi_files()
    match_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    cache: dict[pd.Timestamp, MatchedHMI] = {}
    for request in requests.to_dict("records"):
        requested = pd.Timestamp(request["requested_hmi_time"])
        matched = cache.get(requested)
        report: dict[str, Any] = {}
        if matched is None:
            matched = match_local_hmi(requested, local_files, method, tolerance_minutes)
            if matched.status == "missing" and download:
                matched, report = try_download_hmi(requested, method, tolerance_minutes, timeout, export_timeout)
                local_files = local_hmi_files()
            cache[requested] = matched
        row = {
            **request,
            "matched_hmi_time": matched.matched_time,
            "time_offset_minutes": matched.time_offset_minutes,
            "match_method": matched.match_method,
            "match_status": matched.status,
            "local_path": str(matched.local_path) if matched.local_path else "",
            "match_note": matched.note,
        }
        match_rows.append(row)
        report_rows.append(
            {
                **request,
                "matched_hmi_time": matched.matched_time,
                "time_offset_minutes": matched.time_offset_minutes,
                "status": report.get("status", matched.status),
                "local_path": report.get("local_path", str(matched.local_path) if matched.local_path else ""),
                "jsoc_query": report.get("jsoc_query", ""),
                "url": report.get("url", ""),
                "error_message": report.get("error_message", matched.note),
            }
        )
    return pd.DataFrame(match_rows), pd.DataFrame(report_rows)


def compute_flux_timeseries(matches: pd.DataFrame) -> pd.DataFrame:
    if importlib.util.find_spec("astropy") is None:
        rows = matches.to_dict("records")
        for row in rows:
            row["extraction_error"] = "astropy is not installed"
        return pd.DataFrame(rows)
    flux_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for item in matches.to_dict("records"):
        path_text = str(item.get("local_path", ""))
        row = dict(item)
        if not path_text:
            row["extraction_error"] = "missing local HMI file"
            rows.append(row)
            continue
        if path_text not in flux_cache:
            try:
                flux_cache[path_text] = compute_flux_metrics(Path(path_text))
            except Exception as exc:
                flux_cache[path_text] = {"extraction_error": f"{type(exc).__name__}: {exc}"}
        row.update(flux_cache[path_text])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_event(flux: pd.DataFrame) -> pd.DataFrame:
    if flux.empty or "roi_unsigned_flux_proxy" not in flux.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    launch_rows = flux[flux["sample_offset_hours"].eq(0)].copy()
    for event_id, group in flux.groupby("event_id", sort=True):
        finite = group[np.isfinite(pd.to_numeric(group.get("roi_unsigned_flux_proxy"), errors="coerce"))].copy()
        launch = launch_rows[launch_rows["event_id"].eq(event_id)].copy()
        before = finite[finite["sample_offset_hours"] < 0]
        at = finite[finite["sample_offset_hours"] == 0]
        before_max = float(before["roi_unsigned_flux_proxy"].max()) if not before.empty else float("nan")
        at_max = float(at["roi_unsigned_flux_proxy"].max()) if not at.empty else float("nan")
        rows.append(
            {
                "event_id": int(event_id),
                "hmi_samples": int(len(group)),
                "matched_hmi_samples": int(group["local_path"].astype(str).ne("").sum()),
                "roi_unsigned_flux_proxy_max": float(finite["roi_unsigned_flux_proxy"].max()) if not finite.empty else float("nan"),
                "full_disk_unsigned_flux_proxy_max": float(finite["unsigned_flux_proxy_full_disk"].max()) if "unsigned_flux_proxy_full_disk" in finite else float("nan"),
                "central_disk_unsigned_flux_proxy_max": float(finite["unsigned_flux_proxy_central_disk"].max()) if "unsigned_flux_proxy_central_disk" in finite else float("nan"),
                "roi_unsigned_flux_proxy_at_launch_max": at_max,
                "roi_unsigned_flux_proxy_before_launch_max": before_max,
                "flux_increased_before_launch": bool(before_max > at_max) if np.isfinite(before_max) and np.isfinite(at_max) else False,
                "launch_rows_with_flux": int(len(launch[launch["local_path"].astype(str).ne("")])),
            }
        )
    return pd.DataFrame(rows)


def print_summary(events: pd.DataFrame, launches: pd.DataFrame, flux: pd.DataFrame, summary: pd.DataFrame) -> None:
    if events.empty:
        print("No ICME-like bad intervals found.")
        return
    for event in events.to_dict("records"):
        event_id = int(event["event_id"])
        print(f"\nEvent {event_id}")
        print(
            "event_start={event_start} event_end={event_end} peak_time={peak_time} "
            "peak_observed_speed={peak_observed_speed:.1f} mean_observed_speed={mean_observed_speed:.1f}".format(**event)
        )
        event_launches = launches[launches["event_id"].eq(event_id)]
        for launch in event_launches.to_dict("records"):
            print(
                f"  {launch['speed_definition']}: launch_time_est={pd.Timestamp(launch['launch_time_est'])} "
                f"travel_time_hours={launch['travel_time_hours']:.2f}"
            )
        cols = [
            "speed_definition",
            "sample_offset_hours",
            "requested_hmi_time",
            "matched_hmi_time",
            "time_offset_minutes",
            "unsigned_flux_proxy_full_disk",
            "unsigned_flux_proxy_central_disk",
            "roi_unsigned_flux_proxy",
            "roi_area_pixels",
        ]
        event_flux = flux[flux["event_id"].eq(event_id)].copy()
        available_cols = [col for col in cols if col in event_flux.columns]
        if available_cols:
            print(event_flux[available_cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
        row = summary[summary["event_id"].eq(event_id)]
        if not row.empty:
            print(f"  flux_increased_before_launch={bool(row.iloc[0]['flux_increased_before_launch'])}")


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = HERE / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ICME_RAW_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_prediction_frame(args.year)
    events = identify_events(
        frame,
        args.speed_threshold,
        args.abs_error_threshold,
        args.merge_gap_hours,
        args.expand_hours,
    )
    launches = estimate_launch_times(events) if not events.empty else pd.DataFrame()
    requests = hmi_sampling_requests(launches) if not launches.empty else pd.DataFrame()
    matches, download_report = (
        build_hmi_matches(
            requests,
            args.hmi_match,
            args.hmi_tolerance_minutes,
            args.download,
            args.timeout,
            args.export_timeout,
        )
        if not requests.empty
        else (pd.DataFrame(), pd.DataFrame())
    )
    flux = compute_flux_timeseries(matches) if not matches.empty else pd.DataFrame()
    summary = summarize_by_event(flux)

    events.to_csv(out_dir / "icme_event_windows.csv", index=False)
    launches.to_csv(out_dir / "estimated_launch_times.csv", index=False)
    flux.to_csv(out_dir / "hmi_flux_timeseries.csv", index=False)
    summary.to_csv(out_dir / "hmi_flux_summary_by_event.csv", index=False)
    download_report.to_csv(out_dir / "download_report.csv", index=False)

    print_summary(events, launches, flux, summary)
    print(f"\nSaved outputs to {out_dir}")
    if not args.download:
        print("Download mode was not enabled. Re-run with --download and JSOC_EMAIL set to fetch missing HMI records.")


if __name__ == "__main__":
    main()
