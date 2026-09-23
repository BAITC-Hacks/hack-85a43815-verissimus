"""Operational NOAA GFS forecasts, selected as they were available at issue time.

The public NODD S3 archive preserves original GFS runs. We use only the 0.25
degree forecast product, never analyses/reanalysis/stitched historical weather.
HTTP Last-Modified is retained as evidence of archive object availability; it
is not presented as an independent guarantee of the first dissemination time.

References:
https://registry.opendata.aws/noaa-gfs-bdp-pds/
https://www.nco.ncep.noaa.gov/pmb/products/gfs/
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import uuid

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "weather"
CACHE_VERSION = 1
MAX_WORKERS = 4
MAX_FIELD_BYTES = 16 * 1024 * 1024
_LOCKS = [threading.Lock() for _ in range(64)]
_ECCODES_LOCK = threading.Lock()
_THREAD_LOCAL = threading.local()
_FIELDS = {
    "temperature": ("TMP", "2 m above ground", 2, "t"),
    "u100": ("UGRD", "100 m above ground", 100, "u"),
    "v100": ("VGRD", "100 m above ground", 100, "v"),
}


class WeatherArchiveError(RuntimeError):
    """A complete, verifiably available forecast could not be obtained."""


def _utc(value, name: str) -> pd.Timestamp:
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise WeatherArchiveError(f"{name} must be a timezone-aware timestamp")
    return value.tz_convert("UTC")


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _session() -> requests.Session:
    if not hasattr(_THREAD_LOCAL, "session"):
        session = requests.Session()
        retry = Retry(total=3, connect=3, read=3, backoff_factor=0.5,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=("GET",))
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.headers.update({"User-Agent": "EnergyAI-historical-forecast/1.0"})
        _THREAD_LOCAL.session = session
    return _THREAD_LOCAL.session


def _download(url: str, issue_time: pd.Timestamp, byte_range=None):
    """Stream a bounded response, refusing an ignored Range before reading body."""
    headers = {"Accept-Encoding": "identity"}
    limit = 512 * 1024
    if byte_range is not None:
        start, end = byte_range
        limit = end - start + 1
        if not 0 < limit <= MAX_FIELD_BYTES:
            raise WeatherArchiveError("Invalid or unexpectedly large GRIB field range")
        headers["Range"] = f"bytes={start}-{end}"
    try:
        with _session().get(url, headers=headers, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            if byte_range is not None:
                expected = f"bytes {byte_range[0]}-{byte_range[1]}/"
                if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                    raise WeatherArchiveError("Archive server did not honor the exact byte range; full GRIB download refused")
            elif response.status_code != 200:
                raise WeatherArchiveError("Invalid archive inventory response")
            raw_modified = response.headers.get("Last-Modified")
            if not raw_modified:
                raise WeatherArchiveError("Archive response lacks Last-Modified availability evidence")
            available = _utc(raw_modified, "Last-Modified")
            if available > issue_time:
                raise WeatherArchiveError(f"Archive object was modified at {available.isoformat()}, after issue_time {issue_time.isoformat()}")
            if int(response.headers.get("Content-Length", "0")) > limit:
                raise WeatherArchiveError("Archive response exceeds expected size")
            chunks, size = [], 0
            for chunk in response.iter_content(chunk_size=65536):
                size += len(chunk)
                if size > limit:
                    raise WeatherArchiveError("Archive response exceeds bounded download size")
                chunks.append(chunk)
            payload = b"".join(chunks)
            if byte_range is not None and len(payload) != limit:
                raise WeatherArchiveError("Truncated GRIB range response")
            return payload, {"url": url, "last_modified": available.isoformat(),
                             "etag": response.headers.get("ETag", ""),
                             "sha256": _sha(payload), "byte_count": len(payload)}
    except (requests.RequestException, ValueError) as exc:
        raise WeatherArchiveError(f"NOAA archive request failed for {url}: {exc}") from exc


def _field_ranges(index_text: str, run: pd.Timestamp, lead: int):
    lines = [line.split(":") for line in index_text.splitlines() if line.strip()]
    try:
        offsets = [int(parts[1]) for parts in lines]
    except (ValueError, IndexError) as exc:
        raise WeatherArchiveError("Malformed NOAA GRIB inventory") from exc
    if offsets != sorted(set(offsets)):
        raise WeatherArchiveError("Non-increasing NOAA GRIB inventory offsets")
    ranges = {}
    for name, (parameter, level, _, _) in _FIELDS.items():
        matches = [i for i, parts in enumerate(lines)
                   if len(parts) >= 6 and parts[3:5] == [parameter, level]]
        if len(matches) != 1 or matches[0] + 1 >= len(lines):
            raise WeatherArchiveError(f"Missing or ambiguous {parameter} at {level} in NOAA inventory")
        i = matches[0]
        if lines[i][2] != f"d={run.strftime('%Y%m%d%H')}" or lines[i][5] != f"{lead} hour fcst":
            raise WeatherArchiveError("Inventory run/lead does not match requested forecast")
        ranges[name] = (offsets[i], offsets[i + 1] - 1)
    return ranges


def _decode(payload: bytes, name: str, run: pd.Timestamp, lead: int, cell):
    # ecCodes native definitions/parser initialization is not thread-safe in the
    # Windows wheel. Network I/O stays concurrent; native decoding is serialized.
    with _ECCODES_LOCK:
        return _decode_unlocked(payload, name, run, lead, cell)


def _decode_unlocked(payload: bytes, name: str, run: pd.Timestamp, lead: int, cell):
    try:
        import eccodes
    except ImportError as exc:
        raise WeatherArchiveError("Install requirements.txt: eccodes is required to decode NOAA GRIB2") from exc
    handle = None
    try:
        handle = eccodes.codes_new_from_message(payload)
        get = lambda key: eccodes.codes_get(handle, key)
        _, _, height, short_name = _FIELDS[name]
        actual_short = str(get("shortName"))
        if actual_short not in {short_name, f"{height}{short_name}"}:
            raise WeatherArchiveError(f"Unexpected GRIB parameter: {actual_short}")
        if (get("typeOfLevel") != "heightAboveGround" or int(get("level")) != height
                or str(get("stepType")) != "instant"):
            raise WeatherArchiveError("Unexpected GRIB field level or time aggregation")
        if (int(get("dataDate")) != int(run.strftime("%Y%m%d"))
                or int(get("dataTime")) != run.hour * 100
                or int(get("forecastTime")) != lead or int(get("indicatorOfUnitOfTimeRange")) != 1):
            raise WeatherArchiveError("GRIB payload run/lead does not match inventory")
        expected_valid = run + pd.Timedelta(hours=lead)
        if (int(get("validityDate")) != int(expected_valid.strftime("%Y%m%d"))
                or int(get("validityTime")) != expected_valid.hour * 100):
            raise WeatherArchiveError("GRIB validity timestamp mismatch")
        if get("gridType") != "regular_ll" or not math.isclose(float(get("iDirectionIncrementInDegrees")), .25):
            raise WeatherArchiveError("Expected the GFS regular 0.25 degree grid")
        expected_unit = "K" if name == "temperature" else "m s**-1"
        if get("units") != expected_unit:
            raise WeatherArchiveError("Unexpected GRIB units")
        nearest = eccodes.codes_grib_find_nearest(handle, cell[0], cell[1], is_lsm=False, npoints=1)[0]
        if not math.isclose(float(nearest["lat"]), cell[0], abs_tol=1e-7) or not math.isclose(float(nearest["lon"]) % 360, cell[1] % 360, abs_tol=1e-7):
            raise WeatherArchiveError("Expected grid cell absent from GRIB")
        value = float(nearest["value"])
        if not math.isfinite(value) or abs(value) > 1000:
            raise WeatherArchiveError("Missing/nonfinite GRIB value")
        return value
    except WeatherArchiveError:
        raise
    except Exception as exc:
        raise WeatherArchiveError(f"Cannot decode NOAA {name} field: {exc}") from exc
    finally:
        if handle is not None:
            eccodes.codes_release(handle)


def _check_record(record: dict, run: pd.Timestamp, lead: int, cell, issue: pd.Timestamp):
    if (record["cache_version"] != CACHE_VERSION or record["run_time"] != run.isoformat()
            or record["lead_hours"] != lead or record["grid_cell"] != list(cell)):
        raise WeatherArchiveError("Cached weather input identity mismatch")
    if len(record["sources"]) != 4 or {s["field"] for s in record["sources"]} != {"inventory", *_FIELDS}:
        raise WeatherArchiveError("Cached weather lacks complete source provenance")
    for source in record["sources"]:
        available = _utc(source["last_modified"], "cached availability")
        if not run <= available <= issue:
            raise WeatherArchiveError("Cached archive object was not available at issue_time")
        if not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]):
            raise WeatherArchiveError("Cached source fingerprint is invalid")
    for key in ("wind_speed", "temperature"):
        if not math.isfinite(record[key]):
            raise WeatherArchiveError("Cached weather contains nonfinite values")
    if not 0 <= record["wind_speed"] <= 150 or not -150 <= record["temperature"] <= 80:
        raise WeatherArchiveError("Weather values outside supported physical bounds")


def _hour(run: pd.Timestamp, lead: int, cell, issue: pd.Timestamp, refresh: bool):
    key = f"gfs_{run.strftime('%Y%m%d%H')}_f{lead:03d}_{cell[0]:.2f}_{cell[1]:.2f}"
    path = CACHE_DIR / f"{key}.json"
    with _LOCKS[hash(key) % len(_LOCKS)]:
        if path.exists() and not refresh:
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
                record = envelope["record"]
                if _sha(_canonical(record)) != envelope["record_sha256"]:
                    raise WeatherArchiveError("Weather cache checksum mismatch; use refresh=True")
                _check_record(record, run, lead, cell, issue)
                return record
            except (KeyError, TypeError, ValueError, OSError) as exc:
                raise WeatherArchiveError("Weather cache is corrupt; use refresh=True") from exc
        url = f"{BASE_URL}/gfs.{run.strftime('%Y%m%d')}/{run.hour:02d}/atmos/gfs.t{run.hour:02d}z.pgrb2.0p25.f{lead:03d}"
        index_payload, index_source = _download(url + ".idx", issue)
        ranges = _field_ranges(index_payload.decode("ascii"), run, lead)
        sources = [{**index_source, "field": "inventory"}]
        values = {}
        for name, byte_range in ranges.items():
            payload, source = _download(url, issue, byte_range)
            values[name] = _decode(payload, name, run, lead, cell)
            sources.append({**source, "field": name, "byte_range": list(byte_range)})
        if len({s["etag"] for s in sources[1:]}) != 1 or len({s["last_modified"] for s in sources[1:]}) != 1:
            raise WeatherArchiveError("GRIB archive object changed during download")
        record = {"cache_version": CACHE_VERSION, "run_time": run.isoformat(),
                  "lead_hours": lead, "grid_cell": list(cell), "sources": sources,
                  "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
                  "temperature": values["temperature"] - 273.15,
                  "wind_speed": math.hypot(values["u100"], values["v100"])}
        _check_record(record, run, lead, cell, issue)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps({"record": record, "record_sha256": _sha(_canonical(record))},
                                             indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return record


def fetch_weather(lat: float, lon: float, forecast_index: pd.DatetimeIndex,
                  issue_time: pd.Timestamp, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Return complete hourly forecasts and their archive evidence.

    All weather hours belong to one run, conservatively selected at least six
    hours before the decision. Every object's actual S3 Last-Modified must also
    precede the decision. We intentionally fail closed if that evidence is absent.
    The grid-cell cache is shared by nearby turbines; no measured weather fallback.
    """
    if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 360:
        raise WeatherArchiveError("Invalid weather coordinates")
    issue = _utc(issue_time, "issue_time")
    index = pd.DatetimeIndex(forecast_index)
    if index.tz is None or index.empty or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise WeatherArchiveError("forecast_index must be nonempty, timezone-aware, sorted and unique")
    index = index.tz_convert("UTC")
    if not index.equals(index.floor("h")) or (len(index) > 1 and not (index[1:] - index[:-1] == pd.Timedelta(hours=1)).all()):
        raise WeatherArchiveError("forecast_index must contain every consecutive hour")
    if index[0] <= issue:
        raise WeatherArchiveError("Every forecast hour must be after issue_time")
    run = (issue - pd.Timedelta(hours=6)).floor("6h")
    leads = [int((valid - run) / pd.Timedelta(hours=1)) for valid in index]
    if min(leads) < 1 or max(leads) > 120:
        raise WeatherArchiveError("Supported GFS hourly forecast leads are 1..120 hours")
    # Regular lat/lon grid. Sampling this exact cell lets close turbines reuse
    # downloaded fields and makes the actual spatial resolution explicit.
    cell = (round(float(lat) * 4) / 4, (round((float(lon) % 360) * 4) / 4) % 360)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        records = list(executor.map(lambda lead: _hour(run, lead, cell, issue, refresh), leads))
    frame = pd.DataFrame([{k: r[k] for k in ("wind_speed", "temperature")} for r in records], index=index)
    frame.index.name = "time"
    if len(frame) != len(index) or not np.isfinite(frame.to_numpy()).all():
        raise WeatherArchiveError("NOAA did not return the complete requested weather period")
    sources = [{"valid_time": valid.isoformat(), "lead_hours": lead, **source}
               for valid, lead, record in zip(index, leads, records) for source in record["sources"]]
    available = max(_utc(s["last_modified"], "source availability") for s in sources)
    # Retrieval time is intentionally excluded: refreshing an identical original
    # run should preserve the input hash and not trigger a spurious model revision.
    digest_data = {"run_time": run.isoformat(), "grid_cell": cell,
                   "rows": [{"valid_time": valid.isoformat(), "wind_speed": r["wind_speed"],
                             "temperature": r["temperature"], "sources": r["sources"]}
                            for valid, r in zip(index, records)]}
    metadata = {"provider": "NOAA NODD public S3 archive", "model": "GFS 0.25 degree operational forecast",
                "run_time": run.isoformat(), "issue_time": issue.isoformat(),
                "available_at": available.isoformat(),
                "availability_evidence": "Maximum HTTP Last-Modified of original GRIB objects and inventories; archive evidence, not an independent first-publication log",
                "selection_policy": "Single cycle at least six hours before issue_time; every source Last-Modified <= issue_time",
                "wind_height_m": 100, "temperature_height_m": 2,
                "requested_coordinates": {"latitude": lat, "longitude": lon},
                "grid_cell": {"latitude": cell[0], "longitude": cell[1]},
                "spatial_method": "nearest point on the regular 0.25 degree GFS grid",
                "units": {"wind_speed": "m/s", "temperature": "degC"},
                "sources": sources, "input_sha256": _sha(_canonical(digest_data)),
                "cache_version": CACHE_VERSION, "cache_policy": "Extracted grid-cell values and source hashes; refresh bypasses cache",
                "retrieved_at": max(r["retrieved_at"] for r in records)}
    return frame, metadata
