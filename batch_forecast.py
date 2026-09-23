"""Auditable daily replay. No partial or failed batch replaces a submission."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

LOCAL_TIMEZONE = os.getenv("SCADA_TIMEZONE", "Asia/Almaty")
TURBINES = ("turbine_1", "turbine_2")
TEST_START = date(2026, 2, 1)
TEST_END = date(2026, 2, 28)
COLUMNS = ("turbine", "timestamp", "wind_speed_ms", "temperature_c", "predicted_power")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "submission_february_2026.csv"


def _date(value: str | date) -> date:
    if isinstance(value, datetime):
        raise ValueError("Use a calendar date, not a datetime.")
    parsed = value if isinstance(value, date) else date.fromisoformat(value)
    if not TEST_START <= parsed <= TEST_END:
        raise ValueError("The submission period must be within 2026-02-01..2026-02-28.")
    return parsed


def _aware_time(value: str, label: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a UTC offset.")
    return parsed.astimezone(timezone.utc)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not boolean.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _validate_result(result: dict, turbine: str, start: datetime, issue: datetime) -> list[dict]:
    if not isinstance(result, dict) or result.get("status") != "success":
        message = result.get("message", "forecast failed") if isinstance(result, dict) else "invalid result"
        raise ValueError(f"{turbine} {start.date()}: {message}")
    if not result.get("forecast_id") or not isinstance(result.get("weather"), dict):
        raise ValueError("Forecast is missing its identifier or weather provenance.")
    if _aware_time(result.get("issue_time"), "issue_time") != issue.astimezone(timezone.utc):
        raise ValueError("The returned issue_time differs from the requested decision time.")
    samples = result.get("forecast_sample", [])
    if not isinstance(samples, list) or len(samples) != 24:
        raise ValueError(f"{turbine} {start.date()}: expected exactly 24 hourly samples.")
    expected = {start.astimezone(timezone.utc) + timedelta(hours=i) for i in range(24)}
    rows, seen = [], set()
    for sample in samples:
        stamp = _aware_time(sample["time"], "forecast_sample.time")
        if stamp not in expected or stamp in seen:
            raise ValueError("Forecast hours are duplicated or differ from the requested period.")
        seen.add(stamp)
        wind = _number(sample["wind_speed"], "wind_speed")
        temperature = _number(sample["temperature"], "temperature")
        power = _number(sample["predicted_power"], "predicted_power")
        if wind < 0 or not 0 <= power <= 1:
            raise ValueError("Wind speed must be nonnegative and normalized power must be in [0,1].")
        rows.append({
            "turbine": turbine,
            "timestamp": stamp.astimezone(ZoneInfo(LOCAL_TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S"),
            "wind_speed_ms": wind,
            "temperature_c": temperature,
            "predicted_power": power,
        })
    if seen != expected:
        raise ValueError("Forecast has missing hours.")
    return sorted(rows, key=lambda row: row["timestamp"])


def _stage_bytes(path: Path, payload: bytes) -> Path:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return temporary


def _publish(output: Path, csv_bytes: bytes, manifest: dict) -> Path:
    """Stage both files; CSV is the commit point, hashes detect interrupted pairs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(".provenance.json")
    old_manifest = manifest_path.read_bytes() if manifest_path.exists() else None
    csv_temp = manifest_temp = None
    manifest_published = False
    try:
        csv_temp = _stage_bytes(output, csv_bytes)
        manifest_temp = _stage_bytes(
            manifest_path,
            (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"),
        )
        os.replace(manifest_temp, manifest_path)
        manifest_published = True
        os.replace(csv_temp, output)
    except BaseException:
        if manifest_published:
            if old_manifest is None:
                manifest_path.unlink(missing_ok=True)
            else:
                restore = _stage_bytes(manifest_path, old_manifest)
                try:
                    os.replace(restore, manifest_path)
                finally:
                    restore.unlink(missing_ok=True)
        raise
    finally:
        for temporary in (csv_temp, manifest_temp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return manifest_path


def run_simulation(
    start: str | date = TEST_START,
    end: str | date = TEST_END,
    output: str | Path = DEFAULT_OUTPUT,
    refresh: bool = False,
    forecast_fn=None,
) -> dict:
    """Replay daily decisions at 23:00 local for the following 24 target hours."""
    first, last = _date(start), _date(end)
    if first > last:
        raise ValueError("start must be on or before end.")
    if forecast_fn is None:
        from forecast_engine import generate_agent_forecast
        forecast_fn = generate_agent_forecast
    output = Path(output).resolve()
    local = ZoneInfo(LOCAL_TIMEZONE)
    all_rows, runs = [], []
    for turbine in TURBINES:
        current = first
        while current <= last:
            target_start = datetime.combine(current, time.min, local)
            issue = target_start - timedelta(hours=1)
            result = forecast_fn(
                turbine_id=turbine,
                forecast_date=current.isoformat(),
                horizon_hours=24,
                issue_time=issue.isoformat(),
                refresh=refresh,
            )
            all_rows.extend(_validate_result(result, turbine, target_start, issue))
            metadata = {key: value for key, value in result.items() if key != "forecast_sample"}
            runs.append({"turbine": turbine, "target_date": current.isoformat(), **metadata})
            print(f"Validated {turbine} {current}: 24 hours; decision {issue.isoformat()}", flush=True)
            current += timedelta(days=1)
    expected_count = ((last - first).days + 1) * 24 * len(TURBINES)
    identities = {(row["turbine"], row["timestamp"]) for row in all_rows}
    if len(all_rows) != expected_count or len(identities) != expected_count:
        raise ValueError("The completed batch has missing or duplicate turbine/hour rows.")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(all_rows)
    csv_bytes = stream.getvalue().encode("utf-8")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "csv_file": output.name,
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "row_count": len(all_rows),
        "turbines": list(TURBINES),
        "target_start": first.isoformat(),
        "target_end": last.isoformat(),
        "horizon_hours": 24,
        "decision_policy": f"23:00 {LOCAL_TIMEZONE} on the day before each target date",
        "timestamp_timezone": LOCAL_TIMEZONE,
        "timestamp_csv_format": "YYYY-MM-DD HH:MM:SS; local time, offset omitted for compatibility",
        "timezone_assumption": f"Organizers did not specify the SCADA timezone; {LOCAL_TIMEZONE} is an explicit unconfirmed assumption.",
        "power_unit": "normalized; same dimensionless scale as source CSV; no MW/MWh conversion",
        "validation_scope": "Observed-weather validation is not a historical forecast accuracy measurement. February truth is unavailable.",
        "runs": runs,
    }
    manifest_path = _publish(output, csv_bytes, manifest)
    print(f"Saved {len(all_rows)} validated rows: {output}", flush=True)
    print(f"Provenance and checksum: {manifest_path}", flush=True)
    return {"output": str(output), "manifest": str(manifest_path), "row_count": len(all_rows), "sha256": manifest["csv_sha256"]}


def run_february_simulation():
    return run_simulation()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=TEST_START.isoformat(), help="First target date (within February 2026)")
    parser.add_argument("--end", default=TEST_END.isoformat(), help="Last target date, inclusive")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV path; provenance JSON is written beside it")
    parser.add_argument("--refresh", action="store_true", help="Refresh external weather inputs and recalculate")
    args = parser.parse_args(argv)
    try:
        run_simulation(args.start, args.end, args.output, args.refresh)
    except Exception as exc:
        parser.exit(1, f"Batch failed; the previous submission CSV was not replaced: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
