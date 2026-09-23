"""Verify the saved February submission and provenance without network or API keys."""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["turbine", "timestamp", "wind_speed_ms", "temperature_c", "predicted_power"]
TURBINES = ("turbine_1", "turbine_2")
POWER_UNITS = {"normalized", "normalized; same dimensionless scale as source CSV; no MW/MWh conversion"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def instant(value):
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    require(stamp.utcoffset() is not None, "Provenance timestamps must include a timezone")
    return stamp.astimezone(timezone.utc)


def is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value.lower()))


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def verify_submission(path, *, require_full_month=True):
    path = Path(path)
    raw = path.read_bytes()
    manifest = json.loads(path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(raw).hexdigest()
    require(manifest.get("csv_sha256") == digest, "CSV SHA-256 differs from provenance")
    require(manifest.get("schema_version") == 1, "Unknown provenance schema")
    start, end = date.fromisoformat(manifest["target_start"]), date.fromisoformat(manifest["target_end"])
    require(date(2026, 2, 1) <= start <= end <= date(2026, 2, 28), "Target range is outside February 2026")
    if require_full_month:
        require((start, end) == (date(2026, 2, 1), date(2026, 2, 28)), "Submission does not cover the full month")
    require(manifest.get("horizon_hours") == 24, "Daily submission must use a 24-hour horizon")
    require(set(manifest.get("turbines", [])) == set(TURBINES), "Both turbine identifiers are required")
    require(manifest.get("power_unit") in POWER_UNITS, "Normalized power unit is required")
    local_zone = ZoneInfo(manifest["timestamp_timezone"])
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    expected_hours = {
        (turbine, datetime.combine(day, datetime.min.time()) + timedelta(hours=hour))
        for turbine in TURBINES for day in days for hour in range(24)
    }
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    require(reader.fieldnames == COLUMNS, "Unexpected CSV columns")
    seen = set()
    for row in reader:
        require(None not in row and all(row.get(column) is not None for column in COLUMNS), "Malformed CSV row")
        stamp = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
        key = (row["turbine"], stamp)
        require(key in expected_hours and key not in seen, "Unexpected or duplicated turbine/hour")
        seen.add(key)
        numbers = {column: float(row[column]) for column in COLUMNS[2:]}
        require(all(math.isfinite(value) for value in numbers.values()), "Non-finite numeric input")
        require(numbers["wind_speed_ms"] >= 0, "Negative wind speed")
        require(0 <= numbers["predicted_power"] <= 1, "Power outside normalized [0,1] range")
    require(seen == expected_hours, "Missing hourly forecast rows")
    require(manifest.get("row_count") == len(seen), "Manifest row count differs from CSV")

    expected_runs = {(turbine, day.isoformat()) for turbine in TURBINES for day in days}
    seen_runs, seen_ids, sources_checked = set(), set(), 0
    for run in manifest.get("runs", []):
        key = (run.get("turbine"), run.get("target_date"))
        require(key in expected_runs and key not in seen_runs, "Unexpected or duplicated daily run")
        seen_runs.add(key)
        forecast_id = run.get("forecast_id")
        require(isinstance(forecast_id, str) and forecast_id and forecast_id not in seen_ids, "Missing or repeated forecast ID")
        seen_ids.add(forecast_id)
        require(run.get("status") == "success" and run.get("horizon_hours") == 24, "Incomplete forecast run")
        require(run.get("power_unit") == "normalized", "Run power units differ from normalized scale")
        target = datetime.combine(date.fromisoformat(key[1]), datetime.min.time(), tzinfo=local_zone)
        issued = instant(run["issue_time"])
        require(instant(run["forecast_start"]) == target.astimezone(timezone.utc), "Run starts outside target day")
        require(issued == (target - timedelta(hours=1)).astimezone(timezone.utc), "Unexpected decision time")
        weather = run["weather"]
        require(all(isinstance(weather.get(key), str) and weather[key].strip()
                    for key in ("provider", "model")), "Weather provider and model identity are required")
        require(is_sha256(weather.get("input_sha256")), "Weather input SHA-256 is missing or malformed")
        run_time, available = instant(weather["run_time"]), instant(weather["available_at"])
        require(run_time <= available <= issued, "Weather archive timestamps exceed decision time")
        require(instant(weather["issue_time"]) == issued, "Weather decision time mismatch")
        sources = weather.get("sources", [])
        require(isinstance(sources, list) and bool(sources), "Weather source records are missing")
        target_hours = {target.astimezone(timezone.utc) + timedelta(hours=h) for h in range(24)}
        fields_by_time = {stamp: set() for stamp in target_hours}
        source_times = []
        for source in sources:
            sources_checked += 1
            require(isinstance(source, dict), "Malformed weather source record")
            valid = instant(source["valid_time"])
            require(valid in target_hours, "Weather source targets an unexpected hour")
            modified = instant(source["last_modified"])
            source_times.append(modified)
            require(modified <= available, "Source timestamp exceeds recorded archive availability")
            fields_by_time[valid].add(source.get("field"))
        require(max(source_times) == available, "Archive availability is not the maximum source timestamp")
        require(all({"inventory", "temperature", "u100", "v100"} <= fields for fields in fields_by_time.values()),
                "Missing hourly weather source fields")
        model = run["model"]
        train_end = instant(model["training_end"])
        require(instant(model["training_start"]) <= train_end, "Inverted model training interval")
        require(train_end + timedelta(hours=1) <= instant(model["training_available_at"]) <= issued,
                "Training targets were not complete at decision time")
        calibration = model.get("calibration", {})
        require(isinstance(calibration, dict), "Malformed calibration metadata")
        if "applied" in calibration:
            require(type(calibration["applied"]) is bool, "Calibration applied flag must be boolean")
        if calibration.get("applied"):
            require(instant(calibration["available_at"]) <= issued, "Calibration uses targets unavailable at decision time")
            require(bool(calibration.get("fit_start")) and bool(calibration.get("fit_end")),
                    "Calibration fit interval is missing")
            require(instant(calibration["fit_start"]) <= instant(calibration["fit_end"]),
                    "Calibration fit interval is inverted")
            require(instant(calibration["fit_end"]) + timedelta(hours=1) <= instant(calibration["available_at"]),
                    "Calibration contains an incomplete target hour")
            require(is_sha256(calibration.get("artifact_sha256")), "Calibration artifact SHA-256 is missing or malformed")
            require(calibration.get("status") == "applied" and calibration.get("method") == "ridge_affine"
                    and isinstance(calibration.get("version"), str) and bool(calibration["version"]),
                    "Calibration method, version or status is missing or unsupported")
            parameters = calibration.get("parameters")
            require(isinstance(parameters, dict) and set(parameters) == {"1-24", "25-48"},
                    "Calibration lead-band parameters are missing")
            for values in parameters.values():
                require(isinstance(values, dict), "Malformed calibration coefficients")
                slope, intercept, count = (values.get(key) for key in ("slope", "intercept", "training_rows"))
                require(finite_number(slope) and 0 <= slope <= 2 and finite_number(intercept)
                        and type(count) is int and count >= 48, "Invalid calibration coefficients or training count")
    require(seen_runs == expected_runs, "Missing daily forecast provenance")
    return {"status": "success", "row_count": len(seen), "daily_runs": len(seen_runs),
            "weather_sources_checked": sources_checked, "full_february": len(days) == 28,
            "csv_sha256": digest,
            "checks": ["SHA-256", "hourly coverage", "normalized bounds", "decision timestamps", "weather provenance", "training cutoff"],
            "limitations": ["Verifies internal consistency of saved evidence, not February forecast accuracy",
                            "HTTP Last-Modified is archive evidence, not independent first-publication proof",
                            "SCADA timezone remains an unconfirmed assumption"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=str(ROOT / "submission_february_2026.csv"))
    parser.add_argument("--allow-partial-month", action="store_true")
    args = parser.parse_args()
    try:
        report = verify_submission(args.path, require_full_month=not args.allow_partial_month)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        raise SystemExit(1)
    print(json.dumps(report, ensure_ascii=False, indent=2))
