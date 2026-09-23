"""A small, explicit data contract between forecasting tools and the LLM.

The persisted forecast remains complete. Only the API payload omits per-file
download inventories; numerical values are never rounded or resampled here.
"""
from copy import deepcopy
from datetime import datetime, timedelta
import math


NOTICES = ("warnings", "limitations", "assumptions", "notes")
SUMMARY_FIELDS = (
    "avg_predicted_power", "max_predicted_power", "min_predicted_power",
    "low_generation_hours", "normalized_power_hours", "analysis", *NOTICES,
)
VALIDATION_FIELDS = (
    "scope", "description", "Validation_R2", "Validation_MAE", "Validation_RMSE",
    "validation_start", "validation_end", "validation_hours", "power_unit",
    "train_mean_baseline_mae", *NOTICES,
)
MODEL_FIELDS = (
    "version", "training_start", "training_end", "training_available_at",
    "training_hours", "source_sha256", *NOTICES,
)
CALIBRATION_FIELDS = (
    "applied", "status", "method", "version", "reason", "artifact_sha256",
    "fit_start", "fit_end", "available_at", "training_rows", *NOTICES,
)
WEATHER_FIELDS = (
    "provider", "model", "run_time", "issue_time", "available_at", "input_sha256",
    "availability_evidence", "selection_policy", "spatial_method",
    "wind_height_m", "temperature_height_m", *NOTICES,
)
QUALITY_FIELDS = (
    "raw_rows", "invalid_timestamp_rows", "ambiguous_or_nonexistent_clock_rows_removed",
    "invalid_value_rows_removed", "empty_hours", "incomplete_nonempty_hours_removed",
    "complete_hours", "samples_required_per_hour", "timestamp_convention_assumption",
    "timezone_assumption", *NOTICES,
)


def _scalar_or_list(value):
    if value is None or isinstance(value, (str, bool, int)):
        return deepcopy(value)
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return [_scalar_or_list(item) for item in value]
    raise ValueError("Expected finite JSON scalar or list")


def _pick(value, fields):
    if not isinstance(value, dict):
        raise ValueError("Expected an object")
    return {key: _scalar_or_list(value[key]) for key in fields if key in value}


def _time(value):
    if not isinstance(value, str):
        raise ValueError("Timestamp must be a string")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.utcoffset() is None:
        raise ValueError("Timestamp must include UTC offset")
    return result


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("Expected a finite number")
    return value


def compact_tool_result(result):
    """Return whitelisted evidence for the agent, or a closed error contract.

    Unknown keys are not sent to the external API. In particular, credentials,
    raw responses, source URLs, byte ranges, and chat histories are excluded.
    This is a payload boundary, not a substitute for the engine's checks.
    """
    try:
        if not isinstance(result, dict):
            raise ValueError("Tool result must be an object")
        status = result.get("status")
        if status in ("error", "unavailable"):
            if not (result.get("error") or result.get("message")):
                raise ValueError("Failure must include an explanation")
            return _pick(result, ("status", "error", "message", "error_type", "turbine", "forecast_date", *NOTICES))
        if status != "success":
            raise ValueError("Unknown tool status")
        for key in ("forecast_id", "turbine", "issue_time", "timezone_assumption"):
            if not isinstance(result.get(key), str) or not result[key]:
                raise ValueError("Missing forecast identity or time metadata")
        horizon = result.get("horizon_hours")
        if type(horizon) is not int or horizon not in (24, 48):
            raise ValueError("Expected a 24 or 48 hour forecast")
        if result.get("power_unit") != "normalized":
            raise ValueError("Unexpected power unit")
        issue = _time(result["issue_time"])
        points = result.get("forecast_sample")
        if not isinstance(points, list) or len(points) != horizon:
            raise ValueError("Incomplete forecast")
        compact_points, previous = [], None
        for row in points:
            if not isinstance(row, dict):
                raise ValueError("Invalid forecast row")
            stamp = _time(row.get("time"))
            if stamp <= issue or (previous is not None and stamp - previous != timedelta(hours=1)):
                raise ValueError("Forecast timestamps must form a future hourly grid")
            point = {"time": row["time"]}
            for key in ("wind_speed", "temperature", "predicted_power"):
                point[key] = _number(row.get(key))
            if point["wind_speed"] < 0 or not 0 <= point["predicted_power"] <= 1:
                raise ValueError("Forecast value out of bounds")
            compact_points.append(point)
            previous = stamp
        if "forecast_start" in result and _time(result["forecast_start"]) != _time(points[0]["time"]):
            raise ValueError("Forecast start differs from first row")
        compact = _pick(result, (
            "status", "forecast_id", "turbine", "issue_time", "forecast_start", "period",
            "horizon_hours", "power_unit", "timezone_assumption", "previous_forecast_id",
            "changed_from_previous", *NOTICES,
        ))
        summary = result.get("summary")
        if not isinstance(summary, dict) or any(key not in summary for key in SUMMARY_FIELDS[:5]):
            raise ValueError("Incomplete summary")
        # Refuse to silently discard future summary statistics. Extend the
        # explicit contract when the engine gains another summary metric.
        if set(summary) - set(SUMMARY_FIELDS):
            raise ValueError("Summary schema changed")
        compact["summary"] = _pick(summary, SUMMARY_FIELDS)
        for key in SUMMARY_FIELDS[:5]:
            _number(compact["summary"][key])
        weather = result.get("weather")
        if not isinstance(weather, dict):
            raise ValueError("Missing weather provenance")
        for key in ("provider", "model", "run_time", "available_at", "input_sha256"):
            if not isinstance(weather.get(key), str) or not weather[key]:
                raise ValueError("Incomplete weather provenance")
        if _time(weather["run_time"]) > issue or _time(weather["available_at"]) > issue:
            raise ValueError("Weather was not available at issue time")
        compact["weather"] = _pick(weather, WEATHER_FIELDS)
        for key in ("requested_coordinates", "grid_cell"):
            if key in weather:
                compact["weather"][key] = _pick(weather[key], ("latitude", "longitude"))
        if "units" in weather:
            compact["weather"]["units"] = _pick(weather["units"], ("wind_speed", "temperature"))
        # File-level notices are small, but meaningful, unlike inventories.
        source_notices = []
        for source in weather.get("sources", []):
            notice = _pick(source, NOTICES)
            if notice and notice not in source_notices:
                source_notices.append(notice)
        if source_notices:
            compact["weather"]["source_notices"] = source_notices
        metrics = result.get("validation_metrics")
        if not isinstance(metrics, dict) or not isinstance(metrics.get("scope"), str) or not metrics["scope"]:
            raise ValueError("Validation scope is required")
        compact["validation_metrics"] = _pick(metrics, VALIDATION_FIELDS)
        if "model" in result:
            model = result["model"]
            compact["model"] = _pick(model, MODEL_FIELDS)
            if "calibration" in model:
                calibration = model["calibration"]
                compact["model"]["calibration"] = _pick(calibration, CALIBRATION_FIELDS)
                if "parameters" in calibration:
                    raw_parameters = calibration["parameters"]
                    parameters = _pick(raw_parameters, ("alpha", "beta", "intercept", "slope"))
                    for value in parameters.values():
                        _number(value)
                    for band in ("1-24", "25-48"):
                        if band in raw_parameters:
                            band_parameters = _pick(raw_parameters[band], ("slope", "intercept", "training_rows"))
                            if not all(key in band_parameters for key in ("slope", "intercept", "training_rows")):
                                raise ValueError("Incomplete calibration band parameters")
                            for value in band_parameters.values():
                                _number(value)
                            if type(band_parameters["training_rows"]) is not int or band_parameters["training_rows"] <= 0:
                                raise ValueError("Invalid calibration sample count")
                            parameters[band] = band_parameters
                    compact["model"]["calibration"]["parameters"] = parameters
        if "data_quality" in result:
            compact["data_quality"] = _pick(result["data_quality"], QUALITY_FIELDS)
        compact["forecast_sample"] = compact_points
        return compact
    except (KeyError, TypeError, ValueError, OverflowError):
        return {"status": "error", "error_type": "InvalidToolResult",
                "error": "Ответ инструмента неполон или не соответствует проверяемой схеме. Численный результат нельзя использовать."}
