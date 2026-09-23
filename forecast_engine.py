"""Auditable normalized-power forecasting with a strict information cutoff."""
from __future__ import annotations
from collections import OrderedDict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from power_calibration import apply_calibration

ROOT = Path(__file__).resolve().parent
SITE_TIMEZONE = os.getenv("SCADA_TIMEZONE", "Asia/Almaty")
DATA_DIR = Path(os.getenv("ENERGYAI_DATA_DIR", str(ROOT / "data")))
ARTIFACT_DIR = Path(os.getenv("ENERGYAI_ARTIFACT_DIR", str(ROOT / "artifacts")))
CALIBRATION_FILE = ROOT / "models" / "power_calibration.json"
MODEL_VERSION = "normalized-power-curve-v2"
FEATURES = ["wind_speed", "temperature", "hour", "month", "wind_speed_cubed"]
TURBINE_COORDS = {
    "turbine_1": {"lat": 43.645150, "lon": 78.535604},
    "turbine_2": {"lat": 43.643198, "lon": 78.538828},
}
_MODEL_CACHE = OrderedDict()
_ARTIFACT_LOCK = threading.RLock()


def _find_column(columns, keywords, description):
    matches = [c for c in columns if any(k in c.lower() for k in keywords)]
    if len(matches) != 1:
        raise ValueError(f"Неоднозначная или отсутствующая колонка: {description}")
    return matches[0]


def _features(frame, site_timezone=SITE_TIMEZONE):
    result = frame.copy()
    local = result.index.tz_convert(site_timezone)
    result["hour"] = local.hour
    result["month"] = local.month
    result["wind_speed_cubed"] = result["wind_speed"] ** 3
    return result


@lru_cache(maxsize=8)
def _load_data(path, modified_ns, size, site_timezone):
    raw = pd.read_csv(path)
    cols = {
        "time": _find_column(raw.columns, ["время", "time", "date"], "время"),
        "wind_speed": _find_column(raw.columns, ["скорость", "speed"], "ветер"),
        "target_power": _find_column(raw.columns, ["мощность", "power"], "мощность"),
        "temperature": _find_column(raw.columns, ["температура", "temp"], "температура"),
    }
    timestamps = pd.DatetimeIndex(pd.to_datetime(raw[cols["time"]], errors="coerce"))
    invalid_times = int(timestamps.isna().sum())
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize(site_timezone, ambiguous="NaT", nonexistent="NaT")
    timestamps = timestamps.tz_convert("UTC")
    values = raw[[cols[c] for c in ("wind_speed", "target_power", "temperature")]].apply(pd.to_numeric, errors="coerce")
    values.columns = ["wind_speed", "target_power", "temperature"]
    values.index = timestamps
    clock_rows = int(values.index.isna().sum()) - invalid_times
    values = values.loc[~values.index.isna()].sort_index()
    if values.index.has_duplicates:
        raise ValueError("Повторяющиеся временные метки SCADA; исправьте источник.")
    if ((values.index.minute % 10 != 0) | (values.index.second != 0)).any():
        raise ValueError("Ожидаются десятиминутные метки SCADA.")
    bad = ~np.isfinite(values).all(axis=1) | ~values.target_power.between(0, 1) | (values.wind_speed < 0)
    bad_count = int(bad.sum())
    values.loc[bad, :] = np.nan
    counts = values.resample("1h").count().min(axis=1)
    hourly = _features(values.resample("1h").mean().loc[counts.eq(6)], site_timezone)
    hourly.attrs["data_quality"] = {
        "raw_rows": len(raw), "invalid_timestamp_rows": invalid_times,
        "ambiguous_or_nonexistent_clock_rows_removed": clock_rows,
        "invalid_value_rows_removed": bad_count, "empty_hours": int(counts.eq(0).sum()),
        "incomplete_nonempty_hours_removed": int(((counts > 0) & (counts != 6)).sum()),
        "complete_hours": len(hourly), "samples_required_per_hour": 6,
        "timestamp_convention_assumption": "10-minute interval start; hour available at its end",
        "timezone_assumption": site_timezone,
    }
    hourly.attrs["source_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return hourly


def load_and_preprocess_turbine_data(csv_path, site_timezone=SITE_TIMEZONE):
    path = Path(csv_path).resolve()
    stat = path.stat()
    return _load_data(str(path), stat.st_mtime_ns, stat.st_size, site_timezone).copy()


def train_wind_model(df, split_ratio=0.8):
    """Observed-weather validation is explicitly distinct from forecast skill."""
    if not 0 < split_ratio < 1 or len(df) < 72:
        raise ValueError("Нужно минимум 72 полных часа и корректный split_ratio.")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("История должна быть упорядочена и уникальна.")
    if not np.isfinite(df[FEATURES + ["target_power"]]).all().all():
        raise ValueError("Нечисловые значения в обучении.")
    split = int(len(df) * split_ratio)
    train, validation = df.iloc[:split], df.iloc[split:]
    model = LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=6,
                          random_state=42, n_jobs=2, verbose=-1)
    model.fit(train[FEATURES], train.target_power)
    predicted = np.clip(model.predict(validation[FEATURES]), 0, 1)
    metrics = {
        "scope": "observed_weather_power_curve",
        "description": "Мощность при измеренной погоде; не качество прогноза на 24–48 часов",
        "Validation_R2": float(r2_score(validation.target_power, predicted)),
        "Validation_MAE": float(mean_absolute_error(validation.target_power, predicted)),
        "Validation_RMSE": float(np.sqrt(mean_squared_error(validation.target_power, predicted))),
        "validation_start": validation.index[0].isoformat(),
        "validation_end": validation.index[-1].isoformat(),
        "validation_hours": len(validation), "power_unit": "normalized",
        "train_mean_baseline_mae": float(np.mean(np.abs(validation.target_power - train.target_power.mean()))),
    }
    model.fit(df[FEATURES], df.target_power)
    return model, metrics, 1.0


def _as_utc(value, site_timezone=SITE_TIMEZONE):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Не указана корректная дата.")
    if stamp.tz is None:
        stamp = stamp.tz_localize(site_timezone, ambiguous="raise", nonexistent="raise")
    return stamp.tz_convert("UTC")


def forecast_request(turbine_id, forecast_date, horizon_hours=48, issue_time=None):
    if turbine_id not in TURBINE_COORDS:
        raise ValueError("Допустимы только turbine_1 и turbine_2.")
    if isinstance(horizon_hours, bool) or not isinstance(horizon_hours, int) or horizon_hours not in (24, 48):
        raise ValueError("Горизонт должен быть 24 или 48 часов.")
    start = _as_utc(forecast_date)
    if start.minute or start.second or start.microsecond:
        raise ValueError("Начало прогноза должно совпадать с началом часа.")
    issued = start - pd.Timedelta(hours=1) if issue_time is None else _as_utc(issue_time)
    if issued >= start:
        raise ValueError("Момент выпуска должен быть раньше первого целевого часа.")
    if issued > pd.Timestamp.now(tz="UTC"):
        raise ValueError("Нельзя использовать будущий момент выпуска.")
    return start, issued, pd.date_range(start, periods=horizon_hours, freq="h")


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _persist(result, request_key, refresh):
    with _ARTIFACT_LOCK:
        pointer = ARTIFACT_DIR / "requests" / f"{request_key}.json"
        previous_id = None
        if pointer.exists():
            previous_id = json.loads(pointer.read_text(encoding="utf-8")).get("forecast_id")
        result["previous_forecast_id"] = previous_id
        result["changed_from_previous"] = previous_id is not None and previous_id != result["forecast_id"]
        artifact = ARTIFACT_DIR / "runs" / f"{result['forecast_id']}.json"
        if not artifact.exists():
            _atomic_json(artifact, result)
        _atomic_json(pointer, {"forecast_id": result["forecast_id"]})
        with (ARTIFACT_DIR / "agent_audit.jsonl").open("a", encoding="utf-8") as audit:
            audit.write(json.dumps({
                "event_time": datetime.now(timezone.utc).isoformat(),
                "action": "refresh" if refresh else "forecast", "status": "success",
                "forecast_id": result["forecast_id"], "previous_forecast_id": previous_id,
                "changed": result["changed_from_previous"], "issue_time": result["issue_time"],
                "weather_run_time": result["weather"].get("run_time"),
                "steps": ["validate_request", "select_available_training_hours", "fetch_issued_weather",
                          "validate_weather_coverage_and_availability", "predict", "analyze", "persist"],
            }, ensure_ascii=False) + "\n")


def generate_agent_forecast(turbine_id, forecast_date, horizon_hours=48, issue_time=None, refresh=False):
    """Errors remain errors; no fallback to observations, invented units or zero rows."""
    try:
        from weather_archive import fetch_weather
        start, issued, expected = forecast_request(turbine_id, forecast_date, horizon_hours, issue_time)
        data = load_and_preprocess_turbine_data(DATA_DIR / f"{turbine_id}.csv")
        training = data.loc[data.index + pd.Timedelta(hours=1) <= issued].copy()
        if len(training) < 72:
            raise ValueError("Недостаточно истории, доступной на момент выпуска.")
        key = (data.attrs["source_sha256"], training.index[-1].isoformat(), len(training), SITE_TIMEZONE, MODEL_VERSION)
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = train_wind_model(training)
            if len(_MODEL_CACHE) > 8:
                _MODEL_CACHE.popitem(last=False)
        model, validation, _ = _MODEL_CACHE[key]
        coordinates = TURBINE_COORDS[turbine_id]
        weather, provenance = fetch_weather(coordinates["lat"], coordinates["lon"], expected, issued, refresh=refresh)
        if weather.index.tz is None:
            raise ValueError("Погода должна содержать часовой пояс.")
        weather = weather.copy()
        weather.index = weather.index.tz_convert("UTC")
        if weather.index.has_duplicates or not weather.index.equals(expected):
            raise ValueError(f"Требуется ровно {horizon_hours} последовательных часов погоды без пропусков.")
        if not np.isfinite(weather[["wind_speed", "temperature"]]).all().all() or (weather.wind_speed < 0).any():
            raise ValueError("Погода содержит отсутствующие или недопустимые значения.")
        if not provenance.get("available_at") or _as_utc(provenance["available_at"]) > issued:
            raise ValueError("Погодный выпуск не был доступен на момент решения.")
        if not provenance.get("run_time") or _as_utc(provenance["run_time"]) > issued:
            raise ValueError("Недопустимый момент инициализации погодной модели.")
        predicted = np.asarray(model.predict(_features(weather)[FEATURES]), dtype=float)
        if predicted.shape != (horizon_hours,):
            raise ValueError("Модель вернула неверное число почасовых прогнозов.")
        predicted = np.clip(predicted, 0, 1)
        if not np.isfinite(predicted).all():
            raise ValueError("Модель вернула нечисловой прогноз.")
        raw_predicted = predicted.copy()
        predicted, calibration = apply_calibration(
            predicted, expected, issued, turbine_id, key[0], MODEL_VERSION, SITE_TIMEZONE, CALIBRATION_FILE)
        points = [{"time": t.isoformat(), "wind_speed": float(row.wind_speed),
                   "temperature": float(row.temperature), "predicted_power": float(p)}
                  for (t, row), p in zip(weather.iterrows(), predicted)]
        if calibration["applied"]:
            for point, raw_power in zip(points, raw_predicted):
                point["raw_predicted_power"] = float(raw_power)
        identity = {"turbine": turbine_id, "issue_time": issued.isoformat(), "start": start.isoformat(),
                    "horizon": horizon_hours, "training_source": key[0], "training_end": key[1],
                    "model": MODEL_VERSION, "timezone": SITE_TIMEZONE, "points": points,
                    "weather_run": provenance["run_time"], "weather_provider": provenance.get("provider"),
                    "weather_input_sha256": provenance.get("input_sha256"), "calibration": calibration}
        forecast_id = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()[:24]
        result = {
            "status": "success", "forecast_id": forecast_id, "turbine": turbine_id,
            "issue_time": issued.isoformat(), "forecast_start": start.isoformat(),
            "period": f"{expected[0].isoformat()} — {expected[-1].isoformat()}",
            "horizon_hours": horizon_hours, "power_unit": "normalized", "timezone_assumption": SITE_TIMEZONE,
            "assumptions": ["Часовой пояс SCADA не подтверждён организаторами.",
                            "Метки SCADA приняты за начало десятиминутного интервала.",
                            "Задержка получения завершённого часа SCADA принята нулевой.",
                            "Шкала мощности [0, 1]; перевод в МВт неизвестен.",
                            "Сеточный ветер GFS 100 м используется без калибровки к датчику турбины."],
            "weather": provenance, "validation_metrics": validation,
            "model": {"version": MODEL_VERSION, "training_start": training.index[0].isoformat(),
                      "training_end": training.index[-1].isoformat(),
                      "training_available_at": (training.index[-1] + pd.Timedelta(hours=1)).isoformat(),
                      "training_hours": len(training), "source_sha256": key[0], "calibration": calibration},
            "data_quality": data.attrs["data_quality"],
            "summary": {"avg_predicted_power": float(np.mean(predicted)),
                        "max_predicted_power": float(np.max(predicted)), "min_predicted_power": float(np.min(predicted)),
                        "low_generation_hours": int(np.sum(predicted < .05)),
                        "normalized_power_hours": float(np.sum(predicted)),
                        "analysis": "Низкая прогнозная мощность не доказывает метеорологический штиль."},
            "forecast_sample": points,
        }
        request_key = hashlib.sha256(json.dumps([turbine_id, start.isoformat(), issued.isoformat(), horizon_hours, SITE_TIMEZONE]).encode()).hexdigest()[:24]
        _persist(result, request_key, refresh)
        return result
    except Exception as exc:
        return {"status": "error", "message": str(exc), "error_type": type(exc).__name__,
                "turbine": turbine_id, "forecast_date": str(forecast_date)}


def calculate_market_penalties(*args, **kwargs):
    return {"status": "unavailable", "message": "Денежный расчёт отключён: неизвестны база нормализации, применимая ошибка полного прогноза и параметры рынка."}
