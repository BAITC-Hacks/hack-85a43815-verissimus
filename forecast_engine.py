import os
import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Точные координаты ветропарка в Шелекском коридоре (Нурлы)
TURBINE_COORDS = {
    "turbine_1": {"lat": 43.643198, "lon": 78.538828},
    "turbine_2": {"lat": 43.645150, "lon": 78.535604},
}


def _find_column(columns: list, keywords: list, description: str) -> str:
  """Безопасный поиск колонки по ключевым словам (рус/англ)."""
  for col in columns:
    if any(kw in col.lower() for kw in keywords):
      return col
  raise KeyError(
      f"Не удалось найти колонку для '{description}'. Доступные колонки:"
      f" {list(columns)}"
  )


def load_and_preprocess_turbine_data(csv_path: str) -> pd.DataFrame:
  """Загрузка данных турбины, приведение к единому формату и часовая агрегация."""
  df = pd.read_csv(csv_path)

  time_col = _find_column(
      df.columns, ["время", "time", "date", "timestamp"], "время"
  )
  speed_col = _find_column(
      df.columns, ["скорость", "speed", "wind_spd", "ws"], "скорость ветра"
  )
  power_col = _find_column(
      df.columns, ["мощность", "power", "active_power", "p_act"], "мощность"
  )
  temp_col = _find_column(
      df.columns,
      ["температура", "temp", "temperature", "t_air"],
      "температура",
  )

  df[time_col] = pd.to_datetime(df[time_col])

  # Безопасное приведение к tz-naive для исключения конфликтов срезов дат
  if hasattr(df[time_col].dt, "tz") and df[time_col].dt.tz is not None:
    df[time_col] = df[time_col].dt.tz_convert(None)

  df = df.sort_values(by=time_col).set_index(time_col)

  df_clean = df[[speed_col, power_col, temp_col]].copy()
  df_clean.columns = ["wind_speed", "target_power", "temperature"]

  for col in df_clean.columns:
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

  # Агрегация до 1 часа (требование ТЗ)
  df_hourly = df_clean.resample("1h").mean().dropna()

  # Инженерные признаки (физика ветра по закону Беца: P ~ v^3)
  df_hourly["hour"] = df_hourly.index.hour
  df_hourly["month"] = df_hourly.index.month
  df_hourly["wind_speed_cubed"] = df_hourly["wind_speed"] ** 3

  return df_hourly


def train_wind_model(df: pd.DataFrame, split_ratio: float = 0.8):
  """Обучение градиентного бустинга LightGBM с валидацией качества."""
  features = ["wind_speed", "temperature", "hour", "month", "wind_speed_cubed"]

  split_idx = int(len(df) * split_ratio)
  train_df = df.iloc[:split_idx]
  val_df = df.iloc[split_idx:]

  if len(val_df) == 0 or len(train_df) == 0:
    raise ValueError("Недостаточно данных для разделения выборки на train/val.")

  # Динамический порог максимальной мощности (квантиль 99.9%)
  min_power = 0.0
  max_power = float(df["target_power"].quantile(0.999))

  model = LGBMRegressor(
      n_estimators=150,
      learning_rate=0.05,
      max_depth=6,
      random_state=42,
      verbose=-1,
  )
  model.fit(train_df[features], train_df["target_power"])

  val_preds = np.clip(model.predict(val_df[features]), min_power, max_power)
  rmse = float(np.sqrt(mean_squared_error(val_df["target_power"], val_preds)))

  val_metrics = {
      "Validation_R2": round(
          float(r2_score(val_df["target_power"], val_preds)), 3
      ),
      "Validation_MAE": round(
          float(mean_absolute_error(val_df["target_power"], val_preds)), 4
      ),
      "Validation_RMSE": round(rmse, 4),
  }

  # Финальное дообучение на всем объеме исторических данных
  model.fit(df[features], df["target_power"])
  return model, val_metrics, max_power


def fetch_open_meteo_forecast(
    lat: float, lon: float, start_date: str, end_date: str
) -> pd.DataFrame:
  """Запрашивает архивный прогноз погоды из Open-Meteo (с fallback-механизмом)."""
  # Официальный эндпоинт архивных прогнозов (Historical Forecast API)
  url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
  params = {
      "latitude": lat,
      "longitude": lon,
      "start_date": start_date,
      "end_date": end_date,
      "hourly": ["temperature_2m", "wind_speed_10m", "wind_speed_100m"],
      "wind_speed_unit": "ms",
      "timezone": "UTC",
  }

  try:
    res = requests.get(url, params=params, timeout=15)
    res.raise_for_status()
    data = res.json()
  except Exception:
    # Резервный эндпоинт архива погоды
    backup_url = "https://archive-api.open-meteo.com/v1/archive"
    res = requests.get(backup_url, params=params, timeout=15)
    res.raise_for_status()
    data = res.json()

  times = pd.to_datetime(data["hourly"]["time"]).tz_localize(None)
  hourly = data["hourly"]

  # Ветер на высоте гондолы (100 м) или на уровне земли (10 м)
  wind_speed = hourly.get("wind_speed_100m")
  if wind_speed is None or all(v is None for v in wind_speed):
    wind_speed = hourly["wind_speed_10m"]

  df_weather = pd.DataFrame({
      "time": times,
      "temperature": hourly["temperature_2m"],
      "wind_speed": wind_speed,
  }).set_index("time")

  df_weather["hour"] = df_weather.index.hour
  df_weather["month"] = df_weather.index.month
  df_weather["wind_speed_cubed"] = df_weather["wind_speed"] ** 3
  return df_weather


def generate_agent_forecast(
    turbine_id: str, forecast_date: str, horizon_hours: int = 48
) -> dict:
  """Основной Tool для AI-Агента: выполняет прогноз на 24-48 часов и формирует аудит."""
  csv_file = f"data/{turbine_id}.csv"
  if not os.path.exists(csv_file):
    return {"status": "error", "message": f"Файл {csv_file} не найден"}

  df = load_and_preprocess_turbine_data(csv_file)
  model, val_metrics, max_power = train_wind_model(df)

  start_dt = pd.to_datetime(forecast_date)
  if hasattr(start_dt, "tz") and start_dt.tz is not None:
    start_dt = start_dt.tz_convert(None)

  end_dt = start_dt + pd.Timedelta(hours=horizon_hours - 1)
  coords = TURBINE_COORDS.get(turbine_id, TURBINE_COORDS["turbine_1"])

  df_weather = fetch_open_meteo_forecast(
      coords["lat"],
      coords["lon"],
      start_dt.strftime("%Y-%m-%d"),
      end_dt.strftime("%Y-%m-%d"),
  )

  forecast_window = df_weather.loc[start_dt:end_dt].copy()
  if forecast_window.empty:
    return {
        "status": "error",
        "message": (
            f"Не удалось получить метеоданные для диапазона {start_dt} —"
            f" {end_dt}"
        ),
    }

  features = ["wind_speed", "temperature", "hour", "month", "wind_speed_cubed"]
  preds = model.predict(forecast_window[features])
  preds = np.clip(preds, 0.0, max_power)
  forecast_window["predicted_power"] = np.round(preds, 3)

  # Сохраняем полный почасовой ряд (все 24-48 точек) для отрисовки графиков в Streamlit
  points = []
  for t, row in forecast_window.iterrows():
    points.append({
        "time": t.strftime("%Y-%m-%d %H:%M:%S"),
        "wind_speed": round(float(row["wind_speed"]), 2),
        "temperature": round(float(row["temperature"]), 2),
        "predicted_power": round(float(row["predicted_power"]), 3),
    })

  avg_power = float(np.mean(preds))
  max_power_pred = float(np.max(preds))
  min_power_pred = float(np.min(preds))

  # Порог штиля / отсутствия генерации (менее 5% от рабочей мощности)
  calm_threshold = 0.05 * (max_power if max_power > 0 else 1.0)
  calm_hours = int(np.sum(preds < calm_threshold))

  return {
      "status": "success",
      "turbine": turbine_id,
      "period": f"{start_dt} — {end_dt}",
      "horizon_hours": horizon_hours,
      "validation_metrics": val_metrics,
      "summary": {
          "avg_predicted_power": round(avg_power, 3),
          "avg_predicted_power_pu": round(
              avg_power / max_power if max_power > 0 else avg_power, 3
          ),
          "max_predicted_power": round(max_power_pred, 3),
          "min_predicted_power": round(min_power_pred, 3),
          "calm_risk_hours": calm_hours,
      },
      "forecast_sample": points,  # Полный ряд для построения красивого графика в UI
  }
def calculate_market_penalties(
    turbine_id: str,
    forecast_date: str,
    imbalance_tariff_kzt: float = 25000.0,
    horizon_hours: int = 24,
) -> dict:
  """Рассчитывает прогнозируемые финансовые риски на Балансирующем рынке электроэнергии (БРЭ) Казахстана

  на основе ожидаемой ошибки модели (MAE) и объемов генерации.
  """
  forecast_res = generate_agent_forecast(turbine_id, forecast_date, horizon_hours)
  if forecast_res.get("status") != "success":
    return {"status": "error", "message": "Не удалось сформировать базовый прогноз"}

  summary = forecast_res.get("summary", {})
  val_metrics = forecast_res.get("validation_metrics", {})

  avg_p = summary.get("avg_predicted_power", 0.0)
  total_mwh = avg_p * horizon_hours
  mae = val_metrics.get("Validation_MAE", 0.045)

  # Ожидаемый физический небаланс в МВт*ч
  expected_imbalance_mwh = round(total_mwh * mae, 3)

  # Финансовый риск по тарифу балансирования (в среднем 25 000 KZT / МВт*ч)
  financial_risk_kzt = round(expected_imbalance_mwh * imbalance_tariff_kzt, 2)

  # Экономический эффект внедрения AI по сравнению с базовой константной моделью (~15% ошибки)
  naive_imbalance_mwh = total_mwh * 0.15
  savings_kzt = round(
      (naive_imbalance_mwh - expected_imbalance_mwh) * imbalance_tariff_kzt, 2
  )

  return {
      "status": "success",
      "turbine": turbine_id,
      "forecast_date": forecast_date,
      "horizon_hours": horizon_hours,
      "total_expected_generation_mwh": round(total_mwh, 2),
      "expected_imbalance_mwh": expected_imbalance_mwh,
      "financial_risk_kzt": financial_risk_kzt,
      "prevented_losses_kzt": savings_kzt,
      "recommendation": (
          "Высокий риск небаланса: рекомендуется законтрактовать резерв"
          if expected_imbalance_mwh > 2.0
          else "Риск в пределах допустимого диапазона (ГОСТ/КОРЭМ)"
      ),
  }