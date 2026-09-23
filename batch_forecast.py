import os
import forecast_engine
import pandas as pd


def run_february_simulation():
  print(
      "🚀 Запуск последовательного суточного прогнозирования (февраль 2026)..."
  )

  turbines = ["turbine_1", "turbine_2"]
  dates = pd.date_range(start="2026-02-01", end="2026-02-28", freq="D")

  all_results = []

  for t_id in turbines:
    print(f"\n⚡ Обработка объекта: {t_id}")
    for current_date in dates:
      d_str = current_date.strftime("%Y-%m-%d")
      # Формируем прогноз на 24 часа для каждого дня тестового периода
      res = forecast_engine.generate_agent_forecast(
          turbine_id=t_id, forecast_date=d_str, horizon_hours=24
      )

      if res.get("status") == "success":
        for sample in res.get("forecast_sample", []):
          all_results.append({
              "turbine": t_id,
              "timestamp": sample["time"],
              "wind_speed_ms": sample["wind_speed"],
              "temperature_c": sample["temperature"],
              "predicted_power": sample["predicted_power"],
          })
        calm = res.get("summary", {}).get("calm_risk_hours", 0)
        print(f"  ✓ {d_str} — прогноз сформирован (риск штиля: {calm} ч)")

  df_submission = pd.DataFrame(all_results)
  df_submission.drop_duplicates(subset=["turbine", "timestamp"], inplace=True)
  df_submission.to_csv("submission_february_2026.csv", index=False)
  print(
      "\n Все прогнозы успешно сформированы и сохранены в"
      " 'submission_february_2026.csv'!"
  )
  print(f"Всего строк: {len(df_submission)}")


if __name__ == "__main__":
  run_february_simulation()