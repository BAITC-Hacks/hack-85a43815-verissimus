import json
import os
import dotenv
import forecast_engine
from openai import OpenAI
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="Agentic Wind Forecaster | Самрук-Казына",
    page_icon="🌬️",
    layout="wide",
)

if "messages" not in st.session_state:
  st.session_state.messages = []
if "latest_forecast" not in st.session_state:
  st.session_state.latest_forecast = None
if "latest_penalties" not in st.session_state:
  st.session_state.latest_penalties = None

# Схема инструментов для OpenAI Function Calling
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "generate_agent_forecast",
            "description": (
                "Автономно запрашивает архивный прогноз погоды Open-Meteo"
                " (Шелек), запускает ML-модель LightGBM и выдает почасовой"
                " прогноз выработки на 24-48 часов."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "turbine_id": {
                        "type": "string",
                        "enum": ["turbine_1", "turbine_2"],
                        "description": "Идентификатор ветротурбины",
                    },
                    "forecast_date": {
                        "type": "string",
                        "description": "Дата старта прогноза (ГГГГ-ММ-ДД)",
                    },
                    "horizon_hours": {
                        "type": "integer",
                        "description": "Горизонт в часах (24 или 48)",
                        "default": 48,
                    },
                },
                "required": ["turbine_id", "forecast_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_market_penalties",
            "description": (
                "Рассчитывает коммерческие риски небалансов и предотвращенные"
                " финансовые потери в тенге (₸) на Балансирующем рынке"
                " электроэнергии Казахстана (БРЭ / КОРЭМ)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "turbine_id": {
                        "type": "string",
                        "enum": ["turbine_1", "turbine_2"],
                        "description": "Идентификатор ветротурбины",
                    },
                    "forecast_date": {
                        "type": "string",
                        "description": "Дата расчета (ГГГГ-ММ-ДД)",
                    },
                    "horizon_hours": {
                        "type": "integer",
                        "description": "Горизонт планирования в часах (24 или"
                        " 48)",
                        "default": 48,
                    },
                    "imbalance_tariff_kzt": {
                        "type": "number",
                        "description": (
                            "Тариф за небаланс в тенге за МВт*ч (по умолчанию"
                            " 25000)"
                        ),
                        "default": 25000.0,
                    },
                },
                "required": ["turbine_id", "forecast_date"],
            },
        },
    },
]

# Боковая панель
st.sidebar.title("🌬️ Диспетчер ВЭС")
st.sidebar.markdown(
    "**Объект:** Шелекский ветропарк\n**Координаты:** 43.643°N, 78.538°E"
)
st.sidebar.info("Тестовый период по ТЗ:\n01.02.2026 — 28.02.2026")

selected_turbine = st.sidebar.selectbox(
    "Турбина по умолчанию:", ["turbine_1", "turbine_2"]
)
selected_date = st.sidebar.date_input(
    "Дата старта:", pd.to_datetime("2026-02-01")
)

st.title("⚡ Автономный AI-Диспетчер ВЭС (Самрук-Казына)")
st.caption(
    "Agentic AI мультиагент: прогнозирование генерации 24–48 ч, аудит погоды"
    " Open-Meteo и финансовая оптимизация БРЭ"
)

# Вывод ключевых показателей
if st.session_state.latest_forecast:
  res = st.session_state.latest_forecast
  val = res.get("validation_metrics", {})
  summ = res.get("summary", {})
  pen = st.session_state.latest_penalties or {}

  col1, col2, col3, col4 = st.columns(4)
  col1.metric("Точность модели (R²)", f"{val.get('Validation_R2', 0.0):.3f}")
  col2.metric("Ошибка MAE", f"{val.get('Validation_MAE', 0.0):.4f}")
  col3.metric(
      "Ср. мощность", f"{summ.get('avg_predicted_power_pu', 0.0)*100:.1f} %"
  )

  risk_val = pen.get("financial_risk_kzt")
  if risk_val:
    col4.metric(
        "Риск дисбаланса (БРЭ)",
        f"{risk_val:,.0f} ₸".replace(",", " "),
        delta=(
            f"Экономия AI: {pen.get('prevented_losses_kzt', 0):,.0f} ₸".replace(
                ",", " "
            )
        ),
    )
  else:
    col4.metric(
        "Часов штиля (<5%)",
        f"{summ.get('calm_risk_hours', 0)} ч",
        delta_color="inverse",
    )

  sample = res.get("forecast_sample", [])
  if sample:
    df_sample = pd.DataFrame(sample)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_sample["time"],
            y=df_sample["predicted_power"],
            mode="lines+markers",
            name="Прогноз мощности (p.u. / МВт)",
            line=dict(color="#00FFCC", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_sample["time"],
            y=df_sample["wind_speed"],
            mode="lines",
            name="Скорость ветра, м/с (Open-Meteo)",
            line=dict(color="#FFB300", dash="dash"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title=(
            f"Почасовой график генерации и ветра | Турбина:"
            f" {res.get('turbine')} | Горизонт: {res.get('horizon_hours')} ч"
        ),
        template="plotly_dark",
        height=380,
        yaxis=dict(title="Мощность"),
        yaxis2=dict(
            title="Скорость ветра (м/с)", overlaying="y", side="right"
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# История чата
for msg in st.session_state.messages:
  with st.chat_message(msg["role"]):
    st.markdown(msg["content"])

# Ввод запроса
user_input = st.chat_input(
    "Задайте задачу агенту (например: Сделай прогноз для turbine_1 на 2 февраля"
    " 2026 на 48 часов и оцени финансовые риски на БРЭ)"
)

if user_input:
  st.session_state.messages.append({"role": "user", "content": user_input})
  with st.chat_message("user"):
    st.markdown(user_input)

  if not api_key:
    st.error("Ключ OPENAI_API_KEY не найден в .env")
    st.stop()

  client = OpenAI(api_key=api_key)

  system_prompt = (
      "Ты ведущий инженер-диспетчер и финансовый аналитик системного оператора"
      " энергосетей Казахстана (KEGOC / Самрук-Энерго). "
      "Твоя задача — формировать почасовые прогнозы выработки ВЭС, автономно"
      " вызывая инструмент generate_agent_forecast, "
      "а также при необходимости оценивать финансовые риски и экономический"
      " эффект через инструмент calculate_market_penalties. "
      "Анализируй штиль, провалы генерации, риски штрафов на Балансирующем"
      " рынке (БРЭ) в тенге (₸) и давай прикладные рекомендации."
  )

  api_msgs = [{"role": "system", "content": system_prompt}] + [
      {"role": m["role"], "content": m["content"]}
      for m in st.session_state.messages
  ]

  with st.chat_message("assistant"):
    with st.status(
        "Agentic AI анализирует задачу и координирует вызов инструментов...",
        expanded=True,
    ) as status:
      response = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=api_msgs,
          tools=tools_schema,
          tool_choice="auto",
      )
      msg = response.choices[0].message

      while msg.tool_calls:
        api_msgs.append(msg)
        for tool_call in msg.tool_calls:
          fn = tool_call.function.name
          args = json.loads(tool_call.function.arguments)
          st.write(f"🤖 **Вызов инструмента агента:** `{fn}`")
          st.json(args)

          if fn == "generate_agent_forecast":
            res_data = forecast_engine.generate_agent_forecast(**args)
            st.session_state.latest_forecast = res_data
            result_str = json.dumps(res_data)
          elif fn == "calculate_market_penalties":
            res_data = forecast_engine.calculate_market_penalties(**args)
            st.session_state.latest_penalties = res_data
            result_str = json.dumps(res_data)
          else:
            result_str = json.dumps({"error": "Unknown function"})

          api_msgs.append({
              "role": "tool",
              "tool_call_id": tool_call.id,
              "content": result_str,
          })

        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=api_msgs, tools=tools_schema
        )
        msg = response.choices[0].message

      status.update(
          label="Комплексный прогноз и коммерческий аудит завершены!",
          state="complete",
      )

    st.markdown(msg.content)
    st.session_state.messages.append(
        {"role": "assistant", "content": msg.content}
    )
    st.rerun()