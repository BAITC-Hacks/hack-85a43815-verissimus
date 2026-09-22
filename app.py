import json
import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="AI Dispatcher | Smart Grid", page_icon="⚡", layout="wide"
)

# Симуляция телеметрии энергосети
if "grid_state" not in st.session_state:
    st.session_state.grid_state = {
        "Substation-Север": {
            "load_mw": 485,
            "max_mw": 500,
            "status": "CRITICAL",
            "voltage_kv": 218,
        },
        "Substation-Юг": {
            "load_mw": 140,
            "max_mw": 400,
            "status": "NORMAL",
            "voltage_kv": 222,
        },
        "Substation-Восток": {
            "load_mw": 200,
            "max_mw": 350,
            "status": "NORMAL",
            "voltage_kv": 220,
        },
    }

if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Инструменты (Tools), вызываемые моделью ---
def get_grid_telemetry():
    return json.dumps(st.session_state.grid_state)


def rebalance_load(source_substation: str, target_substation: str, mw: float):
    grid = st.session_state.grid_state
    if source_substation not in grid or target_substation not in grid:
        return json.dumps({"status": "error", "message": "Подстанция не найдена"})

    grid[source_substation]["load_mw"] -= mw
    grid[target_substation]["load_mw"] += mw

    for name, data in grid.items():
        ratio = data["load_mw"] / data["max_mw"]
        data["status"] = (
            "CRITICAL"
            if ratio >= 0.95
            else ("WARNING" if ratio >= 0.85 else "NORMAL")
        )

    return json.dumps(
        {
            "status": "success",
            "transferred_mw": mw,
            "source": source_substation,
            "target": target_substation,
        }
    )


tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_grid_telemetry",
            "description": "Получить текущую телеметрию нагрузки и напряжения подстанций.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rebalance_load",
            "description": "Перераспределить электрическую мощность (МВт) между узлами сети для устранения перегрузки.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_substation": {
                        "type": "string",
                        "description": "Имя перегруженной подстанции",
                    },
                    "target_substation": {
                        "type": "string",
                        "description": "Имя резервной подстанции",
                    },
                    "mw": {
                        "type": "number",
                        "description": "Передаваемая мощность в МВт",
                    },
                },
                "required": ["source_substation", "target_substation", "mw"],
            },
        },
    },
]

# Боковая панель SCADA-мониторинга
st.sidebar.title("⚡ SCADA Мониторинг сети")
for name, data in st.session_state.grid_state.items():
    ratio = data["load_mw"] / data["max_mw"]
    color = "red" if ratio >= 0.95 else ("orange" if ratio >= 0.85 else "green")
    st.sidebar.markdown(f"**{name}**")
    st.sidebar.progress(min(ratio, 1.0))
    st.sidebar.caption(
        f"Нагрузка: {data['load_mw']} / {data['max_mw']} МВт | Статус: :{color}[{data['status']}]"
    )
    st.sidebar.divider()

# Основное окно
st.title("🔋 Автономный AI-Диспетчер Энергосети")
st.caption(
    "Демонстрация Function Calling: агент считывает данные с датчиков и автоматически распределяет нагрузки."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_prompt = st.chat_input(
    "Пример: Оцени состояние энергосети и устрани аварийные нагрузки"
)

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    if not api_key or "sk-" not in api_key:
        st.error(
            "Укажите действующий OPENAI_API_KEY в файле .env (он станет доступен 23 сентября в 12:30)"
        )
        st.stop()

    client = OpenAI(api_key=api_key)

    api_messages = [
        {
            "role": "system",
            "content": "Ты дежурный инженер-диспетчер энергосистемы. Используй функции телеметрии и балансировки для предотвращения аварий.",
        }
    ] + [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        with st.status(
            "AI-Агент анализирует сеть и вызывает инструменты...",
            expanded=True,
        ) as status:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=api_messages,
                tools=tools_schema,
                tool_choice="auto",
            )
            response_msg = response.choices[0].message

            while response_msg.tool_calls:
                api_messages.append(response_msg)
                for tool_call in response_msg.tool_calls:
                    fn_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    st.write(f"⚙️ **Вызов инструмента:** `{fn_name}`")
                    st.json(args)

                    if fn_name == "get_grid_telemetry":
                        res = get_grid_telemetry()
                    elif fn_name == "rebalance_load":
                        res = rebalance_load(**args)
                    else:
                        res = json.dumps({"error": "Unknown function"})

                    api_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": res,
                        }
                    )

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=api_messages,
                    tools=tools_schema,
                )
                response_msg = response.choices[0].message

            status.update(label="Балансировка завершена!", state="complete")

        st.markdown(response_msg.content)
        st.session_state.messages.append(
            {"role": "assistant", "content": response_msg.content}
        )
        st.rerun()