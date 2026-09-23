"""Manual live check: python scripts/smoke_agent.py --run-live

Makes at most three OpenAI requests, with no automatic retries, a 30 second
timeout per request and <=700 completion tokens per request. It calls the real
app.run_agent and tool dispatcher. It does not activate or redeem credits.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class SessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class BoundedClient:
    """Preserve the SDK result while enforcing a hard limit around app calls."""
    def __init__(self, client):
        self.client = client
        self.requests = 0
        self.models = []
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        if self.requests >= 3:
            raise RuntimeError("Live smoke request budget exceeded")
        self.requests += 1
        kwargs.pop("max_tokens", None)
        kwargs["max_completion_tokens"] = 700
        response = self.client.chat.completions.create(**kwargs)
        model = getattr(response, "model", None)
        if model and model not in self.models:
            self.models.append(model)
        usage = getattr(response, "usage", None)
        if usage is not None:
            for key in self.usage:
                self.usage[key] += getattr(usage, key, 0) or 0
        return response


def _redact(text, api_key):
    text = str(text)
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    return re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[REDACTED]", text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-live", action="store_true", help="Explicitly enable up to three billable API requests")
    parser.add_argument("--scenario", choices=("inspect", "forecast"), default="inspect",
                        help="Inspect a saved forecast or let the LLM request a new forecast")
    args = parser.parse_args()
    if not args.run_live:
        parser.error("Pass --run-live only when a live API check is intended.")

    import dotenv
    dotenv.load_dotenv(ROOT / ".env", override=False)
    from openai import OpenAI
    import app
    from agent_payload import compact_tool_result

    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    artifact = {"model": model, "scenario": args.scenario, "usage": {}, "used_tools": [], "status": "error",
                "actual_forecast_id": None, "response": ""}
    bounded, state, raw_client = None, SessionState(), None
    try:
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        selected = {"turbine_id": "turbine_1", "forecast_date": "2026-02-01",
                    "horizon_hours": 48, "issue_time": None, "refresh": False}
        forecast = None
        if args.scenario == "inspect":
            forecast = app.forecast_engine.generate_agent_forecast(**selected)
            if compact_tool_result(forecast).get("status") != "success":
                raise RuntimeError("Forecast validation failed")
        fake_st = SimpleNamespace(session_state=state, warning=lambda *_args, **_kwargs: None)
        raw_client = OpenAI(api_key=api_key, timeout=30.0, max_retries=0)
        bounded = BoundedClient(raw_client)
        with patch.object(app, "st", fake_st):
            app.initialize_state()
            if forecast is not None:
                app.accept_result(forecast, selected)
            command = ("Вызови inspect_forecast для сохраненного прогноза. " if forecast is not None else
                       "Вызови generate_agent_forecast и рассчитай новый прогноз с выбранными параметрами. ")
            state.messages.append({"role": "user", "content": (
                command + "Затем кратко укажи турбину, "
                "период, момент решения, источник и выпуск погоды, среднюю нормализованную "
                "мощность. Объясни ограничения часового пояса и проверки точности. "
                "После успешного вызова не делай дополнительных вызовов инструментов.")})
            reply = app.run_agent(bounded, selected)
        used = [event for event in state.tool_trace if event.get("action") in ("inspect_forecast", "generate_agent_forecast")]
        artifact["used_tools"] = [event["action"] for event in used]
        latest = state.get("latest_forecast")
        if latest is not None:
            artifact["actual_forecast_id"] = latest.get("forecast_id")
        artifact["response"] = _redact(reply, api_key)
        if not used or any(event.get("status") != "success" for event in used) or not reply:
            raise RuntimeError("Live agent tool check failed")
        if args.scenario == "forecast" and "generate_agent_forecast" not in artifact["used_tools"]:
            raise RuntimeError("Agent did not request the forecast tool")
        artifact["status"] = "success"
    except Exception as exc:
        # Do not persist raw provider error bodies, headers, URLs or credentials.
        artifact["status"] = "error"
        artifact["response"] = "Live check failed: " + type(exc).__name__
        artifact["used_tools"] = [event["action"] for event in state.get("tool_trace", [])]
    finally:
        if bounded:
            artifact["usage"] = {**bounded.usage, "requests": bounded.requests}
            artifact["model"] = ", ".join(bounded.models) or model
        if raw_client:
            raw_client.close()
        path = ROOT / "artifacts" / ("live_smoke.json" if args.scenario == "inspect" else "live_smoke_forecast.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"status": artifact["status"], "model": artifact["model"],
                          "usage": artifact["usage"], "used_tools": artifact["used_tools"],
                          "artifact": str(path)}, ensure_ascii=False))
    return 0 if artifact["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
