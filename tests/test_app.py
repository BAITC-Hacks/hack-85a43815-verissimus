"""UI and agent regression tests. No network, API charges, or .env reads."""

from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

import app


def successful_result(turbine_id="turbine_1", forecast_date="2026-02-01", horizon_hours=48,
                      issue_time=None, refresh=False):
    start = datetime.fromisoformat(forecast_date).replace(tzinfo=timezone(timedelta(hours=5)))
    value = 0.35 if refresh else 0.3
    return {
        "status": "success", "forecast_id": "forecast-updated" if refresh else "forecast-original",
        "turbine": turbine_id, "horizon_hours": horizon_hours, "power_unit": "normalized",
        "issue_time": issue_time or (start - timedelta(hours=1)).isoformat(),
        "period": [start.isoformat(), (start + timedelta(hours=horizon_hours - 1)).isoformat()],
        "timezone_assumption": "Asia/Almaty (assumed)",
        "weather": {"provider": "test fixture", "model": "mock model",
                    "run_time": "2026-01-31T12:00:00+00:00", "available_at": "2026-01-31T17:00:00+00:00",
                    "input_sha256": "fixed-weather-input"},
        "validation_metrics": {"Validation_R2": 0.9, "Validation_MAE": 0.04,
                               "Validation_RMSE": 0.06, "scope": "observed_weather_power_curve"},
        "summary": {"avg_predicted_power": value, "max_predicted_power": value,
                    "min_predicted_power": value, "low_generation_hours": 0,
                    "normalized_power_hours": value * horizon_hours},
        "forecast_sample": [{"time": (start + timedelta(hours=h)).isoformat(),
                             "wind_speed": 5.0, "temperature": -2.0,
                             "predicted_power": value} for h in range(horizon_hours)],
    }


class Message:
    def __init__(self, name=None, arguments="{}", content=None):
        self.content = content
        self.tool_calls = [] if name is None else [SimpleNamespace(
            id="call-test", function=SimpleNamespace(name=name, arguments=arguments))]

    def model_dump(self, exclude_none=True):
        return {"role": "assistant", "tool_calls": [
            {"id": call.id, "type": "function", "function": {
                "name": call.function.name, "arguments": call.function.arguments}}
            for call in self.tool_calls]}


def response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.start_patch(patch.dict("os.environ", {"ENERGYAI_LOAD_DOTENV": "0", "OPENAI_API_KEY": ""}))
        self.start_patch(patch.object(app, "PROJECT_DIR", Path(self.temp.name)))
        self.dotenv = self.start_patch(patch.object(app.dotenv, "load_dotenv"))
        self.forecast = self.start_patch(patch.object(app.forecast_engine, "generate_agent_forecast", side_effect=successful_result))
        self.client = Mock()
        self.openai = self.start_patch(patch.object(app, "OpenAI", return_value=self.client))
        self.ui = AppTest.from_string("import app\napp.main()", default_timeout=15).run()
        self.assertFalse(self.ui.exception)

    def start_patch(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def test_initial_load_needs_no_key_or_network_and_does_not_read_dotenv(self):
        self.forecast.assert_not_called()
        self.openai.assert_not_called()
        self.dotenv.assert_not_called()
        self.assertTrue(self.ui.chat_input[0].disabled)
        self.assertTrue(self.ui.button(key="refresh_forecast").disabled)

    def test_sidebar_values_reach_engine_and_units_are_dimensionless(self):
        self.ui.selectbox(key="selected_turbine").set_value("turbine_2")
        self.ui.selectbox(key="selected_horizon").set_value(24)
        self.ui.date_input(key="selected_date").set_value(date(2026, 2, 3))
        self.ui.text_input(key="selected_issue_time").set_value("2026-02-02T23:00:00+05:00")
        self.ui.button(key="generate_forecast").click().run()
        self.assertFalse(self.ui.exception)
        self.forecast.assert_called_once_with(turbine_id="turbine_2", forecast_date="2026-02-03",
            horizon_hours=24, issue_time="2026-02-02T23:00:00+05:00", refresh=False)
        self.assertEqual(len(self.ui.session_state.latest_forecast["forecast_sample"]), 24)
        self.assertEqual(self.ui.metric[0].value, "0.3000")
        self.assertTrue(any("безразмерной" in item.value for item in self.ui.info))
        frame = self.ui.dataframe[0].value
        self.assertEqual(frame.iloc[0]["time_local"], "2026-02-03T00:00:00+05:00")
        self.assertEqual(frame.iloc[0]["time_utc"], "2026-02-02T19:00:00+00:00")
        self.openai.assert_not_called()

    def test_engine_failure_keeps_successful_forecast_and_does_not_show_zero(self):
        self.ui.button(key="generate_forecast").click().run()
        self.forecast.side_effect = RuntimeError("Weather unavailable")
        self.ui.button(key="generate_forecast").click().run()
        self.assertFalse(self.ui.exception)
        self.assertIn("Weather unavailable", self.ui.error[0].value)
        self.assertEqual(self.ui.session_state.latest_forecast["forecast_id"], "forecast-original")
        self.assertEqual(self.ui.metric[0].value, "0.3000")
        self.assertEqual(len(self.ui.session_state.tool_trace), 2)

    def test_refresh_preserves_original_issue_and_compares_versions(self):
        self.ui.button(key="generate_forecast").click().run()
        original = deepcopy(self.ui.session_state.latest_forecast)
        self.ui.selectbox(key="selected_turbine").set_value("turbine_2")
        self.ui.button(key="refresh_forecast").click().run()
        self.assertFalse(self.ui.exception)
        self.forecast.assert_called_with(turbine_id="turbine_1", forecast_date="2026-02-01",
            horizon_hours=48, issue_time="2026-01-31T23:00:00+05:00", refresh=True)
        comparison = self.ui.session_state.refresh_comparison
        self.assertTrue(comparison["hourly_data_changed"])
        self.assertAlmostEqual(comparison["max_absolute_power_change"], 0.05)
        self.assertEqual(self.ui.session_state.forecast_history["forecast-original"], original)

    def test_retrieval_metadata_does_not_count_as_a_weather_change(self):
        self.ui.button(key="generate_forecast").click().run()
        same_weather = deepcopy(self.ui.session_state.latest_forecast)
        same_weather["weather"]["retrieved_at"] = "2026-09-23T12:00:00+00:00"
        self.forecast.side_effect = None
        self.forecast.return_value = same_weather
        self.ui.button(key="refresh_forecast").click().run()
        self.assertFalse(self.ui.exception)
        comparison = self.ui.session_state.refresh_comparison
        self.assertFalse(comparison["weather_changed"])
        self.assertFalse(comparison["hourly_data_changed"])

    def test_invalid_issue_time_is_visible_without_calling_engine(self):
        self.ui.text_input(key="selected_issue_time").set_value("2026-01-31T23:00:00")
        self.ui.button(key="generate_forecast").click().run()
        self.assertFalse(self.ui.exception)
        self.assertIn("UTC-смещение", self.ui.error[0].value)
        self.forecast.assert_not_called()
        self.assertFalse(self.ui.metric)

    def test_structured_engine_error_displays_its_reason(self):
        self.forecast.side_effect = None
        self.forecast.return_value = {"status": "error", "message": "Required forecast run unavailable"}
        self.ui.button(key="generate_forecast").click().run()
        self.assertFalse(self.ui.exception)
        self.assertIn("Required forecast run unavailable", self.ui.error[0].value)
        self.assertIsNone(self.ui.session_state.latest_forecast)

    def enable_chat(self):
        self.start_patch(patch.dict("os.environ", {"OPENAI_API_KEY": "unit-test-placeholder"}))
        self.ui.run()

    def test_chat_uses_sidebar_context_recovers_bad_json_and_stores_trace(self):
        self.enable_chat()
        self.ui.selectbox(key="selected_turbine").set_value("turbine_2")
        self.ui.selectbox(key="selected_horizon").set_value(24)
        self.client.chat.completions.create.side_effect = [
            response(Message("generate_agent_forecast", "not-json")),
            response(Message("generate_agent_forecast", "{}")),
            response(Message(content="Получен прогноз в нормализованной шкале."))]
        self.ui.chat_input[0].set_value("Рассчитай выбранный прогноз").run()
        self.assertFalse(self.ui.exception)
        self.forecast.assert_called_once_with(turbine_id="turbine_2", forecast_date="2026-02-01",
            horizon_hours=24, issue_time=None, refresh=False)
        messages = self.client.chat.completions.create.call_args_list[0].kwargs["messages"]
        self.assertIn('"turbine_id": "turbine_2"', messages[0]["content"])
        self.assertIn('"horizon_hours": 24', messages[0]["content"])
        self.assertEqual([event["status"] for event in self.ui.session_state.tool_trace], ["error", "success"])
        audit = Path(self.temp.name) / "artifacts" / "chat_trace.jsonl"
        events = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(events), 2)
        self.assertNotIn("unit-test-placeholder", audit.read_text(encoding="utf-8"))

    def test_chat_has_a_finite_tool_loop(self):
        self.enable_chat()
        self.client.chat.completions.create.return_value = response(Message("generate_agent_forecast"))
        self.ui.chat_input[0].set_value("Рассчитай прогноз").run()
        self.assertFalse(self.ui.exception)
        self.assertEqual(self.client.chat.completions.create.call_count, 6)
        self.assertEqual(self.forecast.call_count, 6)
        self.assertIn("лимит", self.ui.session_state.messages[-1]["content"])

    def test_inspection_uses_snapshot_without_recomputing(self):
        self.ui.button(key="generate_forecast").click().run()
        self.enable_chat()
        self.client.chat.completions.create.side_effect = [
            response(Message("inspect_forecast")), response(Message(content="Сохраненный результат."))]
        self.ui.chat_input[0].set_value("Покажи сохраненный прогноз").run()
        self.assertFalse(self.ui.exception)
        self.assertEqual(self.forecast.call_count, 1)
        self.assertEqual(self.ui.session_state.tool_trace[-1]["action"], "inspect_forecast")


class ValidationTests(unittest.TestCase):
    def test_rejects_unknown_tools_parameters_and_types(self):
        base = {"turbine_id": "turbine_1", "forecast_date": "2026-02-01"}
        for override in ({"horizon_hours": True}, {"horizon_hours": "48"}, {"horizon_hours": 1},
                         {"turbine_id": "../data"}, {"refresh": "false"}, {"issue_time": 0}, {"unknown": 1}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                app.validate_request({**base, **override})


if __name__ == "__main__":
    unittest.main()
