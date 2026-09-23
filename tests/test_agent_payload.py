"""The LLM gets exact forecast numbers and limitations, not download inventories."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from agent_payload import compact_tool_result
from scripts.smoke_agent import BoundedClient, _redact


def forecast_fixture():
    start = datetime(2026, 1, 31, 19, tzinfo=timezone.utc)
    return {
        "status": "success", "forecast_id": "fixture-id", "turbine": "turbine_1",
        "horizon_hours": 48, "power_unit": "normalized", "timezone_assumption": "Asia/Almaty",
        "issue_time": (start - timedelta(hours=1)).isoformat(), "forecast_start": start.isoformat(),
        "period": f"{start.isoformat()} — {(start + timedelta(hours=47)).isoformat()}",
        "summary": {"avg_predicted_power": 0.31234567890123, "max_predicted_power": 0.8,
                    "min_predicted_power": 0.01, "low_generation_hours": 2,
                    "normalized_power_hours": 14.99259258725904, "analysis": "Low power is not proof of calm."},
        "assumptions": ["Timezone is assumed.", "Power normalization is unknown."],
        "warnings": ["Historical replay, not February accuracy."],
        "weather": {"provider": "NOAA archive", "model": "GFS", "input_sha256": "weather-hash",
                    "run_time": "2026-01-31T12:00:00+00:00", "available_at": "2026-01-31T16:00:00+00:00",
                    "availability_evidence": "HTTP Last-Modified; not an independent first-publication log.",
                    "grid_cell": {"latitude": 43.75, "longitude": 78.5},
                    "limitations": ["Nearest grid point, not an on-site sensor."],
                    "sources": [{"url": "https://example.test/large-inventory", "byte_range": "0-20000000",
                                 "inventory": "irrelevant archive metadata " * 100,
                                 "warnings": ["File availability is an archive observation."]} for _ in range(144)]},
        "validation_metrics": {"scope": "observed_weather_power_curve", "Validation_R2": 0.98,
                               "Validation_MAE": 0.023, "Validation_RMSE": 0.032,
                               "description": "Known observed weather; not 24–48 hour forecast skill."},
        "model": {"version": "fixture-model", "training_hours": 10000,
                  "training_available_at": "2026-01-31T18:00:00+00:00",
                  "calibration": {"applied": True, "method": "linear", "parameters": {"alpha": 0.23456789123, "beta": 0.62},
                                  "fit_start": "2026-01-01", "fit_end": "2026-01-15",
                                  "limitations": ["Short calibration sample."]}},
        "forecast_sample": [{"time": (start + timedelta(hours=h)).isoformat(), "wind_speed": 5.123456789,
                             "temperature": -2.3456789, "predicted_power": 0.31234567890123} for h in range(48)],
    }


class CompactToolResultTests(unittest.TestCase):
    def test_reduces_payload_without_rounding_or_dropping_hours(self):
        original = forecast_fixture()
        before = deepcopy(original)
        result = compact_tool_result(original)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["forecast_sample"], original["forecast_sample"])
        self.assertEqual(result["summary"], original["summary"])
        self.assertEqual(original, before)
        self.assertLess(len(json.dumps(result)), len(json.dumps(original)) / 10)
        self.assertNotIn("sources", result["weather"])

    def test_preserves_error_and_unavailable_status_without_unknown_fields(self):
        for status in ("error", "unavailable"):
            original = {"status": status, "message": "Forecast not available before cutoff.",
                        "error_type": "WeatherArchiveError", "api_key": "secret-sentinel"}
            result = compact_tool_result(original)
            self.assertEqual(result["status"], status)
            self.assertEqual(result["message"], original["message"])
            self.assertEqual(result["error_type"], original["error_type"])
            self.assertNotIn("secret-sentinel", json.dumps(result))

    def test_preserves_scope_warnings_assumptions_and_calibration(self):
        original = forecast_fixture()
        result = compact_tool_result(original)
        self.assertEqual(result["validation_metrics"], original["validation_metrics"])
        self.assertEqual(result["warnings"], original["warnings"])
        self.assertEqual(result["assumptions"], original["assumptions"])
        self.assertEqual(result["weather"]["availability_evidence"], original["weather"]["availability_evidence"])
        self.assertEqual(result["weather"]["limitations"], original["weather"]["limitations"])
        self.assertEqual(result["model"]["calibration"], original["model"]["calibration"])
        self.assertEqual(len(result["weather"]["source_notices"]), 1)

    def test_whitelist_blocks_secrets_in_unexpected_nested_fields(self):
        original = forecast_fixture()
        sentinel = "secret-sentinel-not-for-api"
        original["api_key"] = sentinel
        original["messages"] = [{"role": "user", "content": sentinel}]
        original["weather"]["authorization"] = sentinel
        original["weather"]["grid_cell"]["api_key"] = sentinel
        original["model"]["source_url"] = sentinel
        original["model"]["calibration"]["api_key"] = sentinel
        original["model"]["calibration"]["parameters"]["api_key"] = sentinel
        original["forecast_sample"][0]["raw_secret"] = sentinel
        result = compact_tool_result(original)
        self.assertEqual(result["status"], "success")
        self.assertNotIn(sentinel, json.dumps(result))
        self.assertNotIn("https://", json.dumps(result))
        self.assertNotIn("byte_range", json.dumps(result))

    def test_malformed_or_incomplete_success_becomes_error(self):
        for invalid in (None, [], {}, {"status": "success"}, {"status": "unknown"}, {"status": "error"}):
            with self.subTest(invalid=invalid):
                self.assertEqual(compact_tool_result(invalid)["status"], "error")
        mutations = [
            lambda f: f["forecast_sample"].pop(),
            lambda f: f["forecast_sample"][0].update(predicted_power=float("nan")),
            lambda f: f["forecast_sample"][0].update(temperature=float("inf")),
            lambda f: f["forecast_sample"][0].update(predicted_power=True),
            lambda f: f["forecast_sample"][0].update(time="2026-02-01T00:00:00"),
            lambda f: f["forecast_sample"][1].update(time=f["forecast_sample"][0]["time"]),
            lambda f: f["weather"].update(available_at="2026-02-01T20:00:00+00:00"),
            lambda f: f["validation_metrics"].pop("scope"),
            lambda f: f["summary"].update(avg_predicted_power=float("nan")),
            lambda f: f.update(power_unit="MW"),
            lambda f: f.update(horizon_hours=True),
        ]
        for mutation in mutations:
            original = forecast_fixture()
            mutation(original)
            with self.subTest(mutation=mutation):
                self.assertEqual(compact_tool_result(original)["status"], "error")

    def test_new_summary_schema_does_not_silently_drop_statistics(self):
        original = forecast_fixture()
        original["summary"]["unexpected_new_statistic"] = 7.3
        self.assertEqual(compact_tool_result(original)["status"], "error")

    def test_preserves_per_horizon_calibration_parameters_with_whitelist(self):
        original = forecast_fixture()
        parameters = {"1-24": {"slope": 0.7456789123, "intercept": 0.023456789, "training_rows": 168},
                      "25-48": {"slope": 0.4567891234, "intercept": 0.134567891, "training_rows": 144}}
        original["model"]["calibration"]["parameters"] = deepcopy(parameters)
        original["model"]["calibration"]["parameters"]["1-24"]["api_key"] = "secret-sentinel"
        original["model"]["calibration"]["parameters"]["unknown_band"] = "secret-sentinel"
        result = compact_tool_result(original)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["model"]["calibration"]["parameters"], parameters)
        self.assertNotIn("secret-sentinel", json.dumps(result))
        for bad in (float("nan"), "0.2", True):
            broken = deepcopy(original)
            broken["model"]["calibration"]["parameters"]["1-24"]["slope"] = bad
            self.assertEqual(compact_tool_result(broken)["status"], "error")
        broken = deepcopy(original)
        broken["model"]["calibration"]["parameters"]["25-48"]["training_rows"] = 0
        self.assertEqual(compact_tool_result(broken)["status"], "error")


class LiveSmokeGuardTests(unittest.TestCase):
    def test_enforces_request_and_completion_limits_without_live_requests(self):
        sdk = Mock()
        sdk.chat.completions.create.return_value = SimpleNamespace(
            model="test-model", usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120))
        client = BoundedClient(sdk)
        for _ in range(3):
            client.chat.completions.create(model="test-model", messages=[], max_tokens=9000, max_completion_tokens=8000)
        with self.assertRaises(RuntimeError):
            client.chat.completions.create(model="test-model", messages=[])
        self.assertEqual(sdk.chat.completions.create.call_count, 3)
        self.assertEqual(client.requests, 3)
        self.assertEqual(client.usage, {"prompt_tokens": 300, "completion_tokens": 60, "total_tokens": 360})
        for call in sdk.chat.completions.create.call_args_list:
            self.assertNotIn("max_tokens", call.kwargs)
            self.assertEqual(call.kwargs["max_completion_tokens"], 700)

    def test_redacts_actual_key_and_openai_key_like_strings(self):
        text = "prefix actual-test-secret sk-proj-abcdefghijklmnopqrst end"
        result = _redact(text, "actual-test-secret")
        self.assertNotIn("actual-test-secret", result)
        self.assertNotIn("sk-proj-", result)
        self.assertTrue(result.endswith(" end"))


if __name__ == "__main__":
    unittest.main()
