"""Automatic refresh/recovery tests; no weather, API calls or real waiting."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from watch_forecast import main, watch_forecast


def result(identity):
    return {"status": "success", "forecast_id": identity,
            "issue_time": "2026-01-31T18:00:00+00:00",
            "weather": {"input_sha256": identity + "-weather"}}


class WatchForecastTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name) / "events.jsonl"
        self.sleep = Mock()
        self.emit = Mock()

    def run_watch(self, generate, cycles):
        return watch_forecast("turbine_1", "2026-02-01", 48,
                              issue_time="2026-01-31T18:00:00Z", cycles=cycles, interval=1,
                              output_path=self.output, generate=generate, sleep=self.sleep, emit=self.emit)

    def test_automatic_recheck_detects_new_version_and_suppresses_unchanged_console(self):
        generate = Mock(side_effect=[result("v1"), result("v1"), result("v2")])
        events = self.run_watch(generate, 3)
        self.assertEqual([event["event"] for event in events], ["changed", "unchanged", "changed"])
        self.assertTrue(events[0]["initial"])
        self.assertEqual(events[-1]["previous_forecast_id"], "v1")
        self.assertEqual(events[-1]["forecast_id"], "v2")
        self.assertEqual([call.kwargs["refresh"] for call in generate.call_args_list], [False, True, True])
        self.assertEqual({call.kwargs["issue_time"] for call in generate.call_args_list}, {"2026-01-31T18:00:00Z"})
        self.assertEqual(self.sleep.call_count, 2)
        self.assertEqual(self.emit.call_count, 2)
        saved = [json.loads(line) for line in self.output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(saved, events)

    def test_error_is_logged_and_next_iteration_recovers_without_resetting_identity(self):
        generate = Mock(side_effect=[result("v1"), {"status": "error", "message": "unavailable at issue time"}, result("v1")])
        events = self.run_watch(generate, 3)
        self.assertEqual([event["event"] for event in events], ["changed", "error", "unchanged"])
        self.assertIsNone(events[1]["forecast_id"])
        self.assertEqual(events[1]["previous_forecast_id"], "v1")
        self.assertEqual(events[2]["previous_forecast_id"], "v1")
        self.assertIn("unavailable", events[1]["message"])
        self.assertEqual(generate.call_count, 3)

    def test_exception_and_missing_identity_never_become_success(self):
        generate = Mock(side_effect=[ConnectionError("offline"), {"status": "success"}])
        events = self.run_watch(generate, 2)
        self.assertEqual([event["event"] for event in events], ["error", "error"])
        self.assertEqual(events[0]["error_type"], "ConnectionError")
        self.assertIsNone(events[1]["forecast_id"])

    def test_default_is_one_attempt_without_sleep(self):
        generate = Mock(return_value=result("v1"))
        events = watch_forecast(output_path=self.output, generate=generate, sleep=self.sleep, emit=self.emit)
        self.assertEqual(len(events), 1)
        generate.assert_called_once()
        self.assertFalse(generate.call_args.kwargs["refresh"])
        self.sleep.assert_not_called()

    def test_invalid_loop_arguments_rejected_before_running(self):
        generate = Mock()
        for options in [{"cycles": 0}, {"cycles": -1}, {"cycles": True}, {"cycles": 1.5},
                        {"interval": 0}, {"interval": float("inf")}, {"interval": True}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                watch_forecast(output_path=self.output, generate=generate, **options)
        generate.assert_not_called()

    def test_cli_aliases_and_error_exit_status(self):
        with patch("watch_forecast.watch_forecast", return_value=[{"event": "error"}]) as run:
            code = main(["--turbine_id", "turbine_2", "--forecast-date", "2026-02-02",
                         "--horizon", "24", "--cycles", "2", "--interval", "3", "--output", str(self.output)])
        self.assertEqual(code, 1)
        self.assertEqual(run.call_args.args[:3], ("turbine_2", "2026-02-02", 24))
        self.assertEqual(run.call_args.kwargs["cycles"], 2)
        self.assertEqual(run.call_args.kwargs["interval"], 3)


if __name__ == "__main__":
    unittest.main()
