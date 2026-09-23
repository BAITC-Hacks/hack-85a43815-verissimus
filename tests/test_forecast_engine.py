"""Offline tests for forecast boundaries, units and durable agent revisions."""
from collections import OrderedDict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import forecast_engine as engine


class MockModel:
    def predict(self, frame):
        return np.resize(np.array([-.1, .25, 1.2]), len(frame))


class ForecastEngineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.artifacts = Path(self.temporary.name) / "artifacts"
        self.start = pd.Timestamp("2026-01-31T19:00:00Z")
        self.issue = pd.Timestamp("2026-01-31T18:00:00Z")
        history_index = pd.date_range("2026-01-27T00:00:00Z", periods=150, freq="h")
        self.history = pd.DataFrame({"wind_speed": 5., "temperature": -2., "target_power": .25}, index=history_index)
        self.history = engine._features(self.history)
        self.history.attrs = {"source_sha256": "a" * 64, "data_quality": {"complete_hours": len(self.history)}}
        self.load = self.start_patch(patch.object(engine, "load_and_preprocess_turbine_data", return_value=self.history))
        self.train = self.start_patch(patch.object(engine, "train_wind_model", return_value=(MockModel(), {"scope": "observed_weather_power_curve"}, .01)))
        self.weather = self.start_patch(patch("weather_archive.fetch_weather", side_effect=self.fake_weather))
        self.start_patch(patch.object(engine, "ARTIFACT_DIR", self.artifacts))
        self.start_patch(patch.object(engine, "CALIBRATION_FILE", Path(self.temporary.name) / "no-calibration.json"))
        self.start_patch(patch.object(engine, "_MODEL_CACHE", OrderedDict()))

    def start_patch(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def fake_weather(self, lat, lon, expected, issued, refresh=False):
        frame = pd.DataFrame({"wind_speed": 5., "temperature": -2.}, index=expected)
        provenance = {"run_time": "2026-01-31T12:00:00Z", "available_at": "2026-01-31T16:00:00Z",
                      "provider": "test forecast archive", "input_sha256": "e" * 64,
                      "retrieved_at": "2026-09-23T00:00:00Z"}
        return frame, provenance

    def forecast(self, **kwargs):
        args = dict(turbine_id="turbine_1", forecast_date="2026-02-01", horizon_hours=48)
        args.update(kwargs)
        return engine.generate_agent_forecast(**args)

    def test_training_cutoff_accepts_only_hours_complete_at_decision(self):
        result = self.forecast()
        self.assertEqual(result["status"], "success", result)
        training = self.train.call_args.args[0]
        self.assertEqual(training.index[-1], self.issue - pd.Timedelta(hours=1))
        self.assertTrue((training.index + pd.Timedelta(hours=1) <= self.issue).all())
        self.assertNotIn(self.issue, training.index)
        self.assertEqual(pd.Timestamp(result["model"]["training_available_at"]), self.issue)
        self.assertEqual(pd.Timestamp(result["issue_time"]), self.issue)
        self.assertEqual(pd.Timestamp(result["forecast_start"]), self.start)

    def test_normalized_prediction_clips_once_and_never_divides_by_capacity(self):
        result = self.forecast()
        self.assertEqual(result["status"], "success", result)
        values = [p["predicted_power"] for p in result["forecast_sample"]]
        self.assertEqual(values[:3], [0., .25, 1.])
        self.assertEqual(len(values), 48)
        self.assertEqual(result["power_unit"], "normalized")
        self.assertAlmostEqual(result["summary"]["normalized_power_hours"], sum(values))
        self.assertTrue(all(0 <= power <= 1 for power in values))
        self.assertNotIn("MW", json.dumps(result))

    def test_one_weather_row_cannot_succeed_for_48_hours(self):
        def incomplete(*args, **kwargs):
            frame, meta = self.fake_weather(*args, **kwargs)
            return frame.iloc[:1], meta
        self.weather.side_effect = incomplete
        result = self.forecast()
        self.assertEqual(result["status"], "error", result)
        self.assertNotIn("forecast_sample", result)
        self.assertFalse((self.artifacts / "runs").exists())

    def test_nan_weather_cannot_become_a_zero_forecast(self):
        def nan_weather(*args, **kwargs):
            frame, meta = self.fake_weather(*args, **kwargs)
            frame.iloc[5, 0] = np.nan
            return frame, meta
        self.weather.side_effect = nan_weather
        result = self.forecast()
        self.assertEqual(result["status"], "error", result)
        self.assertNotIn("summary", result)

    def test_publication_after_issue_is_rejected(self):
        def future_weather(*args, **kwargs):
            frame, meta = self.fake_weather(*args, **kwargs)
            meta["available_at"] = "2026-01-31T18:00:01Z"
            return frame, meta
        self.weather.side_effect = future_weather
        result = self.forecast()
        self.assertEqual(result["status"], "error", result)

    def test_initialization_after_issue_is_rejected(self):
        def future_weather(*args, **kwargs):
            frame, meta = self.fake_weather(*args, **kwargs)
            meta["run_time"] = "2026-02-01T00:00:00Z"
            return frame, meta
        self.weather.side_effect = future_weather
        self.assertEqual(self.forecast()["status"], "error")

    def test_invalid_turbine_horizon_and_issue_fail_before_reading_data(self):
        invalid = [{"turbine_id": "../secret"}, {"horizon_hours": True}, {"horizon_hours": 25},
                   {"horizon_hours": 48.0}, {"issue_time": self.start},
                   {"forecast_date": "2026-02-01T00:30:00"}]
        for args in invalid:
            with self.subTest(args=args):
                self.assertEqual(self.forecast(**args)["status"], "error")
        self.load.assert_not_called()
        self.weather.assert_not_called()

    def test_unchanged_refresh_retains_id_and_changed_weather_creates_audited_revision(self):
        first = self.forecast()
        self.assertEqual(first["status"], "success", first)
        def retrieved_later(*args, **kwargs):
            frame, meta = self.fake_weather(*args, **kwargs)
            meta["retrieved_at"] = "2026-09-23T01:00:00Z"
            return frame, meta
        self.weather.side_effect = retrieved_later
        same = self.forecast(refresh=True)
        self.assertEqual(same["forecast_id"], first["forecast_id"])
        self.assertFalse(same["changed_from_previous"])
        def changed_weather(*args, **kwargs):
            frame, meta = self.fake_weather(*args, **kwargs)
            frame.iloc[5, 0] += 1
            meta["input_sha256"] = "f" * 64
            return frame, meta
        self.weather.side_effect = changed_weather
        changed = self.forecast(refresh=True)
        self.assertNotEqual(changed["forecast_id"], first["forecast_id"])
        self.assertTrue(changed["changed_from_previous"])
        self.assertEqual(changed["previous_forecast_id"], first["forecast_id"])
        events = [json.loads(line) for line in (self.artifacts / "agent_audit.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(events), 3)
        self.assertEqual(events[-1]["action"], "refresh")
        self.assertTrue(events[-1]["changed"])
        self.assertEqual(events[-1]["previous_forecast_id"], first["forecast_id"])
        self.assertEqual(len(list((self.artifacts / "runs").glob("*.json"))), 2)
        pointer = json.loads(next((self.artifacts / "requests").glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(pointer["forecast_id"], changed["forecast_id"])

    def test_changed_archive_input_hash_creates_new_auditable_identity(self):
        first = self.forecast()
        def new_input_hash(*args, **kwargs):
            frame, meta = self.fake_weather(*args, **kwargs)
            meta["input_sha256"] = "f" * 64
            return frame, meta
        self.weather.side_effect = new_input_hash
        changed = self.forecast(refresh=True)
        self.assertNotEqual(changed["forecast_id"], first["forecast_id"])

    def test_model_returning_one_value_for_48_hours_is_an_error(self):
        class IncompleteModel:
            def predict(self, _):
                return np.array([.5])
        self.train.return_value = (IncompleteModel(), {"scope": "observed_weather_power_curve"}, 1.)
        self.assertEqual(self.forecast()["status"], "error")


class ScadaClockTests(unittest.TestCase):
    def test_2024_almaty_clock_overlap_is_explicitly_removed(self):
        # 23:00..23:50 on Feb29 occurs twice during UTC+6 -> UTC+5. With no
        # offset or repetition marker, either interpretation would be invented.
        timestamps = pd.date_range("2024-02-29T22:00:00", periods=24, freq="10min")
        raw = pd.DataFrame({"time": timestamps, "wind_speed": 5., "target_power": .3, "temperature": 2.})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scada.csv"
            raw.to_csv(path, index=False)
            data = engine.load_and_preprocess_turbine_data(path, site_timezone="Asia/Almaty")
        self.assertEqual(data.attrs["data_quality"]["ambiguous_or_nonexistent_clock_rows_removed"], 6)
        self.assertEqual(len(data), 3)
        self.assertEqual(data.index[0], pd.Timestamp("2024-02-29T16:00:00Z"))
        self.assertEqual(data.index[-1], pd.Timestamp("2024-02-29T20:00:00Z"))
        self.assertFalse(data.index.has_duplicates)

    def test_ambiguous_forecast_time_requires_an_explicit_offset(self):
        result = engine.generate_agent_forecast("turbine_1", "2024-02-29T23:00:00", 24)
        self.assertEqual(result["status"], "error")
        one = engine._as_utc("2024-02-29T23:00:00+06:00")
        two = engine._as_utc("2024-02-29T23:00:00+05:00")
        self.assertEqual(two - one, pd.Timedelta(hours=1))


if __name__ == "__main__":
    unittest.main()
