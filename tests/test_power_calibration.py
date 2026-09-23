"""Temporal and provenance boundaries for frozen power correction."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from power_calibration import apply_calibration


class PowerCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "calibration.json"
        self.issue = pd.Timestamp("2026-01-16T18:00:00Z")
        self.index = pd.date_range(self.issue + pd.Timedelta(hours=1), periods=48, freq="h")
        self.raw = np.full(48, .8)
        self.config = {"schema_version": 1, "method": "ridge_affine", "version": "fixture-v1",
                       "promotion_rule_passed": True,
                       "base_model_version": "base-v2", "timezone_assumption": "Asia/Almaty",
                       "power_unit": "normalized", "fit_start": "2026-01-01T00:00:00Z",
                       "fit_end": "2026-01-16T17:00:00Z", "available_at": self.issue.isoformat(),
                       "training_source_sha256": {"turbine_1": "source-1"},
                       "parameters": {"turbine_1": {
                           "1-24": {"slope": .5, "intercept": -.1, "training_rows": 100},
                           "25-48": {"slope": .75, "intercept": -.2, "training_rows": 90}}}}
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.config), encoding="utf-8")

    def apply(self, **kwargs):
        args = dict(predicted=self.raw, valid_times=self.index, issue_time=self.issue,
                    turbine="turbine_1", source_sha256="source-1", base_model_version="base-v2",
                    timezone_assumption="Asia/Almaty", artifact_path=self.path)
        args.update(kwargs)
        return apply_calibration(**args)

    def test_applies_distinct_lead_bands_without_mutating_raw_values(self):
        values, meta = self.apply()
        np.testing.assert_allclose(values[:24], .3)
        np.testing.assert_allclose(values[24:], .4)
        np.testing.assert_allclose(self.raw, .8)
        self.assertTrue(meta["applied"])
        self.assertEqual(meta["training_rows"], 190)

    def test_missing_artifact_explicitly_keeps_original_model(self):
        values, meta = self.apply(artifact_path=self.path.with_name("absent.json"))
        np.testing.assert_array_equal(values, self.raw)
        self.assertEqual(meta["status"], "not_configured")
        self.assertFalse(meta["applied"])

    def test_one_second_before_training_availability_cannot_use_calibration(self):
        values, meta = self.apply(issue_time=self.issue - pd.Timedelta(seconds=1))
        np.testing.assert_array_equal(values, self.raw)
        self.assertEqual(meta["status"], "not_available_at_issue")

    def test_incomplete_last_training_hour_is_rejected(self):
        self.config["fit_end"] = self.issue.isoformat()
        self.save()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.apply()

    def test_fingerprint_timezone_and_model_version_must_match(self):
        for args in ({"source_sha256": "new-source"}, {"timezone_assumption": "UTC"},
                     {"base_model_version": "new-model"}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.apply(**args)

    def test_nan_negative_slope_and_insufficient_history_are_rejected(self):
        original = dict(self.config["parameters"]["turbine_1"]["1-24"])
        for change in ({"slope": float("nan")}, {"slope": -.1}, {"training_rows": 47}):
            with self.subTest(change=change):
                self.config["parameters"]["turbine_1"]["1-24"] = {**original, **change}
                self.save()
                with self.assertRaises(ValueError):
                    self.apply()

    def test_valid_coefficients_clip_to_source_scale(self):
        values, _ = self.apply(predicted=np.resize([0., 1.], 48))
        self.assertTrue(((values >= 0) & (values <= 1)).all())
        self.assertEqual(values[0], 0.)

    def test_other_decision_leads_explicitly_remain_uncalibrated(self):
        values, meta = self.apply(valid_times=self.index + pd.Timedelta(hours=12))
        np.testing.assert_array_equal(values, self.raw)
        self.assertEqual(meta["status"], "unsupported_lead_time")

    def test_artifact_change_is_visible_in_fingerprint(self):
        _, first = self.apply()
        self.config["parameters"]["turbine_1"]["1-24"]["slope"] = .55
        self.save()
        _, second = self.apply()
        self.assertNotEqual(first["artifact_sha256"], second["artifact_sha256"])

    def test_failed_evaluation_cannot_be_installed_as_a_calibration(self):
        self.config["promotion_rule_passed"] = False
        self.save()
        with self.assertRaisesRegex(ValueError, "evaluation rule"):
            self.apply()

    def test_late_weather_evidence_invalidates_calibration_availability(self):
        self.config["latest_weather_archive_available_at"] = "2026-01-16T18:01:00Z"
        self.save()
        with self.assertRaisesRegex(ValueError, "input evidence"):
            self.apply()


if __name__ == "__main__":
    unittest.main()
