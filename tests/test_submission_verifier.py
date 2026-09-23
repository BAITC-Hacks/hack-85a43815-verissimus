"""Adversarial checks of a submission with an otherwise consistent manifest."""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.verify_submission import COLUMNS, verify_submission


class SubmissionVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "submission.csv"
        start = datetime(2026, 1, 31, 19, tzinfo=timezone.utc)
        self.rows = []
        runs = []
        for turbine in ("turbine_1", "turbine_2"):
            self.rows.extend({"turbine": turbine, "timestamp": f"2026-02-01 {hour:02}:00:00",
                              "wind_speed_ms": 5, "temperature_c": -2, "predicted_power": .4}
                             for hour in range(24))
            runs.append({"turbine": turbine, "target_date": "2026-02-01", "status": "success",
                         "forecast_id": turbine + "_forecast", "horizon_hours": 24, "power_unit": "normalized",
                         "issue_time": "2026-01-31T18:00:00Z", "forecast_start": start.isoformat(),
                         "model": {"training_start": "2025-01-01T00:00:00Z",
                                   "training_end": "2026-01-31T17:00:00Z",
                                   "training_available_at": "2026-01-31T18:00:00Z"},
                         "weather": {"provider": "NOAA archive", "model": "GFS 0.25",
                                     "input_sha256": "a" * 64, "run_time": "2026-01-31T12:00:00Z",
                                     "issue_time": "2026-01-31T18:00:00Z",
                                     "available_at": "2026-01-31T15:00:00Z",
                                     "sources": [{"valid_time": (start + timedelta(hours=hour)).isoformat(),
                                                  "last_modified": "2026-01-31T15:00:00Z", "field": field}
                                                 for hour in range(24)
                                                 for field in ("inventory", "temperature", "u100", "v100")]}})
        self.manifest = {"schema_version": 1, "target_start": "2026-02-01", "target_end": "2026-02-01",
                         "horizon_hours": 24, "turbines": ["turbine_1", "turbine_2"],
                         "timestamp_timezone": "Asia/Almaty", "power_unit": "normalized", "row_count": 48,
                         "runs": runs}
        self.save()

    def save(self):
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(self.rows)
        raw = buffer.getvalue().encode("utf-8")
        self.path.write_bytes(raw)
        self.manifest["csv_sha256"] = hashlib.sha256(raw).hexdigest()
        self.path.with_suffix(".provenance.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def verify(self):
        return verify_submission(self.path, require_full_month=False)

    def test_complete_partial_fixture_passes_only_when_explicitly_allowed(self):
        result = self.verify()
        self.assertEqual(result["row_count"], 48)
        self.assertEqual(result["weather_sources_checked"], 192)
        self.assertFalse(result["full_february"])
        with self.assertRaisesRegex(ValueError, "full month"):
            verify_submission(self.path)

    def test_byte_tampering_is_rejected_before_semantic_checks(self):
        self.path.write_bytes(self.path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.verify()

    def test_duplicate_hour_rejected_even_with_updated_hash(self):
        self.rows[1] = dict(self.rows[0])
        self.save()
        with self.assertRaisesRegex(ValueError, "duplicated"):
            self.verify()

    def test_nan_and_out_of_range_power_rejected(self):
        for value in (float("nan"), float("inf"), -.01, 1.01):
            with self.subTest(value=value):
                self.rows[0]["predicted_power"] = value
                self.save()
                with self.assertRaises(ValueError):
                    self.verify()

    def test_missing_run_rejected_even_with_full_csv(self):
        self.manifest["runs"].pop()
        self.save()
        with self.assertRaisesRegex(ValueError, "Missing daily"):
            self.verify()

    def test_future_source_cannot_hide_behind_earlier_aggregate_timestamp(self):
        self.manifest["runs"][0]["weather"]["sources"][0]["last_modified"] = "2026-02-01T00:00:00Z"
        self.save()
        with self.assertRaisesRegex(ValueError, "Source timestamp"):
            self.verify()

    def test_missing_wind_component_rejected(self):
        self.manifest["runs"][0]["weather"]["sources"].pop()
        self.save()
        with self.assertRaisesRegex(ValueError, "weather source fields"):
            self.verify()

    def test_incomplete_training_hour_rejected(self):
        self.manifest["runs"][0]["model"]["training_end"] = "2026-01-31T18:00:00Z"
        self.save()
        with self.assertRaisesRegex(ValueError, "Training targets"):
            self.verify()

    def test_future_calibration_rejected(self):
        self.manifest["runs"][0]["model"]["calibration"] = {
            "applied": True, "available_at": "2026-02-01T00:00:00Z"}
        self.save()
        with self.assertRaisesRegex(ValueError, "Calibration"):
            self.verify()

    def test_archive_availability_must_equal_maximum_source_timestamp(self):
        self.manifest["runs"][0]["weather"]["available_at"] = "2026-01-31T18:00:00Z"
        self.save()
        with self.assertRaisesRegex(ValueError, "maximum source"):
            self.verify()

    def test_unit_prefix_does_not_validate_an_unknown_unit(self):
        for value in ("normalized_but_actually_MW", "normalized MW", "MW"):
            with self.subTest(value=value):
                self.manifest["power_unit"] = value
                self.save()
                with self.assertRaisesRegex(ValueError, "Normalized power"):
                    self.verify()
        self.manifest["power_unit"] = "normalized; same dimensionless scale as source CSV; no MW/MWh conversion"
        self.save()
        self.assertEqual(self.verify()["status"], "success")

    def test_weather_identity_and_hash_are_required(self):
        original = dict(self.manifest["runs"][0]["weather"])
        for key in ("provider", "model", "input_sha256"):
            with self.subTest(key=key):
                self.manifest["runs"][0]["weather"] = {k: v for k, v in original.items() if k != key}
                self.save()
                with self.assertRaisesRegex(ValueError, "Weather"):
                    self.verify()
        self.manifest["runs"][0]["weather"] = {**original, "input_sha256": "not-a-fingerprint"}
        self.save()
        with self.assertRaisesRegex(ValueError, "Weather input SHA"):
            self.verify()

    def test_applied_calibration_requires_complete_fit_and_artifact_evidence(self):
        calibration = {"applied": True, "status": "applied", "method": "ridge_affine", "version": "fixture-v1",
                       "fit_start": "2026-01-01T00:00:00Z", "fit_end": "2026-01-16T17:00:00Z",
                       "available_at": "2026-01-16T18:00:00Z", "artifact_sha256": "b" * 64,
                       "parameters": {"1-24": {"slope": .6, "intercept": -.06, "training_rows": 383},
                                      "25-48": {"slope": .57, "intercept": -.06, "training_rows": 359}}}
        self.manifest["runs"][0]["model"]["calibration"] = calibration
        self.save()
        self.assertEqual(self.verify()["status"], "success")
        for key in ("fit_start", "fit_end", "artifact_sha256", "parameters", "method", "version"):
            with self.subTest(key=key):
                self.manifest["runs"][0]["model"]["calibration"] = {k: v for k, v in calibration.items() if k != key}
                self.save()
                with self.assertRaisesRegex(ValueError, "Calibration"):
                    self.verify()


if __name__ == "__main__":
    unittest.main()
