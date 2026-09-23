"""Batch acceptance checks use fabricated values, never real weather or LLM calls."""

import copy
import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import batch_forecast as batch


def good_forecast(**kwargs):
    start = datetime.fromisoformat(kwargs["forecast_date"]).replace(tzinfo=ZoneInfo(batch.LOCAL_TIMEZONE))
    return {
        "status": "success",
        "forecast_id": f"mock-{kwargs['turbine_id']}-{kwargs['forecast_date']}",
        "issue_time": kwargs["issue_time"],
        "weather": {"source": "unit-test fixture", "sha256": "mock-hash"},
        "forecast_sample": [
            {
                "time": (start + timedelta(hours=i)).astimezone(timezone.utc).isoformat(),
                "wind_speed": 4.0,
                "temperature": -5.0,
                "predicted_power": 0.2,
            }
            for i in range(24)
        ],
    }


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name) / "submission.csv"
        self.manifest = self.output.with_suffix(".provenance.json")

    def tearDown(self):
        self.temporary.cleanup()

    def run_batch(self, forecast=good_forecast, **kwargs):
        with redirect_stdout(io.StringIO()):
            return batch.run_simulation("2026-02-01", kwargs.pop("end", "2026-02-01"), self.output, forecast_fn=forecast, **kwargs)

    def assert_preserves_existing(self, forecast):
        self.output.write_bytes(b"previous-complete-submission\n")
        self.manifest.write_bytes(b"previous-manifest\n")
        with self.assertRaises((ValueError, TypeError, KeyError)):
            self.run_batch(forecast)
        self.assertEqual(self.output.read_bytes(), b"previous-complete-submission\n")
        self.assertEqual(self.manifest.read_bytes(), b"previous-manifest\n")

    def test_full_month_has_1344_rows_and_matching_provenance(self):
        seen = []

        def record(**kwargs):
            seen.append(kwargs)
            return good_forecast(**kwargs)

        result = self.run_batch(record, end="2026-02-28", refresh=True)
        with self.output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(result["row_count"], 1344)
        self.assertEqual(len(rows), 1344)
        self.assertEqual(len({(r["turbine"], r["timestamp"]) for r in rows}), 1344)
        self.assertEqual(rows[0]["timestamp"], "2026-02-01 00:00:00")
        self.assertEqual(rows[-1]["timestamp"], "2026-02-28 23:00:00")
        self.assertEqual(set(rows[0]), set(batch.COLUMNS))
        self.assertEqual(seen[0]["issue_time"], "2026-01-31T23:00:00+05:00")
        self.assertEqual(seen[-1]["issue_time"], "2026-02-27T23:00:00+05:00")
        self.assertTrue(all(call["refresh"] for call in seen))
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["csv_sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(len(manifest["runs"]), 56)
        self.assertEqual(manifest["timestamp_timezone"], "Asia/Almaty")
        self.assertNotIn("forecast_sample", manifest["runs"][0])

    def test_partial_success_does_not_overwrite(self):
        def partial(**kwargs):
            result = good_forecast(**kwargs)
            if kwargs["turbine_id"] == "turbine_2":
                result["forecast_sample"] = result["forecast_sample"][:1]
            return result
        self.assert_preserves_existing(partial)

    def test_error_response_does_not_overwrite(self):
        self.assert_preserves_existing(lambda **kwargs: {"status": "error", "message": "archive unavailable"})

    def test_duplicate_or_shifted_hours_do_not_overwrite(self):
        for mutation in ("duplicate", "shifted", "naive"):
            with self.subTest(mutation=mutation):
                def malformed(**kwargs):
                    result = good_forecast(**kwargs)
                    samples = result["forecast_sample"]
                    if mutation == "duplicate":
                        samples[-1] = copy.deepcopy(samples[0])
                    elif mutation == "shifted":
                        for sample in samples:
                            sample["time"] = (datetime.fromisoformat(sample["time"]) + timedelta(hours=1)).isoformat()
                    else:
                        samples[0]["time"] = "2026-02-01 00:00:00"
                    return result
                self.assert_preserves_existing(malformed)

    def test_invalid_numbers_do_not_overwrite(self):
        for field, value in (("predicted_power", float("nan")), ("predicted_power", float("inf")), ("predicted_power", 1.1), ("predicted_power", -0.1), ("wind_speed", -1), ("temperature", float("nan"))):
            with self.subTest(field=field, value=value):
                def malformed(**kwargs):
                    result = good_forecast(**kwargs)
                    result["forecast_sample"][0][field] = value
                    return result
                self.assert_preserves_existing(malformed)

    def test_issue_time_mismatch_and_missing_provenance_fail(self):
        for field, value in (("issue_time", "2026-02-01T23:00:00+05:00"), ("forecast_id", None), ("weather", None)):
            with self.subTest(field=field):
                def malformed(**kwargs):
                    result = good_forecast(**kwargs)
                    result[field] = value
                    return result
                self.assert_preserves_existing(malformed)

    def test_csv_commit_failure_restores_existing_manifest(self):
        self.output.write_bytes(b"old-csv")
        self.manifest.write_bytes(b"old-manifest")
        original_replace = batch.os.replace

        def fail_csv(source, destination):
            if Path(destination) == self.output:
                raise PermissionError("simulated locked CSV")
            return original_replace(source, destination)

        with patch.object(batch.os, "replace", side_effect=fail_csv):
            with self.assertRaises(PermissionError):
                self.run_batch()
        self.assertEqual(self.output.read_bytes(), b"old-csv")
        self.assertEqual(self.manifest.read_bytes(), b"old-manifest")
        self.assertEqual(len(list(self.output.parent.glob("*.tmp"))), 0)

    def test_invalid_date_range_fails_before_forecast(self):
        with patch("batch_forecast._validate_result") as validate:
            with self.assertRaises(ValueError):
                batch.run_simulation("2026-01-31", "2026-02-01", self.output, forecast_fn=good_forecast)
            with self.assertRaises(ValueError):
                batch.run_simulation("2026-02-02", "2026-02-01", self.output, forecast_fn=good_forecast)
            validate.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
