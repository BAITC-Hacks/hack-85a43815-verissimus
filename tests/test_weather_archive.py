"""Temporal provenance and bounded-download regression tests (offline)."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

import weather_archive as weather


ISSUE = pd.Timestamp("2026-01-31T18:00:00Z")
RUN = pd.Timestamp("2026-01-31T12:00:00Z")
CELL = (43.75, 78.5)
INDEX = pd.date_range("2026-01-31T19:00:00Z", periods=24, freq="h")


def record(lead=7):
    return {"cache_version": weather.CACHE_VERSION, "run_time": RUN.isoformat(),
            "lead_hours": lead, "grid_cell": list(CELL), "wind_speed": 4.0,
            "temperature": -2.0, "retrieved_at": "2026-09-23T00:00:00+00:00",
            "sources": [{"field": field, "last_modified": "2026-01-31T16:00:00+00:00",
                         "sha256": "a" * 64, "url": "https://example.test/gfs", "etag": "1"}
                        for field in ("inventory", "temperature", "u100", "v100")]}


class ArchiveTests(unittest.TestCase):
    def test_one_run_and_same_grid_cell_reused_across_turbines(self):
        def hour(run, lead, cell, issue, refresh):
            self.assertEqual(run, RUN)
            self.assertEqual(cell, CELL)
            self.assertEqual(issue, ISSUE)
            return record(lead)
        with patch.object(weather, "_hour", side_effect=hour) as call:
            frame, meta = weather.fetch_weather(43.645150, 78.535604, INDEX, ISSUE)
            other, other_meta = weather.fetch_weather(43.643198, 78.538828, INDEX, ISSUE)
        self.assertEqual(call.call_count, 48)
        self.assertTrue(frame.index.equals(INDEX))
        self.assertTrue(frame.equals(other))
        self.assertEqual(meta["input_sha256"], other_meta["input_sha256"])
        self.assertEqual(len(meta["sources"]), 96)
        self.assertLessEqual(pd.Timestamp(meta["available_at"]), ISSUE)

    def test_inputs_reject_naive_gaps_duplicates_and_past_hours(self):
        for index in [INDEX.tz_localize(None), INDEX.delete(3), INDEX.insert(1, INDEX[0]), INDEX - pd.Timedelta(days=1)]:
            with self.subTest(index=index[:2]), self.assertRaises(weather.WeatherArchiveError):
                weather.fetch_weather(43, 78, index, ISSUE)
        with self.assertRaises(weather.WeatherArchiveError):
            weather.fetch_weather(43, 78, INDEX, ISSUE.tz_localize(None))

    def test_archive_metadata_after_decision_is_rejected_even_from_cache(self):
        item = record()
        item["sources"][1]["last_modified"] = "2026-02-01T00:00:00Z"
        with self.assertRaisesRegex(weather.WeatherArchiveError, "not available"):
            weather._check_record(item, RUN, 7, CELL, ISSUE)

    def test_record_requires_all_fields_and_finite_values(self):
        item = record()
        item["sources"].pop()
        with self.assertRaises(weather.WeatherArchiveError):
            weather._check_record(item, RUN, 7, CELL, ISSUE)
        item = record()
        item["temperature"] = float("nan")
        with self.assertRaises(weather.WeatherArchiveError):
            weather._check_record(item, RUN, 7, CELL, ISSUE)

    def test_server_ignoring_range_is_rejected_without_reading_body(self):
        response = MagicMock()
        response.status_code = 200
        response.headers = {"Content-Length": "550000000"}
        session = MagicMock()
        session.get.return_value.__enter__.return_value = response
        with patch.object(weather, "_session", return_value=session):
            with self.assertRaisesRegex(weather.WeatherArchiveError, "full GRIB download refused"):
                weather._download("https://example.test/gfs", ISSUE, (100, 199))
        response.iter_content.assert_not_called()

    def test_future_publication_is_rejected_before_reading_body(self):
        response = MagicMock()
        response.status_code = 206
        response.headers = {"Content-Range": "bytes 100-199/500", "Last-Modified": "Sun, 01 Feb 2026 01:00:00 GMT"}
        session = MagicMock()
        session.get.return_value.__enter__.return_value = response
        with patch.object(weather, "_session", return_value=session):
            with self.assertRaisesRegex(weather.WeatherArchiveError, "after issue_time"):
                weather._download("https://example.test/gfs", ISSUE, (100, 199))
        response.iter_content.assert_not_called()

    def test_truncated_range_is_rejected(self):
        response = MagicMock()
        response.status_code = 206
        response.headers = {"Content-Range": "bytes 100-199/500", "Last-Modified": "Sat, 31 Jan 2026 16:00:00 GMT"}
        response.iter_content.return_value = [b"x" * 99]
        session = MagicMock()
        session.get.return_value.__enter__.return_value = response
        with patch.object(weather, "_session", return_value=session):
            with self.assertRaisesRegex(weather.WeatherArchiveError, "Truncated"):
                weather._download("https://example.test/gfs", ISSUE, (100, 199))

    def test_inventory_analysis_or_wrong_run_is_rejected(self):
        index = ("1:0:d=2026013112:TMP:2 m above ground:7 hour fcst:\n"
                 "2:100:d=2026013112:UGRD:100 m above ground:7 hour fcst:\n"
                 "3:200:d=2026013112:VGRD:100 m above ground:7 hour fcst:\n"
                 "4:300:d=2026013112:END:surface:7 hour fcst:\n")
        self.assertEqual(weather._field_ranges(index, RUN, 7)["temperature"], (0, 99))
        for incorrect in [index.replace("7 hour fcst", "anl"), index.replace("2026013112", "2026020112")]:
            with self.assertRaises(weather.WeatherArchiveError):
                weather._field_ranges(incorrect, RUN, 7)

    def test_corrupt_cache_cannot_silently_modify_input_weather(self):
        with tempfile.TemporaryDirectory() as directory:
            item = record()
            envelope = {"record": item, "record_sha256": weather._sha(weather._canonical(item))}
            item["wind_speed"] = 99
            path = Path(directory) / "gfs_2026013112_f007_43.75_78.50.json"
            path.write_text(json.dumps(envelope), encoding="utf-8")
            with patch.object(weather, "CACHE_DIR", Path(directory)), patch.object(weather, "_download") as download:
                with self.assertRaisesRegex(weather.WeatherArchiveError, "checksum"):
                    weather._hour(RUN, 7, CELL, ISSUE, False)
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
