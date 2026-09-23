"""Report exports do not need network access and never imply unchecked success."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
import unittest

from reporting import build_forecast_report, forecast_checks


def snapshot():
    start = datetime(2026, 1, 31, 19, tzinfo=timezone.utc)
    return {
        "status": "success", "forecast_id": "sample-snapshot", "turbine": "turbine_1", "power_unit": "normalized",
        "forecast_start": start.isoformat(), "horizon_hours": 24,
        "issue_time": "2026-01-31T18:00:00+00:00", "timezone_assumption": "Asia/Almaty",
        "weather": {"provider": "NOAA GFS", "model": "GFS 0.25",
                    "run_time": "2026-01-31T12:00:00+00:00", "available_at": "2026-01-31T16:00:00+00:00",
                    "sources": [{"last_modified": "2026-01-31T16:00:00+00:00"}]},
        "model": {"version": "test-power-v1", "training_start": "2023-03-11T00:00:00+00:00",
                  "training_end": "2026-01-31T17:00:00+00:00", "training_hours": 100,
                  "training_available_at": "2026-01-31T18:00:00+00:00"},
        "data_quality": {"raw_rows": 600, "complete_hours": 100, "samples_required_per_hour": 6},
        "summary": {"avg_predicted_power": 999},
        "forecast_sample": [{"time": (start + timedelta(hours=index)).isoformat(),
                             "predicted_power": index / 24, "wind_speed": 3 + index / 10,
                             "temperature": -2} for index in range(24)],
    }


class ParsedReport(HTMLParser):
    def __init__(self, report):
        super().__init__()
        self.tags, self.attributes, self.text = [], [], []
        self.feed(report)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attributes.extend(attrs)

    def handle_data(self, data):
        self.text.append(data)


class CheckTests(unittest.TestCase):
    def results(self, data):
        return {item["id"]: item["status"] for item in forecast_checks(data)}

    def test_valid_snapshot_passes_recomputed_checks(self):
        checks = self.results(snapshot())
        self.assertEqual(len(checks), 10)
        self.assertEqual(set(checks.values()), {"pass"})

    def test_output_count_duplicate_and_gap_are_independent_failures(self):
        data = snapshot()
        data["forecast_sample"].pop()
        self.assertEqual(self.results(data)["horizon"], "fail")
        data = snapshot()
        data["forecast_sample"][3]["time"] = data["forecast_sample"][2]["time"]
        results = self.results(data)
        self.assertEqual(results["unique_hours"], "fail")
        self.assertEqual(results["hourly_grid"], "fail")
        data = snapshot()
        data["forecast_sample"][0]["time"] = "2026-01-31T17:00:00+00:00"
        self.assertEqual(self.results(data)["hourly_grid"], "fail")
        self.assertEqual(self.results(data)["decision_cutoff"], "fail")

    def test_nan_out_of_range_bool_and_missing_weather_fail(self):
        for value in (float("nan"), float("inf"), -0.01, 1.01, None, True):
            with self.subTest(value=value):
                data = snapshot()
                data["forecast_sample"][0]["predicted_power"] = value
                self.assertEqual(self.results(data)["power_bounds"], "fail")
        data = snapshot()
        data["forecast_sample"][0]["wind_speed"] = -1
        self.assertEqual(self.results(data)["weather_values"], "fail")

    def test_missing_or_naive_cutoffs_never_become_green(self):
        data = snapshot()
        del data["model"]["training_available_at"]
        data["weather"]["available_at"] = "2026-01-31T16:00:00"
        del data["weather"]["sources"]
        results = self.results(data)
        self.assertEqual(results["training_cutoff"], "unknown")
        self.assertEqual(results["weather_cutoff"], "unknown")
        self.assertEqual(results["source_cutoff"], "unknown")

    def test_success_unit_and_declared_start_cannot_be_assumed(self):
        data = snapshot()
        data["status"] = "error"
        self.assertEqual(self.results(data)["snapshot_contract"], "fail")
        data["status"], data["power_unit"] = "success", "MW"
        self.assertEqual(self.results(data)["snapshot_contract"], "fail")
        del data["forecast_start"]
        self.assertEqual(self.results(data)["hourly_grid"], "unknown")

    def test_corrupt_metadata_is_unconfirmed_instead_of_crashing(self):
        data = snapshot()
        data.update(weather="corrupt", model=["corrupt"], data_quality="corrupt")
        results = self.results(data)
        self.assertEqual(results["training_cutoff"], "unknown")
        self.assertEqual(results["weather_cutoff"], "unknown")
        self.assertIn("Не подтверждено", build_forecast_report(data))

    def test_archive_and_training_cutoffs_reject_future_information(self):
        for key in ("run_time", "available_at"):
            with self.subTest(key=key):
                data = snapshot()
                data["weather"][key] = "2026-01-31T19:00:00+00:00"
                self.assertEqual(self.results(data)["weather_cutoff"], "fail")
        data = snapshot()
        data["weather"]["sources"][0]["last_modified"] = "2026-01-31T19:00:00+00:00"
        self.assertEqual(self.results(data)["source_cutoff"], "fail")
        data = snapshot()
        data["model"]["training_end"] = "2026-01-31T18:00:00+00:00"
        self.assertEqual(self.results(data)["training_cutoff"], "fail")

    def test_source_maximum_must_match_reported_availability(self):
        data = snapshot()
        data["weather"]["sources"][0]["last_modified"] = "2026-01-31T17:00:00+00:00"
        self.assertEqual(self.results(data)["source_cutoff"], "fail")
        data["weather"]["sources"][0]["last_modified"] = "broken"
        self.assertEqual(self.results(data)["source_cutoff"], "fail")

    def test_empty_snapshot_has_no_false_success_or_crash(self):
        self.assertNotIn("pass", self.results({}).values())
        self.assertIn("График недоступен", build_forecast_report({}))

    def test_optional_calibration_checks_completed_hours_and_availability(self):
        data = snapshot()
        data["model"]["calibration"] = {"applied": True, "fit_end": "2026-01-31T17:00:00+00:00",
                                       "available_at": "2026-01-31T18:00:00+00:00"}
        self.assertEqual(self.results(data)["calibration_cutoff"], "pass")
        data["model"]["calibration"]["fit_end"] = "2026-01-31T18:00:00+00:00"
        self.assertEqual(self.results(data)["calibration_cutoff"], "fail")
        del data["model"]["calibration"]["available_at"]
        self.assertEqual(self.results(data)["calibration_cutoff"], "unknown")
        data["model"]["calibration"]["applied"] = False
        self.assertNotIn("calibration_cutoff", self.results(data))


class ExportTests(unittest.TestCase):
    def test_external_strings_cannot_create_elements_or_attributes(self):
        data = snapshot()
        attack = '<script>alert("x")</script><img src="https://evil.test/x" onerror="alert(1)"> & \' quoted'
        data["weather"]["provider"] = attack
        data["turbine"] = attack
        data["forecast_id"] = attack
        data["model"]["calibration"] = {"note": attack}
        data["assumptions"] = [attack]
        report = build_forecast_report(data)
        parsed = ParsedReport(report)
        self.assertNotIn("script", parsed.tags)
        self.assertNotIn("img", parsed.tags)
        self.assertNotIn("onerror", [key for key, _ in parsed.attributes])
        self.assertIn("&lt;script&gt;", report)
        self.assertIn(attack, "".join(parsed.text))

    def test_offline_export_has_svg_no_external_dependencies_and_all_hours(self):
        data = snapshot()
        report = build_forecast_report(data)
        parsed = ParsedReport(report)
        self.assertIn("svg", parsed.tags)
        self.assertIn("polyline", parsed.tags)
        self.assertFalse(set(parsed.tags) & {"script", "link", "iframe", "img"})
        self.assertFalse({key for key, _ in parsed.attributes} & {"href", "src"})
        self.assertIn("2026-02-01T00:00+05:00", report)
        self.assertIn("2026-01-31T19:00+00:00", report)
        self.assertIn("2026-02-01T23:00+05:00", report)
        self.assertIn("HTTP Last-Modified", report)
        self.assertIn("не независимый журнал", report)
        self.assertIn("Исходных десятиминутных строк", report)
        self.assertIn("0.4792", report)  # mean of the actual points, not supplied summary
        self.assertNotIn("999.0000", report)

    def test_report_generation_never_changes_snapshot(self):
        data = snapshot()
        before = deepcopy(data)
        build_forecast_report(data)
        self.assertEqual(data, before)

    def test_report_uses_snapshot_timezone_and_discloses_explicit_override(self):
        data = snapshot()
        data["timezone_assumption"] = "UTC"
        report = build_forecast_report(data)
        self.assertNotIn("+05:00", report)
        override = build_forecast_report(data, "Asia/Almaty")
        self.assertIn("+05:00", override)
        self.assertIn("Часовой пояс отображения отчёта</dt><dd>Asia/Almaty", override)
        data["timezone_assumption"] = "unknown"
        self.assertIn("исходный пояс не распознан", build_forecast_report(data))

    def test_invalid_curve_is_not_silently_clipped_or_drawn(self):
        data = snapshot()
        data["forecast_sample"][0]["predicted_power"] = float("nan")
        parsed = ParsedReport(build_forecast_report(data))
        self.assertNotIn("polyline", parsed.tags)
        self.assertIn("График недоступен", "".join(parsed.text))


if __name__ == "__main__":
    unittest.main()
