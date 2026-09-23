"""Bounded automatic rechecks of one historical forecast request.

The decision time stays fixed throughout a watch. Refreshing an archive can
detect revised admissible inputs or recover from a failed request, but never
permits weather published after the original decision. Replay successive
historical decisions with batch_forecast.py instead.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from forecast_engine import ARTIFACT_DIR, generate_agent_forecast


def watch_forecast(turbine_id="turbine_1", forecast_date="2026-02-01",
                   horizon_hours=48, issue_time=None, *, cycles=1, interval=60,
                   output_path=None, generate=None, sleep=None, emit=None):
    """Recheck and log every outcome; emit only initial/changed/error events.

    ``cycles`` includes the initial run and is always a positive finite count.
    Callbacks make the automatic retry/change behavior testable without network
    calls or real waiting. No background process survives this function.
    """
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1:
        raise ValueError("cycles must be a positive integer")
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval < 1:
        raise ValueError("interval must be at least one second")
    generate = generate if generate is not None else generate_agent_forecast
    sleep = sleep if sleep is not None else time.sleep
    emit = emit if emit is not None else print
    output = Path(output_path) if output_path is not None else ARTIFACT_DIR / "watch.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_id = None
    events = []
    for cycle in range(1, cycles + 1):
        event = {"event_time": datetime.now(timezone.utc).isoformat(), "cycle": cycle,
                 "turbine": turbine_id, "forecast_date": str(forecast_date),
                 "horizon_hours": horizon_hours,
                 "requested_issue_time": str(issue_time) if issue_time is not None else None,
                 "refresh": cycle > 1, "previous_forecast_id": previous_id}
        try:
            result = generate(turbine_id, forecast_date, horizon_hours,
                              issue_time=issue_time, refresh=cycle > 1)
            if not isinstance(result, dict) or result.get("status") != "success":
                message = result.get("message", "Forecast failed") if isinstance(result, dict) else "Invalid forecast response"
                raise RuntimeError(message)
            forecast_id = result.get("forecast_id")
            if not isinstance(forecast_id, str) or not forecast_id:
                raise RuntimeError("Successful forecast lacks its input identity")
            event.update({"event": "unchanged" if forecast_id == previous_id else "changed",
                          "initial": previous_id is None, "forecast_id": forecast_id,
                          "issue_time": result.get("issue_time"),
                          "weather_input_sha256": result.get("weather", {}).get("input_sha256")})
            previous_id = forecast_id
        except Exception as exc:
            event.update({"event": "error", "forecast_id": None,
                          "error_type": type(exc).__name__, "message": str(exc)})
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        events.append(event)
        if event["event"] == "error":
            emit(f"[{cycle}/{cycles}] ERROR: {event['message']}")
        elif event["event"] == "changed":
            label = "INITIAL" if event["initial"] else "CHANGED"
            emit(f"[{cycle}/{cycles}] {label}: {event['previous_forecast_id'] or '-'} -> {event['forecast_id']}")
        if cycle < cycles:
            sleep(interval)
    return events


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1; continuous background execution is not supported")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turbine", "--turbine_id", dest="turbine_id", choices=("turbine_1", "turbine_2"), default="turbine_1")
    parser.add_argument("--forecast-date", default="2026-02-01", help="Target start in SCADA_TIMEZONE unless an explicit offset is supplied")
    parser.add_argument("--horizon", type=int, choices=(24, 48), default=48)
    parser.add_argument("--issue-time", default=None, help="Fixed historical decision timestamp; default is one hour before forecast start")
    parser.add_argument("--cycles", type=_positive_int, default=1, help="Finite number of attempts including initial run (default: 1)")
    parser.add_argument("--interval", type=_positive_int, default=60, help="Seconds between attempts, minimum 1 (default: 60)")
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "watch.jsonl", help="Append-only JSONL event log")
    args = parser.parse_args(argv)
    events = watch_forecast(args.turbine_id, args.forecast_date, args.horizon, args.issue_time,
                            cycles=args.cycles, interval=args.interval, output_path=args.output)
    return 1 if any(event["event"] == "error" for event in events) else 0


if __name__ == "__main__":
    raise SystemExit(main())
