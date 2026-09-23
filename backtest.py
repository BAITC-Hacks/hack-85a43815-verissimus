"""Rolling historical evaluation with issued weather and matched baselines."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import forecast_engine as engine


def evaluate(start="2026-01-24", end="2026-01-30", horizon=48, output="validation/backtest.json"):
    if horizon not in (24, 48):
        raise ValueError("horizon must be 24 or 48")
    days = pd.date_range(start, end, freq="D")
    if not len(days) or len(days) > 31:
        raise ValueError("Choose between 1 and 31 issue dates")
    records, provenance = [], []
    for turbine in engine.TURBINE_COORDS:
        truth = engine.load_and_preprocess_turbine_data(engine.DATA_DIR / f"{turbine}.csv")
        for day in days:
            result = engine.generate_agent_forecast(turbine, day.strftime("%Y-%m-%d"), horizon)
            if result.get("status") != "success":
                raise RuntimeError(f"{turbine} {day.date()}: {result.get('message')}")
            issued = pd.Timestamp(result["issue_time"])
            known = truth.loc[truth.index + pd.Timedelta(hours=1) <= issued, "target_power"]
            persistence, mean = float(known.iloc[-1]), float(known.mean())
            run_meta = {k:result[k] for k in ("forecast_id", "turbine", "issue_time", "model")}
            run_meta["weather"] = {k:result["weather"][k] for k in ("provider", "run_time", "available_at", "input_sha256")}
            provenance.append(run_meta)
            for point in result["forecast_sample"]:
                valid = pd.Timestamp(point["time"])
                actual = float(truth.loc[valid, "target_power"]) if valid in truth.index else np.nan
                lead = int((valid - issued) / pd.Timedelta(hours=1))
                records.append({"turbine":turbine, "forecast_id":result["forecast_id"],
                                "issue_time":issued.isoformat(), "valid_time":valid.isoformat(),
                                "lead_hours":lead, "lead_band":"1-24" if lead<=24 else "25-48",
                                "actual":actual, "predicted":point["predicted_power"],
                                "persistence":persistence, "train_mean":mean})
            print(f"Evaluated {turbine} {day.date()}: {horizon} forecast hours", flush=True)
    frame = pd.DataFrame(records)
    metrics = []
    for (turbine, band), group in frame.groupby(["turbine", "lead_band"]):
        paired = group.dropna(subset=["actual"])
        if paired.empty:
            raise ValueError("No measured targets available for this validation group")
        row = {"turbine":turbine, "lead_band":band, "forecast_rows":len(group), "scored_rows":len(paired),
               "coverage":len(paired)/len(group), "methods":{}}
        for method in ("predicted", "persistence", "train_mean"):
            residual = paired[method] - paired.actual
            row["methods"][method] = {"mae":float(residual.abs().mean()),
                                       "rmse":float(np.sqrt((residual**2).mean())),
                                       "bias":float(residual.mean())}
        metrics.append(row)
    out = Path(output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.with_suffix(".predictions.csv")
    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    csv_path.write_bytes(csv_bytes)
    report = {"scope":"rolling_issued_weather_forecast", "start":str(days[0].date()),
              "end":str(days[-1].date()), "horizon_hours":horizon, "power_unit":"normalized",
              "timezone_assumption":engine.SITE_TIMEZONE,
              "protocol":"Forecast each local day at previous-day23:00; fit only completed SCADA hours available then; score all methods on identical available target rows.",
              "baselines":{"persistence":"Latest complete power measurement at decision time, held constant over the horizon",
                           "train_mean":"Mean of all complete training hours available at decision time"},
              "limitations":["Short historical check, not February2026 test accuracy", "No independent timezone confirmation", "No NWP bias calibration; grid wind may differ from turbine sensor", "Overlapping48h forecasts are separate decisions, not independent observations"],
              "metrics":metrics, "runs":provenance, "predictions_csv":csv_path.name,
              "predictions_sha256":hashlib.sha256(csv_bytes).hexdigest()}
    engine._atomic_json(out, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-01-24")
    parser.add_argument("--end", default="2026-01-30")
    parser.add_argument("--horizon", type=int, choices=(24,48), default=48)
    parser.add_argument("--output", default=str(engine.ROOT / "validation" / "backtest.json"))
    arguments = parser.parse_args()
    report = evaluate(arguments.start, arguments.end, arguments.horizon, arguments.output)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
