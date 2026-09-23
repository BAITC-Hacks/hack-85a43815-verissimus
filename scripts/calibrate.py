"""Reproduce the fixed January calibration experiment without February targets.

Run from the repository root: python scripts/calibrate.py
This writes a candidate under artifacts/; it never installs a production model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import forecast_engine as engine

FIT_CUTOFF = pd.Timestamp("2026-01-16T23:00:00+05:00").tz_convert("UTC")
METHODS = ("raw_power", "ridge_affine", "ridge_bias", "persistence", "train_mean")
PROTOCOL = {
    "version": "ridge-affine-jan2026-v1",
    "preregistered_at": "2026-09-23T11:13:00Z",
    "training_start_dates": "2026-01-01 through 2026-01-16; 48 hours except January16 uses24 hours",
    "training_target_availability_cutoff": FIT_CUTOFF.isoformat(),
    "evaluation_start_dates": "2026-01-17 through 2026-01-23; each48 hours",
    "primary": "Separate per-turbine/per-lead-band ridge affine clip(a*x+b,0,1); penalty24*((a-1)^2+b^2); slope constrained to[0,2]",
    "secondary": "Bias-only sensitivity reference clip(x+b,0,1), b=sum(y-x)/(n+48); never automatically selected",
    "promotion_rule": "Primary pooled MAE improves>=5%, pooled RMSE does not worsen, no turbine/lead-band MAE worsens>3%",
    "prior_inspection": "January24-30 issue-date results were inspected before this experiment. Report a target<January24 slice separately because the last fresh48h run overlaps January24.",
    "limitations": ["Short historical evaluation, not February test accuracy", "Timezone and telemetry latency assumptions remain unconfirmed", "Overlapping forecast windows are not independent observations"],
    "parameters_policy": "Fit and freeze before evaluating; never tune on evaluation targets",
}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    engine._atomic_json(path, value)


def collect(days, split, output):
    records, runs = [], []
    for day in days:
        horizon = 24 if split == "train" and day.day == 16 else 48
        for turbine in engine.TURBINE_COORDS:
            result = engine.generate_agent_forecast(turbine, str(day.date()), horizon)
            if result.get("status") != "success":
                raise RuntimeError(f"{turbine} {day.date()}: {result.get('message')}")
            calibration = result.get("model", {}).get("calibration", result.get("calibration", {}))
            if calibration.get("applied") or calibration.get("status") == "applied":
                raise RuntimeError("Reproduction must use uncalibrated base forecasts")
            issued = pd.Timestamp(result["issue_time"])
            if pd.Timestamp(result["model"]["training_available_at"]) > issued:
                raise ValueError("Base power model uses future training data")
            if pd.Timestamp(result["weather"]["available_at"]) > issued:
                raise ValueError("Base forecast uses unavailable weather")
            raw_path = output / "raw" / f"{turbine}_{day.date()}_{horizon}.json"
            write_json(raw_path, result)
            truth = engine.load_and_preprocess_turbine_data(engine.DATA_DIR / f"{turbine}.csv")
            known = truth.loc[truth.index + pd.Timedelta(hours=1) <= issued, "target_power"]
            persistence, mean = float(known.iloc[-1]), float(known.mean())
            runs.append({"split":split, "turbine":turbine, "forecast_id":result["forecast_id"],
                         "issue_time":issued.isoformat(), "model":result["model"], "weather":result["weather"],
                         "raw_file":str(raw_path.relative_to(output))})
            for point in result["forecast_sample"]:
                valid = pd.Timestamp(point["time"])
                if split == "train" and valid + pd.Timedelta(hours=1) > FIT_CUTOFF:
                    continue
                lead = int((valid-issued)/pd.Timedelta(hours=1))
                actual = float(truth.loc[valid,"target_power"]) if valid in truth.index else np.nan
                records.append({"split":split, "forecast_start":str(day.date()), "turbine":turbine,
                                "forecast_id":result["forecast_id"], "issue_time":issued.isoformat(),
                                "valid_time":valid.isoformat(), "lead_hours":lead,
                                "lead_band":"1-24" if lead<=24 else "25-48", "actual":actual,
                                "raw_power":point["predicted_power"], "wind_speed_gfs":point["wind_speed"],
                                "persistence":persistence, "train_mean":mean})
            print(f"Collected {split} {turbine} {day.date()}", flush=True)
    return pd.DataFrame(records), runs


def fit_parameters(training):
    if not training.split.eq("train").all():
        raise ValueError("Only explicitly designated training rows may enter calibration")
    if ((pd.to_datetime(training.valid_time, utc=True)+pd.Timedelta(hours=1)) > FIT_CUTOFF).any():
        raise ValueError("Calibration target was unavailable at the first validation issue")
    paired = training.dropna(subset=["actual"])
    parameters = []
    for (turbine, band), group in paired.groupby(["turbine","lead_band"]):
        if len(group) < 72:
            raise ValueError("Insufficient paired hours for calibration")
        x,y = group.raw_power.to_numpy(), group.actual.to_numpy()
        if not np.isfinite(np.column_stack([x,y])).all() or not ((x>=0)&(x<=1)&(y>=0)&(y<=1)).all():
            raise ValueError("Calibration powers must be finite and normalized")
        matrix = np.column_stack([x,np.ones(len(x))])
        a,b = np.linalg.solve(matrix.T@matrix+24*np.eye(2), matrix.T@y+24*np.array([1.,0.]))
        if not 0 <= a <= 2:
            a=float(np.clip(a,0,2))
            b=float(np.sum(y-a*x)/(len(x)+24))
        parameters.append({"turbine":turbine, "lead_band":band, "a":float(a),"b":float(b),
                           "bias":float(np.sum(y-x)/(len(x)+48)), "fit_rows":len(group),
                           "fit_target_start":group.valid_time.min(),"fit_target_end":group.valid_time.max(),
                           "fit_available_at":FIT_CUTOFF.isoformat()})
    expected={(t,b) for t in engine.TURBINE_COORDS for b in ("1-24","25-48")}
    if {(p["turbine"],p["lead_band"]) for p in parameters} != expected:
        raise ValueError("Both turbines and both lead bands are required")
    return parameters


def predict(frame, parameters):
    result=frame.copy()
    for p in parameters:
        mask=result.turbine.eq(p["turbine"]) & result.lead_band.eq(p["lead_band"])
        raw=result.loc[mask,"raw_power"]
        result.loc[mask,"ridge_affine"]=np.clip(p["a"]*raw+p["b"],0,1)
        result.loc[mask,"ridge_bias"]=np.clip(raw+p["bias"],0,1)
    if not np.isfinite(result[["raw_power","ridge_affine","ridge_bias"]]).all().all():
        raise ValueError("Every forecast row requires valid calibration parameters")
    return result


def metrics(group):
    paired=group.dropna(subset=["actual"])
    if paired.empty:
        raise ValueError("No paired target hours")
    result={"forecast_rows":len(group),"scored_rows":len(paired),"coverage":len(paired)/len(group),"methods":{}}
    for method in METHODS:
        residual=paired[method]-paired.actual
        if not np.isfinite(residual).all():
            raise ValueError("All methods must score identical finite target rows")
        result["methods"][method]={"mae":float(residual.abs().mean()),"rmse":float(np.sqrt((residual**2).mean())),"bias":float(residual.mean())}
    return result


def promotion_passed(pooled, groups):
    raw,cal=pooled["methods"]["raw_power"],pooled["methods"]["ridge_affine"]
    return bool(raw["mae"] > 0 and cal["mae"] <= .95*raw["mae"] and cal["rmse"] <= raw["rmse"] and
                all(g["methods"]["ridge_affine"]["mae"] <= 1.03*g["methods"]["raw_power"]["mae"] for g in groups))


def reproduce(output):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    disabled=output/"disabled.json"
    if disabled.exists():
        raise ValueError("The reserved disabled.json path must not exist")
    if not hasattr(engine,"CALIBRATION_FILE"):
        raise RuntimeError("This reproduction script requires the engine CALIBRATION_FILE guard")
    previous_file,previous_artifacts=engine.CALIBRATION_FILE,engine.ARTIFACT_DIR
    engine.CALIBRATION_FILE=disabled
    engine.ARTIFACT_DIR=output/"engine_runs"
    try:
        write_json(output/"protocol.json",PROTOCOL)
        training,train_runs=collect(pd.date_range("2026-01-01","2026-01-16"),"train",output)
        # Freeze the serialized training data as the exact numerical fit input.
        fit_bytes=training.to_csv(index=False).encode("utf-8")
        (output/"training_predictions.csv").write_bytes(fit_bytes)
        parameters=fit_parameters(pd.read_csv(output/"training_predictions.csv"))
        frozen={"fixed_at":pd.Timestamp.now(tz="UTC").isoformat(),"parameters":parameters}
        write_json(output/"fitted_before_evaluation.json",frozen)
        evaluation,evaluation_runs=collect(pd.date_range("2026-01-17","2026-01-23"),"evaluation",output)
        all_rows=predict(pd.concat([training,evaluation],ignore_index=True),parameters)
        scored=all_rows.loc[all_rows.split.eq("evaluation")]
        groups=[{"turbine":t,"lead_band":b,**metrics(g)} for (t,b),g in scored.groupby(["turbine","lead_band"])]
        pooled=metrics(scored)
        target_only=scored.loc[pd.to_datetime(scored.valid_time,utc=True)<pd.Timestamp("2026-01-24T00:00:00+05:00")]
        fit_targets=training.dropna(subset=["actual"])
        latest_target=(pd.to_datetime(fit_targets.valid_time,utc=True)+pd.Timedelta(hours=1)).max()
        latest_weather=max(pd.Timestamp(r["weather"]["available_at"]) for r in train_runs)
        available=max(latest_target,latest_weather)
        if available>pd.to_datetime(evaluation.issue_time,utc=True).min():
            raise ValueError("Calibration was unavailable at the first evaluation issue")
        csv_bytes=all_rows.to_csv(index=False).encode("utf-8")
        (output/"predictions.csv").write_bytes(csv_bytes)
        runs=train_runs+evaluation_runs
        write_json(output/"provenance.json",runs)
        report={"protocol":PROTOCOL,"parameters":parameters,"metrics":groups,"pooled":pooled,
                "new_target_only_pooled":metrics(target_only),
                "primary_passes_prefixed_promotion_rule":promotion_passed(pooled,groups),
                "predictions_sha256":hashlib.sha256(csv_bytes).hexdigest()}
        write_json(output/"report.json",report)
        candidate={"schema_version":1,"method":"ridge_affine","version":PROTOCOL["version"],
                   "base_model_version":engine.MODEL_VERSION,"timezone_assumption":engine.SITE_TIMEZONE,
                   "fit_start":fit_targets.valid_time.min(),"fit_end":fit_targets.valid_time.max(),
                   "available_at":available.isoformat(),"power_unit":"normalized",
                   "training_source_sha256":{r["turbine"]:r["model"]["source_sha256"] for r in train_runs},
                   "parameters":{t:{p["lead_band"]:{"slope":p["a"],"intercept":p["b"],"training_rows":p["fit_rows"]} for p in parameters if p["turbine"]==t} for t in engine.TURBINE_COORDS},
                   "protocol_sha256":hashlib.sha256((output/"protocol.json").read_bytes()).hexdigest(),
                   "training_predictions_sha256":hashlib.sha256(fit_bytes).hexdigest(),
                   "training_forecast_ids":[r["forecast_id"] for r in train_runs],
                   "promotion_rule_passed":report["primary_passes_prefixed_promotion_rule"],
                   "validation_report_sha256":hashlib.sha256((output/"report.json").read_bytes()).hexdigest()}
        candidate["artifact_sha256"]=canonical_sha(candidate)
        write_json(output/"calibration_candidate.json",candidate)
        return report
    finally:
        engine.CALIBRATION_FILE=previous_file
        engine.ARTIFACT_DIR=previous_artifacts


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts"/"calibration_experiment")
    arguments=parser.parse_args()
    result=reproduce(arguments.output)
    print(json.dumps({"pooled":result["pooled"],"promotion_passed":result["primary_passes_prefixed_promotion_rule"]},indent=2))
