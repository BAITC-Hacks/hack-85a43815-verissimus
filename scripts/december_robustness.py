"""Bounded December robustness check. No production files are changed."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument("--output", type=Path)
arguments = parser.parse_args()
PROJECT = arguments.project.resolve()
OUT = (arguments.output or PROJECT / "artifacts" / "december_robustness").resolve()
if OUT.exists() and any(OUT.iterdir()):
    raise ValueError("Use a fresh empty output directory; existing evidence is never overwritten")
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
import forecast_engine as engine

if engine.SITE_TIMEZONE != "Asia/Almaty":
    raise ValueError("The fixed December protocol requires SCADA_TIMEZONE=Asia/Almaty")

STARTED = time.monotonic()
CUTOFF = pd.Timestamp("2025-12-16T23:00:00+05:00").tz_convert("UTC")
METHODS = ("raw_power", "ridge_affine", "persistence", "train_mean")
PROTOCOL = {
    "scope": "backward-in-time robustness check of a method chosen on January; not a prospective independent test",
    "frozen_at": pd.Timestamp.now(tz="UTC").isoformat(),
    "training_start_dates": "2025-12-01 through 2025-12-16; 48h except December16 uses24h",
    "training_target_end_cutoff": CUTOFF.isoformat(),
    "evaluation_start_dates": "2025-12-17 through 2025-12-23; each48h",
    "method": "Separate per turbine and decision lead band1-24/25-48; clip(a*raw+b,0,1)",
    "objective": "sum((y-a*x-b)^2)+24*((a-1)^2+b^2)",
    "constraints": "slope a in[0,2]; recompute intercept analytically if slope hits boundary",
    "fixed_policy": "No new candidates or hyperparameter selection. Fit coefficients only from December training rows, freeze before evaluation. January coefficients are disabled.",
    "comparators": list(METHODS),
    "power_unit": "normalized",
    "timezone_assumption": engine.SITE_TIMEZONE,
    "known_limitations": ["Method selection previously used January results; this earlier-time check tests transfer across periods", "Overlapping48h issue windows are not independent observations", "Unknown SCADA timezone and zero-latency assumptions remain", "No February measured targets are used"],
    "budget_seconds": 1150,
    "engine_model_version": engine.MODEL_VERSION,
    "engine_file_sha256": hashlib.sha256((PROJECT/"forecast_engine.py").read_bytes()).hexdigest(),
}
RECORDS, RUNS = [], []


def write(path, value):
    engine._atomic_json(path, value)


def status(state, **kwargs):
    write(OUT/"status.json", {"status":state,"elapsed_seconds":time.monotonic()-STARTED,
                             "completed_forecasts":len(RUNS),"saved_rows":len(RECORDS),**kwargs})


def budget_stop():
    status("incomplete_budget", message="The bounded worker reached its wall-clock budget; no complete evaluation is claimed.")
    os._exit(124)


def collect(days, split):
    for day in days:
        horizon=24 if split=="train" and day.day==16 else 48
        for turbine in engine.TURBINE_COORDS:
            if time.monotonic()-STARTED > 1100:
                raise TimeoutError("Insufficient budget for another forecast")
            result=engine.generate_agent_forecast(turbine,str(day.date()),horizon)
            if result.get("status")!="success":
                raise RuntimeError(f"{turbine} {day.date()}: {result.get('message')}")
            if result["model"]["calibration"]["applied"]:
                raise ValueError("January calibration leaked into the December base forecast")
            issue=pd.Timestamp(result["issue_time"])
            if pd.Timestamp(result["model"]["training_available_at"])>issue or pd.Timestamp(result["weather"]["available_at"])>issue:
                raise ValueError("An input was unavailable at the decision time")
            for source in result["weather"]["sources"]:
                if pd.Timestamp(source["last_modified"])>issue:
                    raise ValueError("An archive source was unavailable at issue time")
            file=OUT/"raw"/f"{turbine}_{day.date()}_{horizon}.json"
            write(file,result)
            truth=engine.load_and_preprocess_turbine_data(engine.DATA_DIR/f"{turbine}.csv")
            known=truth.loc[truth.index+pd.Timedelta(hours=1)<=issue,"target_power"]
            persistence,mean=float(known.iloc[-1]),float(known.mean())
            RUNS.append({"split":split,"turbine":turbine,"forecast_id":result["forecast_id"],"issue_time":issue.isoformat(),
                         "model":result["model"],"weather":result["weather"],"raw_file":str(file.relative_to(OUT))})
            for point in result["forecast_sample"]:
                valid=pd.Timestamp(point["time"])
                if split=="train" and valid+pd.Timedelta(hours=1)>CUTOFF:
                    continue
                lead=int((valid-issue)/pd.Timedelta(hours=1))
                RECORDS.append({"split":split,"forecast_start":str(day.date()),"turbine":turbine,
                                "forecast_id":result["forecast_id"],"issue_time":issue.isoformat(),"valid_time":valid.isoformat(),
                                "lead_hours":lead,"lead_band":"1-24" if lead<=24 else "25-48",
                                "actual":float(truth.loc[valid,"target_power"]) if valid in truth.index else np.nan,
                                "raw_power":point["predicted_power"],"persistence":persistence,"train_mean":mean})
            pd.DataFrame(RECORDS).to_csv(OUT/"progress.csv",index=False)
            write(OUT/"provenance.json",RUNS)
            status("running",phase=split,last_forecast=f"{turbine} {day.date()}")
            print(f"COLLECT {split} {turbine} {day.date()} elapsed={time.monotonic()-STARTED:.1f}s",flush=True)


def fit():
    training=pd.DataFrame(RECORDS)
    if not training.split.eq("train").all():
        raise ValueError("Evaluation rows cannot enter fitting")
    training.to_csv(OUT/"training_predictions.csv",index=False)
    # The serialized training file is the exact, reproducible fit input.
    training=pd.read_csv(OUT/"training_predictions.csv")
    paired=training.dropna(subset=["actual"])
    latest_target=(pd.to_datetime(paired.valid_time,utc=True)+pd.Timedelta(hours=1)).max()
    latest_weather=max(pd.Timestamp(r["weather"]["available_at"]) for r in RUNS)
    available=max(latest_target,latest_weather)
    if available>CUTOFF:
        raise ValueError("Calibration inputs violate the first-test information cutoff")
    params=[]
    for (t,band),group in paired.groupby(["turbine","lead_band"]):
        x,y=group.raw_power.to_numpy(),group.actual.to_numpy()
        if len(x)<72 or not np.isfinite(np.column_stack([x,y])).all():
            raise ValueError("Insufficient valid training pairs")
        design=np.column_stack([x,np.ones(len(x))])
        a,b=np.linalg.solve(design.T@design+24*np.eye(2),design.T@y+24*np.array([1.,0.]))
        if not 0<=a<=2:
            a=float(np.clip(a,0,2));b=float(np.sum(y-a*x)/(len(x)+24))
        params.append({"turbine":t,"lead_band":band,"slope":float(a),"intercept":float(b),"training_rows":len(x),
                       "fit_start":group.valid_time.min(),"fit_end":group.valid_time.max()})
    if len(params)!=4:
        raise ValueError("All four turbine/lead groups are required")
    frozen={"frozen_at":pd.Timestamp.now(tz="UTC").isoformat(),"available_at":available.isoformat(),
            "parameters":params,"training_predictions_sha256":hashlib.sha256((OUT/"training_predictions.csv").read_bytes()).hexdigest()}
    write(OUT/"fitted_before_evaluation.json",frozen)
    return frozen


def metrics(group):
    paired=group.dropna(subset=["actual"])
    if paired.empty:
        raise ValueError("No measured evaluation targets")
    report={"forecast_rows":len(group),"scored_rows":len(paired),"coverage":len(paired)/len(group),"methods":{}}
    for method in METHODS:
        residual=paired[method]-paired.actual
        if not np.isfinite(residual).all():
            raise ValueError("Nonfinite evaluation predictions")
        report["methods"][method]={"mae":float(residual.abs().mean()),"rmse":float(np.sqrt((residual**2).mean())),"bias":float(residual.mean())}
    return report


def evaluate(frozen):
    frame=pd.DataFrame(RECORDS)
    evaluation=frame.loc[frame.split.eq("evaluation")].copy()
    if pd.Timestamp(frozen["available_at"])>pd.to_datetime(evaluation.issue_time,utc=True).min():
        raise ValueError("Frozen calibration was unavailable at an evaluation decision")
    for p in frozen["parameters"]:
        mask=evaluation.turbine.eq(p["turbine"]) & evaluation.lead_band.eq(p["lead_band"])
        evaluation.loc[mask,"ridge_affine"]=np.clip(p["slope"]*evaluation.loc[mask,"raw_power"]+p["intercept"],0,1)
    data=evaluation.to_csv(index=False).encode("utf-8")
    (OUT/"evaluation_predictions.csv").write_bytes(data)
    groups=[{"turbine":t,"lead_band":b,**metrics(g)} for (t,b),g in evaluation.groupby(["turbine","lead_band"])]
    pooled=metrics(evaluation)
    report={"protocol":PROTOCOL,"frozen_calibration":frozen,"metrics":groups,"pooled":pooled,
            "training_coverage":[{"turbine":t,"lead_band":b,"forecast_rows":len(g),"paired_rows":int(g.actual.notna().sum())} for (t,b),g in frame[frame.split.eq("train")].groupby(["turbine","lead_band"])],
            "evaluation_predictions_sha256":hashlib.sha256(data).hexdigest(),
            "protocol_sha256":hashlib.sha256((OUT/"protocol.json").read_bytes()).hexdigest(),
            "production_changed":False,"elapsed_seconds":time.monotonic()-STARTED}
    write(OUT/"report.json",report)
    print(json.dumps({"metrics":groups,"pooled":pooled},indent=2),flush=True)


def main():
    disabled=OUT/"disabled_calibration.json"
    if disabled.exists():
        raise ValueError("The reserved disabled calibration path must not exist")
    if (OUT/"protocol.json").exists():
        raise ValueError("Use a fresh directory or intentionally archive the prior run; protocol is immutable")
    write(OUT/"protocol.json",PROTOCOL)
    timer=threading.Timer(1150,budget_stop);timer.daemon=True;timer.start()
    previous_calibration,previous_artifacts=engine.CALIBRATION_FILE,engine.ARTIFACT_DIR
    engine.CALIBRATION_FILE=disabled
    engine.ARTIFACT_DIR=OUT/"engine_runs"
    try:
        collect(pd.date_range("2025-12-01","2025-12-16"),"train")
        frozen=fit()
        print("FIT_FROZEN "+json.dumps(frozen),flush=True)
        collect(pd.date_range("2025-12-17","2025-12-23"),"evaluation")
        evaluate(frozen)
        status("complete",report="report.json")
    except Exception as exc:
        status("incomplete_error",error=type(exc).__name__,message=str(exc))
        raise
    finally:
        timer.cancel()
        engine.CALIBRATION_FILE=previous_calibration
        engine.ARTIFACT_DIR=previous_artifacts


if __name__=="__main__":
    main()
