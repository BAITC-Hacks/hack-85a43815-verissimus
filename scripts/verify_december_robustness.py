"""Independent arithmetic and chronology audit of the completed robustness run."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument("--output",type=Path)
arguments=parser.parse_args()
PROJECT=arguments.project.resolve()
ROOT=(arguments.output or PROJECT/"artifacts"/"december_robustness").resolve()
status=json.loads((ROOT/"status.json").read_text(encoding="utf-8"))
assert status["status"]=="complete", "An incomplete run cannot be verified as successful"
protocol=json.loads((ROOT/"protocol.json").read_text(encoding="utf-8"))
report=json.loads((ROOT/"report.json").read_text(encoding="utf-8"))
frozen=json.loads((ROOT/"fitted_before_evaluation.json").read_text(encoding="utf-8"))
training=pd.read_csv(ROOT/"training_predictions.csv")
evaluation=pd.read_csv(ROOT/"evaluation_predictions.csv")
runs=json.loads((ROOT/"provenance.json").read_text(encoding="utf-8"))
assert report["protocol"]==protocol
assert len(runs)==46
assert sum(r["split"]=="train" for r in runs)==32
assert sum(r["split"]=="evaluation" for r in runs)==14
assert len({r["forecast_id"] for r in runs})==46
assert report["frozen_calibration"]==frozen
assert hashlib.sha256((ROOT/"training_predictions.csv").read_bytes()).hexdigest()==frozen["training_predictions_sha256"]
assert hashlib.sha256((ROOT/"evaluation_predictions.csv").read_bytes()).hexdigest()==report["evaluation_predictions_sha256"]
assert hashlib.sha256((ROOT/"protocol.json").read_bytes()).hexdigest()==report["protocol_sha256"]
assert training.split.eq("train").all() and evaluation.split.eq("evaluation").all()
assert len(evaluation)==7*48*2
assert not evaluation.duplicated(["turbine","issue_time","valid_time"]).any()
expected=set()
for turbine in ("turbine_1","turbine_2"):
    for day in pd.date_range("2025-12-17","2025-12-23"):
        start=day.tz_localize("Asia/Almaty").tz_convert("UTC")
        issue=start-pd.Timedelta(hours=1)
        expected.update((turbine,issue,valid) for valid in pd.date_range(start,periods=48,freq="h"))
actual=set(zip(evaluation.turbine,pd.to_datetime(evaluation.issue_time,utc=True),pd.to_datetime(evaluation.valid_time,utc=True)))
assert actual==expected, "The complete turbine/issue/valid grid must match the frozen protocol"
for column in ("raw_power","ridge_affine","persistence","train_mean"):
    assert np.isfinite(evaluation[column]).all()
    assert evaluation[column].between(0,1).all()
assert evaluation.actual.dropna().between(0,1).all()
cutoff=pd.Timestamp(report["protocol"]["training_target_end_cutoff"])
assert (pd.to_datetime(training.valid_time,utc=True)+pd.Timedelta(hours=1)<=cutoff).all()
assert pd.Timestamp(frozen["available_at"])<=pd.to_datetime(evaluation.issue_time,utc=True).min()
for run in runs:
    issue=pd.Timestamp(run["issue_time"])
    assert pd.Timestamp(run["model"]["training_available_at"])<=issue
    assert pd.Timestamp(run["weather"]["available_at"])<=issue
    assert pd.Timestamp(run["weather"]["run_time"])<=issue
    assert not run["model"]["calibration"]["applied"]
    assert all(pd.Timestamp(s["last_modified"])<=issue for s in run["weather"]["sources"])
    raw=json.loads((ROOT/run["raw_file"]).read_text(encoding="utf-8"))
    assert raw["forecast_id"]==run["forecast_id"]
    assert raw["weather"]["input_sha256"]==run["weather"]["input_sha256"]
    saved_points={pd.Timestamp(point["time"]):point["predicted_power"] for point in raw["forecast_sample"]}
    frame=training if run["split"]=="train" else evaluation
    rows=frame.loc[frame.forecast_id.eq(run["forecast_id"])]
    assert not rows.empty
    for row in rows.itertuples():
        assert row.turbine==run["turbine"] and pd.Timestamp(row.issue_time)==issue
        assert np.isclose(row.raw_power,saved_points[pd.Timestamp(row.valid_time)],rtol=0,atol=1e-12)
for param in frozen["parameters"]:
    subset=training[training.turbine.eq(param["turbine"]) & training.lead_band.eq(param["lead_band"])].dropna(subset=["actual"])
    x,y=subset.raw_power.to_numpy(),subset.actual.to_numpy()
    matrix=np.column_stack([x,np.ones(len(x))])
    a,b=np.linalg.solve(matrix.T@matrix+24*np.eye(2),matrix.T@y+24*np.array([1.,0.]))
    if not 0<=a<=2:
        a=float(np.clip(a,0,2));b=float(np.sum(y-a*x)/(len(x)+24))
    np.testing.assert_allclose([a,b],[param["slope"],param["intercept"]],rtol=0,atol=1e-12)
    assert param["training_rows"]==len(subset)
    group=evaluation[evaluation.turbine.eq(param["turbine"]) & evaluation.lead_band.eq(param["lead_band"])]
    np.testing.assert_allclose(group.ridge_affine,np.clip(a*group.raw_power+b,0,1),rtol=0,atol=1e-12)

def check_metrics(frame, recorded):
    paired=frame.dropna(subset=["actual"])
    assert recorded["forecast_rows"]==len(frame)
    assert recorded["scored_rows"]==len(paired)
    assert np.isclose(recorded["coverage"],len(paired)/len(frame),rtol=0,atol=1e-15)
    for method,values in recorded["methods"].items():
        residual=paired[method].to_numpy()-paired.actual.to_numpy()
        expected={"mae":np.abs(residual).mean(),"rmse":np.sqrt(np.mean(residual**2)),"bias":residual.mean()}
        for key,value in expected.items():
            assert np.isclose(values[key],value,rtol=0,atol=1e-12)

check_metrics(evaluation,report["pooled"])
assert len(report["metrics"])==4
assert {(g["turbine"],g["lead_band"]) for g in report["metrics"]}=={(t,b) for t in ("turbine_1","turbine_2") for b in ("1-24","25-48")}
for group in report["metrics"]:
    check_metrics(evaluation[evaluation.turbine.eq(group["turbine"]) & evaluation.lead_band.eq(group["lead_band"])],group)
result={"status":"passed","checks":["fixed protocol checksum","frozen coefficients","December-only fit target availability","raw model training cutoffs","all weather source availability","January calibration disabled","all672 evaluation slots","all coefficient fits and corrected predictions","all reported MAE/RMSE/bias and coverage"],"run_count":len(runs),"evaluation_rows":len(evaluation)}
(ROOT/"verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
