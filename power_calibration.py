"""Frozen affine correction fitted on earlier issued-weather forecast errors."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _stamp(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Calibration timestamps must include UTC offsets")
    return stamp.tz_convert("UTC")


def apply_calibration(predicted, valid_times, issue_time, turbine, source_sha256,
                      base_model_version, timezone_assumption, artifact_path):
    """Apply a frozen artifact only when its training targets were available.

    A missing artifact or earlier historical issue uses the uncalibrated model
    with an explicit status. An incompatible or malformed installed artifact is
    an error, so a broken calibration cannot silently pass as an updated model.
    """
    predicted = np.asarray(predicted, dtype=float)
    index = pd.DatetimeIndex(valid_times)
    if (index.tz is None or len(index) != len(predicted) or predicted.ndim != 1
            or not np.isfinite(predicted).all() or not ((0 <= predicted) & (predicted <= 1)).all()):
        raise ValueError("Invalid power or timestamps supplied to calibration")
    issued = _stamp(issue_time)
    path = Path(artifact_path)
    if not path.exists():
        return predicted.copy(), {"applied": False, "status": "not_configured",
                                  "reason": "No frozen calibration artifact is installed"}
    raw = path.read_bytes()
    config = json.loads(raw)
    if config.get("schema_version") != 1 or config.get("method") != "ridge_affine":
        raise ValueError("Unknown power calibration schema or method")
    if config.get("promotion_rule_passed") is not True:
        raise ValueError("Calibration did not pass its predeclared evaluation rule")
    if config.get("power_unit") != "normalized":
        raise ValueError("Calibration must use normalized power")
    if config.get("base_model_version") != base_model_version:
        raise ValueError("Calibration was fitted for another base model version")
    if config.get("timezone_assumption") != timezone_assumption:
        raise ValueError("Calibration timezone differs from SCADA interpretation")
    if config.get("training_source_sha256", {}).get(turbine) != source_sha256:
        raise ValueError("Calibration SCADA fingerprint differs from the current source")
    fit_start, fit_end, available = (_stamp(config[key]) for key in ("fit_start", "fit_end", "available_at"))
    if fit_start > fit_end or fit_end + pd.Timedelta(hours=1) > available:
        raise ValueError("Calibration contains targets incomplete at its availability cutoff")
    for key in ("latest_target_available_at", "latest_weather_archive_available_at"):
        if config.get(key) and _stamp(config[key]) > available:
            raise ValueError("Calibration input evidence is later than its availability cutoff")
    digest = hashlib.sha256(raw).hexdigest()
    if available > issued:
        return predicted.copy(), {"applied": False, "status": "not_available_at_issue",
                                  "reason": "Calibration targets were not yet available at this historical decision",
                                  "available_at": available.isoformat(), "artifact_sha256": digest}
    parameters = config.get("parameters", {}).get(turbine)
    if not isinstance(parameters, dict) or set(parameters) != {"1-24", "25-48"}:
        raise ValueError("Calibration must provide both forecast lead bands")
    checked = {}
    for band, values in parameters.items():
        slope, intercept, count = (values.get(key) for key in ("slope", "intercept", "training_rows"))
        if (isinstance(slope, bool) or isinstance(intercept, bool)
                or not isinstance(slope, (int, float)) or not isinstance(intercept, (int, float))
                or not np.isfinite([slope, intercept]).all() or not 0 <= slope <= 2
                or type(count) is not int or count < 48):
            raise ValueError("Invalid or insufficiently supported calibration coefficients")
        checked[band] = {"slope": float(slope), "intercept": float(intercept), "training_rows": count}
    leads = np.asarray((index.tz_convert("UTC") - issued) / pd.Timedelta(hours=1), dtype=float)
    if not ((leads >= 1) & (leads <= 48) & (leads == np.floor(leads))).all():
        return predicted.copy(), {"applied": False, "status": "unsupported_lead_time",
                                  "reason": "Calibration was fitted only for integer decision lead hours 1-48",
                                  "artifact_sha256": digest}
    corrected = predicted.copy()
    for band, mask in (("1-24", leads <= 24), ("25-48", leads > 24)):
        values = checked[band]
        corrected[mask] = np.clip(values["slope"] * predicted[mask] + values["intercept"], 0, 1)
    metadata = {"applied": True, "status": "applied", "method": config["method"],
                "version": config["version"], "artifact_sha256": digest,
                "fit_start": fit_start.isoformat(), "fit_end": fit_end.isoformat(),
                "available_at": available.isoformat(),
                "training_rows": sum(values["training_rows"] for values in checked.values()),
                "parameters": checked,
                "limitations": ["Frozen correction fitted on a short January period",
                                "Corrects power forecast bias; does not calibrate the weather model itself",
                                "Future-period improvement is not guaranteed"]}
    return corrected, metadata
