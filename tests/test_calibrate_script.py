"""Chronology and reproducibility controls for calibration development."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scripts import calibrate


def training_fixture():
    rows=[]
    for turbine in ("turbine_1","turbine_2"):
        for band in ("1-24","25-48"):
            for i,stamp in enumerate(pd.date_range("2026-01-01", periods=96, freq="h",tz="UTC")):
                raw=i/95
                rows.append({"split":"train","turbine":turbine,"lead_band":band,
                             "valid_time":stamp.isoformat(),"raw_power":raw,"actual":max(0,.5*raw-.02)})
    return pd.DataFrame(rows)


class CalibrationExperimentTests(unittest.TestCase):
    def test_future_target_is_rejected(self):
        training=training_fixture()
        training.loc[0,"valid_time"]=calibrate.FIT_CUTOFF.isoformat()
        with self.assertRaisesRegex(ValueError,"unavailable"):
            calibrate.fit_parameters(training)

    def test_evaluation_labels_cannot_enter_fit(self):
        training=training_fixture()
        training.loc[0,"split"]="evaluation"
        with self.assertRaisesRegex(ValueError,"training rows"):
            calibrate.fit_parameters(training)

    def test_lead_bands_are_applied_separately_and_bounded(self):
        frame=pd.DataFrame({"turbine":["turbine_1"]*4,"lead_band":["1-24"]*2+["25-48"]*2,
                            "raw_power":[0,1,0,1]})
        params=[{"turbine":"turbine_1","lead_band":"1-24","a":2,"b":-.25,"bias":0},
                {"turbine":"turbine_1","lead_band":"25-48","a":0,"b":.4,"bias":0}]
        result=calibrate.predict(frame,params)
        np.testing.assert_allclose(result.ridge_affine,[0,1,.4,.4])
        self.assertNotIn("ridge_affine",frame)

    def test_failed_replay_restores_production_configuration(self):
        previous_file=calibrate.engine.CALIBRATION_FILE
        previous_artifacts=calibrate.engine.ARTIFACT_DIR
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(calibrate,"collect",side_effect=RuntimeError("weather unavailable")):
                with self.assertRaisesRegex(RuntimeError,"weather unavailable"):
                    calibrate.reproduce(Path(directory))
        self.assertEqual(calibrate.engine.CALIBRATION_FILE,previous_file)
        self.assertEqual(calibrate.engine.ARTIFACT_DIR,previous_artifacts)

    def test_pool_improvement_cannot_hide_regressed_group(self):
        pooled={"methods":{"raw_power":{"mae":.2,"rmse":.3},"ridge_affine":{"mae":.15,"rmse":.2}}}
        groups=[{"methods":{"raw_power":{"mae":.1},"ridge_affine":{"mae":.15}}}]
        self.assertFalse(calibrate.promotion_passed(pooled,groups))


if __name__=="__main__":
    unittest.main()
