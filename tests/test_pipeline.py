"""Scientific invariants and a complete small training/inference run."""

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from arx_rpc.data import FeatureBank, ModelConfig, demo_frame, load_frame
from arx_rpc.metrics import error_analysis, evaluate_windows, metrics
from arx_rpc.models import ConservativeGate, ResidualTCN
from arx_rpc.pipeline import ARXRPC


class ScientificTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = demo_frame(days=8, seed=7)
        cls.config = ModelConfig(horizon=4, lookback=16, lags=[1, 4, 16], rolling_windows=[4, 16],
                                 dilations=[1, 2], channels=4, epochs=2, tree_iterations=10,
                                 train_stride=8, predict_stride=2, batch_size=32)
        cls.model = ARXRPC(cls.config).fit(cls.frame, "2025-01-05", "2025-01-07")

    def test_entire_forecast_chain(self):
        prediction, paths = self.model.predict(self.frame, [576, 578, 580])
        self.assertEqual(len(paths), 3 * self.config.horizon)
        self.assertTrue(np.isfinite(prediction[["arx", "rpc", "cs_rpc"]]).all().all())
        self.assertGreater(float(paths.tcn_residual.abs().sum()), 0)
        self.assertTrue((paths.origin_time <= paths.timestamp).all())
        self.assertTrue((paths.horizon_step.between(1, self.config.horizon)).all())

    def test_future_prices_and_covariates_cannot_change_current_forecast(self):
        origin = 590
        before, _ = self.model.predict(self.frame, [origin])
        changed = self.frame.copy()
        changed.loc[origin:, ["price", *self.config.features]] += 100000.0
        after, _ = self.model.predict(changed, [origin])
        np.testing.assert_allclose(before[["arx", "rpc", "cs_rpc"]], after[["arx", "rpc", "cs_rpc"]], atol=1e-8)

    def test_future_targets_not_required_for_forecasting(self):
        origin = 590
        full, _ = self.model.predict(self.frame, [origin])
        history_only, _ = self.model.predict(self.frame.iloc[:origin].copy(), [origin])
        np.testing.assert_allclose(full[["arx", "rpc", "cs_rpc"]], history_only[["arx", "rpc", "cs_rpc"]], atol=1e-8)

    def test_event_prices_not_used_for_model_selection(self):
        changed = self.frame.copy()
        changed.loc[576:, "price"] += 50000.0
        changed.loc[576:, self.config.features] += 50000.0
        refit = ARXRPC(self.config).fit(changed, "2025-01-05", "2025-01-07")
        np.testing.assert_allclose(self.model.anchor.coef_, refit.anchor.coef_)
        self.assertEqual(vars(self.model.gate), vars(refit.gate))
        self.assertEqual(self.model.alpha_tree, refit.alpha_tree)
        self.assertEqual(self.model.alpha_tcn, refit.alpha_tcn)
        for name, weight in self.model.temporal.state_dict().items():
            torch.testing.assert_close(weight, refit.temporal.state_dict()[name])

    def test_training_and_validation_target_ranges_do_not_overlap(self):
        bank = FeatureBank(self.frame, self.config)
        train = bank.phase_origins(0, 384, self.config.train_stride)
        validation = bank.phase_origins(384, 576, self.config.predict_stride)
        self.assertLess(int(train[-1] + self.config.horizon - 1), int(validation[0]))
        self.assertLess(int(validation[-1] + self.config.horizon - 1), 576)

    def test_temporal_encoder_is_causal(self):
        torch.manual_seed(3)
        model = ResidualTCN(2, 4, 4, [1, 2])
        first = torch.randn(1, 2, 20)
        second = first.clone()
        second[:, :, 12:] += 1000.0
        torch.testing.assert_close(model.encoder(first)[:, :, :12], model.encoder(second)[:, :, :12])

    def test_gate_is_convex_and_does_not_need_true_prices(self):
        arx = np.array([10., 20., 30., 40.])
        rpc = np.array([15., 15., 45., 10.])
        gate = ConservativeGate().fit(np.array([12., 18., 32., 20.]), arx, rpc, 0.9)
        pred, strength = gate.predict(arx, rpc)
        self.assertTrue(((strength >= 0) & (strength <= 1)).all())
        self.assertTrue((pred >= np.minimum(arx, rpc)).all())
        self.assertTrue((pred <= np.maximum(arx, rpc)).all())

    def test_metric_definitions(self):
        actual = np.array([10., 20., 30., 40., 50.])
        pred = np.array([12., 18., 34., 39., 42.])
        result = metrics(actual, pred)
        self.assertAlmostEqual(result["MAE"], 3.4)
        self.assertAlmostEqual(result["ExtremeMAE"], 8.0)
        self.assertAlmostEqual(result["AbsPeakBias"], 8.0)
        self.assertAlmostEqual(result["Coverage"], 1.0)

    def test_error_identity(self):
        prediction, _ = self.model.predict(self.frame, [580, 582, 584])
        analysis = error_analysis(prediction, self.frame)
        for name in ["rpc", "cs_rpc"]:
            np.testing.assert_allclose(analysis[f"{name}_squared_error_change"], analysis[f"{name}_analytic_change"], atol=1e-8)

    def test_missing_predictions_count_against_coverage(self):
        actual = np.array([10., 20., 30.])
        result = metrics(actual, np.array([11., np.nan, 29.]))
        self.assertAlmostEqual(result["Coverage"], 2 / 3)

    def test_persistence_and_frozen_parameters(self):
        audit_before = json.dumps(self.model.audit, sort_keys=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            self.model.save(path)
            loaded = ARXRPC.load(path)
            original, _ = self.model.predict(self.frame, [580])
            restored, _ = loaded.predict(self.frame, [580])
            pd.testing.assert_frame_equal(original, restored)
        self.assertEqual(audit_before, json.dumps(self.model.audit, sort_keys=True))

    def test_bad_time_grid_and_config_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            self.frame.drop(index=4).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_frame(path, self.config.features)
        with self.assertRaises(ValueError):
            ARXRPC(ModelConfig(horizon=4, predict_stride=8))

    def test_windows_use_requested_targets_and_all_variants(self):
        prediction, _ = self.model.predict(self.frame, [576, 578, 580, 582])
        window = [{"name": "fixed", "start": "2025-01-07 00:00:00", "end": "2025-01-07 02:00:00"}]
        report = evaluate_windows(prediction, window, self.frame)
        self.assertEqual(set(report.model), {"arx", "rpc", "cs_rpc"})
        self.assertTrue(report.Coverage.eq(1.0).all())


if __name__ == "__main__":
    unittest.main()
