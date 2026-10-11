"""Synthetic-only calibration tests: never open any corpus or locked data."""

import json
import hashlib
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

import calibration as C
import evaluate as E
import features as F
import verify_layout as V
from make_smoke_data import to_canonical_layout


class PlattTests(unittest.TestCase):
    def test_fold_is_one_linear_head_and_preserves_original(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(80, 4))
        y = (X[:, 0] + rng.normal(size=80) > 0).astype(int)
        model = LogisticRegression().fit(X, y)
        original = model.coef_.copy()
        params = {"slope": 0.4, "intercept": -0.7}
        folded = C.fold_platt(model, params)
        self.assertEqual(folded.coef_.shape, (1, 4))
        np.testing.assert_array_equal(model.coef_, original)
        np.testing.assert_allclose(folded.predict_proba(X)[:, 1],
                                   C.apply_platt(model.decision_function(X), params),
                                   atol=1e-14, rtol=0)

    def test_overconfidence_is_corrected_and_parameters_round_trip(self):
        rng = np.random.default_rng(10)
        latent = rng.normal(size=4000)
        y = rng.binomial(1, expit(latent + 0.5))
        logits = latent * 4
        params = json.loads(json.dumps(C.fit_platt(logits, y), allow_nan=False))
        calibrated = C.apply_platt(logits, params)
        before = C.reliability_report(y, expit(logits))
        after = C.reliability_report(y, calibrated)
        self.assertLess(after["log_loss"], before["log_loss"])
        self.assertLess(after["brier_score"], before["brier_score"])
        self.assertAlmostEqual(after["roc_auc"], before["roc_auc"])
        self.assertGreater(params["slope"], 0)
        self.assertLess(params["slope"], 0.5)
        self.assertEqual(sum(b["count"] for b in after["reliability_bins"]), len(y))

    def test_constant_logits_produce_finite_prior(self):
        logits = np.ones(10)
        y = np.array([0] * 7 + [1] * 3)
        params = C.fit_platt(logits, y)
        probabilities = C.apply_platt(logits, params)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(np.allclose(probabilities, probabilities[0]))

    def test_bad_inputs_fail(self):
        with self.assertRaisesRegex(ValueError, "both"):
            C.fit_platt([0, 1], [0, 0])
        with self.assertRaisesRegex(ValueError, "finite"):
            C.apply_platt([np.nan], {"slope": 1, "intercept": 0})
        with self.assertRaisesRegex(ValueError, "match"):
            C.fit_platt([0, 1], [0, 2])

    def test_reliability_edges_and_single_class_auc(self):
        report = C.reliability_report([0, 0], [0, 1], n_bins=4)
        self.assertEqual([b["count"] for b in report["reliability_bins"]], [1, 0, 0, 1])
        self.assertIsNone(report["reliability_bins"][1]["fraction_positive"])
        self.assertIsNone(report["roc_auc"])
        self.assertTrue(np.isfinite(report["log_loss"]))
        json.dumps(report, allow_nan=False)


class ThresholdTests(unittest.TestCase):
    def test_selects_maximum_j_under_sensitivity_constraint(self):
        y = np.array([0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1])
        p = np.array([0.7, 0.9, 0.4, 0.8, 0.2, 0.9, 0.7, 0.1, 0.6, 0.6, 0.3])
        chosen = C.select_operating_threshold(y, p)
        candidates = [C.operating_point_report(y, p, threshold) for threshold in np.unique(p)]
        valid = [r for r in candidates if r["sensitivity"] >= 0.85]
        oracle = max(valid, key=lambda r: (r["youden_j"], r["specificity"], r["threshold"]))
        self.assertEqual(chosen["threshold"], oracle["threshold"])
        self.assertIn(chosen["threshold"], p)
        self.assertGreaterEqual(chosen["sensitivity"], 0.85)
        self.assertTrue(np.isfinite(chosen["threshold"]))

    def test_tied_scores_use_observed_finite_threshold(self):
        chosen = C.select_operating_threshold([0, 1, 0, 1], [0.5] * 4)
        self.assertEqual(chosen["threshold"], 0.5)
        self.assertEqual(chosen["sensitivity"], 1)
        self.assertEqual(chosen["specificity"], 0)


class ConformalTests(unittest.TestCase):
    def test_conservative_order_statistic_uses_speakers_not_clips(self):
        speakers = np.repeat([f"synthetic_{i}" for i in range(20)], 2)
        residuals = np.repeat(np.arange(1, 21, dtype=float), 2)
        residuals[::2] = 0
        params = C.fit_conformal(residuals, np.zeros(40), speakers)
        self.assertEqual(params["rank"], 19)
        self.assertEqual(params["q"], 19)
        self.assertEqual(params["n_calibration_units"], 20)
        self.assertEqual(params["n_calibration_clips"], 40)
        lo, hi = C.conformal_interval(np.array([0, 20]), params)
        np.testing.assert_array_equal(lo, [-19, 1])
        np.testing.assert_array_equal(hi, [19, 39])
        json.dumps(params, allow_nan=False)

    def test_tiny_sample_has_unbounded_not_interpolated_radius(self):
        params = C.fit_conformal([1, 2], [0, 0], ["a", "b"])
        self.assertEqual(params["status"], "unbounded")
        self.assertIsNone(params["q"])
        self.assertTrue(params["q_is_infinite"])
        lo, hi = C.conformal_interval([3], params)
        self.assertEqual(lo[0], -np.inf)
        self.assertEqual(hi[0], np.inf)
        json.dumps(params, allow_nan=False)

    def test_empty_sample_is_explicitly_unavailable(self):
        params = C.fit_conformal([], [], [])
        self.assertEqual(params["status"], "unavailable")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            C.conformal_interval([0], params)

    def test_duplicate_clips_do_not_increase_calibration_units(self):
        params = C.fit_conformal(np.ones(100), np.zeros(100), ["one_speaker"] * 100)
        self.assertEqual(params["n_calibration_units"], 1)
        self.assertEqual(params["status"], "unbounded")

    def test_native_interval_and_both_coverage_units(self):
        params = C.fit_conformal(np.ones(10), np.zeros(10), [str(i) for i in range(10)])
        report = E.conformal_report(np.zeros(3), [49, 65, 60], ["a", "a", "b"],
                                    params, "synthetic", {"synthetic": (50, 10)})
        self.assertEqual(report["q_native"], 10)
        self.assertEqual(report["empirical_clip_coverage"], 2 / 3)
        self.assertEqual(report["empirical_speaker_coverage"], 1 / 2)
        self.assertEqual(report["intervals"][0]["lower"], 40)
        self.assertEqual(report["intervals"][0]["upper"], 60)


class EvaluationTests(unittest.TestCase):
    def test_apply_only_saved_platt_and_operating_point(self):
        rng = np.random.default_rng(8)
        latent = rng.normal(size=(32, 398))
        X = to_canonical_layout(latent)
        y = (latent[:, 0] > 0).astype(int)
        corpus = np.array(["synthetic"] * len(X))
        speaker = np.array([f"synthetic_speaker_{i}" for i in range(len(X))])
        rows = {"X": X, "y_severity": 40 + 10 * latent[:, 0], "y_binary": y,
                "corpus": corpus, "severity_kind": np.array(["synthetic"] * len(X)),
                "speaker_id": speaker,
                "feature_layout": np.full(len(X), V.ACOUSTIC_FIRST)}
        V.assert_canonical_input(rows, label="synthetic calibration fixture")
        processor = F.FeatureProcessor("F3").fit(rows)
        model = LogisticRegression().fit(processor.transform(rows), y)
        params = {"slope": 0.4, "intercept": -0.8}
        with tempfile.TemporaryDirectory(prefix="synthetic_calibration_") as tmp:
            root = Path(tmp)
            processor.save(root)
            np.savez(root / "scaler.npz", mean_=processor.scaler.mean_,
                     scale_=processor.scaler.scale_, feature_set="F3")
            (root / "logreg.pkl").write_bytes(pickle.dumps(model))
            (root / "severity_norm.json").write_text(json.dumps({"synthetic": [40, 10]}))
            (root / "calibration.json").write_text(json.dumps({"schema_version": 1, "models": {
                "logreg": {"platt": params, "operating_point": {"threshold": 0.7}}}}))
            lock = root / "synthetic_lock.txt"
            lock.write_text("\n".join(speaker))
            names = ["feature_processor.json", "feature_processor.npz", "scaler.npz",
                     "logreg.pkl", "severity_norm.json", "calibration.json"]
            manifest = {"schema_version": 2, "models": ["logreg"],
                        "deployable_models": ["logreg"], "feature_set": "F3",
                        "input_layout_provenance": V.LAYOUT_PROVENANCE,
                        "lock_test_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
                        "artifact_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                            for name in names}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            args = ["evaluate.py", "--bundle", tmp, "--model", "logreg", "--test-npz", "synthetic_fixture",
                    "--lock-test", str(lock), "--report", str(root / "report.json"),
                    "--native-corpus", "synthetic"]
            with patch("sys.argv", args), patch.object(E.D, "load_npz", return_value=rows), \
                    patch.object(E, "bootstrap_ci", return_value=(0.0, 1.0)), \
                    patch.object(C, "fit_platt", side_effect=AssertionError("must not fit")), \
                    patch.object(C, "fit_conformal", side_effect=AssertionError("must not fit")), \
                    patch.object(C, "select_operating_threshold", side_effect=AssertionError("must not select")), \
                    patch("builtins.print"):
                E.main()
            report = json.loads((root / "report.json").read_text())
            self.assertTrue(report["classification"]["platt_applied"])
            self.assertEqual(report["classification"]["operating_point"]["threshold"], 0.7)
            expected = C.reliability_report(y, C.apply_platt(model.decision_function(processor.transform(rows)), params))
            self.assertEqual(report["classification"]["after"]["brier_score"], expected["brier_score"])
            # Declared safe bundles must not silently accept undeclared input
            # whose 398 columns could have been assembled in another order.
            undeclared_rows = {key: value for key, value in rows.items() if key != "feature_layout"}
            with patch("sys.argv", args), patch.object(E.D, "load_npz", return_value=undeclared_rows), \
                    patch.object(F.FeatureProcessor, "transform") as transform:
                with self.assertRaisesRegex(ValueError, "feature_layout"):
                    E.main()
                transform.assert_not_called()
            # A changed calibrator is rejected before any synthetic data loader call.
            (root / "calibration.json").write_text("{}")
            with patch("sys.argv", args), patch.object(E.D, "load_npz") as loader:
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    E.main()
                loader.assert_not_called()

    def test_lgbm_is_only_a_development_probe(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_probe_") as tmp:
            manifest = {"schema_version": 2, "models": ["lgbm"], "deployable_models": []}
            with self.assertRaisesRegex(ValueError, "development-only probe"):
                E.verify_bundle_artifacts(tmp, manifest, "lgbm")

    def test_required_calibration_cannot_silently_disappear(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_missing_calibration_") as tmp:
            names = ["ridge.pkl", "feature_processor.json", "feature_processor.npz",
                     "scaler.npz", "severity_norm.json"]
            manifest = {"schema_version": 2, "models": ["ridge"], "deployable_models": ["ridge"],
                        "calibration": {"conformal": True},
                        "artifact_sha256": {name: "not_used" for name in names}}
            with self.assertRaisesRegex(ValueError, "missing artifact hashes.*calibration.json"):
                E.verify_bundle_artifacts(tmp, manifest, "ridge")

    def test_required_head_parameters_are_validated_before_evaluation(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_missing_head_") as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps({"schema_version": 1, "models": {}}))
            for name, flag in [("ridge", "conformal"), ("ridge_2stage", "conformal"), ("logreg", "platt")]:
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, "missing"):
                        E.load_calibration(tmp, {"calibration": {flag: True}}, name)


if __name__ == "__main__":
    unittest.main()
