"""Synthetic-only regression tests for the linear model and split contracts.

Run with ``python -m unittest discover -s training -p 'test_*.py'``.
These tests create all observations in memory and never open a dataset or a
locked-speaker list from the workspace.
"""

from __future__ import annotations

import unittest
import pickle
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge

import data as D
import modeling as M
import pipeline as P
import evaluate as E


class TwoStageRidgeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(819)
        # Nonzero means/intercepts expose incorrect intercept folding.
        self.X = rng.normal(loc=0.4, size=(72, 206))
        self.Xdev = rng.normal(loc=-0.2, size=(24, 206))
        signal = np.zeros(206)
        signal[:4] = [1.0, -0.7, 0.3, 0.9]
        signal[14:18] = [0.8, -0.2, 0.1, 0.4]
        self.y = 6.5 + self.X @ signal + rng.normal(0, 0.15, 72)
        self.ydev = 6.5 + self.Xdev @ signal + rng.normal(0, 0.15, 24)

    def test_fold_matches_serial_on_unseen_rows_and_is_one_linear_model(self):
        acoustic = Ridge(alpha=0.8).fit(self.X[:, :14], self.y)
        residual = self.y - acoustic.predict(self.X[:, :14])
        embedding = Ridge(alpha=4.0).fit(self.X[:, 14:], residual)

        folded = M.fold_two_stage(acoustic, embedding)

        serial = acoustic.predict(self.Xdev[:, :14]) + embedding.predict(
            self.Xdev[:, 14:]
        )
        self.assertLess(float(np.max(np.abs(folded.predict(self.Xdev) - serial))), 1e-9)
        assert_array_equal(folded.coef_, np.concatenate([acoustic.coef_, embedding.coef_]))
        self.assertEqual(folded.intercept_, acoustic.intercept_ + embedding.intercept_)
        self.assertEqual(folded.n_features_in_, 206)
        self.assertIsInstance(folded, Ridge)
        # A deployable folded predictor must not depend on stage estimators.
        self.assertFalse(any(isinstance(value, BaseEstimator) for value in vars(folded).values()))

        restored = pickle.loads(pickle.dumps(folded))
        assert_array_equal(restored.predict(self.Xdev), folded.predict(self.Xdev))
        self.assertFalse(any(isinstance(value, BaseEstimator) for value in vars(restored).values()))

    def test_both_stages_use_the_same_weights_and_true_acoustic_residual(self):
        weights = np.ones(len(self.y))
        weights[:24] = 2.0
        alpha = 2.75
        acoustic = Ridge(alpha=alpha).fit(self.X[:, :14], self.y, sample_weight=weights)
        embedding = Ridge(alpha=alpha).fit(
            self.X[:, 14:], self.y - acoustic.predict(self.X[:, :14]),
            sample_weight=weights,
        )

        folded, metrics, verification = M.fit_two_stage_ridge(
            self.X, self.y, self.Xdev, self.ydev,
            sample_weight=weights, alphas=(alpha,),
        )

        assert_allclose(folded.coef_, np.r_[acoustic.coef_, embedding.coef_], atol=1e-12, rtol=1e-12)
        assert_allclose(folded.intercept_, acoustic.intercept_ + embedding.intercept_, atol=1e-12)
        self.assertEqual(metrics["alpha_acoustic"], alpha)
        self.assertEqual(metrics["alpha_embedding"], alpha)
        self.assertTrue(verification["passed"])
        self.assertLess(verification["max_abs_error"], 1e-9)
        self.assertEqual(verification["n_features"], 206)

    def test_integer_weights_equal_repeated_training_observations(self):
        weights = np.ones(len(self.y), dtype=int)
        weights[::3] = 2
        weighted, _, _ = M.fit_two_stage_ridge(
            self.X, self.y, self.Xdev, self.ydev,
            sample_weight=weights, alphas=(1.5,),
        )
        repeated, _, _ = M.fit_two_stage_ridge(
            np.repeat(self.X, weights, axis=0), np.repeat(self.y, weights),
            self.Xdev, self.ydev, alphas=(1.5,),
        )

        assert_allclose(weighted.predict(self.Xdev), repeated.predict(self.Xdev), rtol=1e-10, atol=1e-10)

    def test_dev_labels_do_not_enter_either_stage_fit(self):
        # With a fixed alpha, changing every dev target may change metrics but
        # must not change fitted coefficients, intercepts, or predictions.
        original, _, _ = M.fit_two_stage_ridge(
            self.X, self.y, self.Xdev, self.ydev, alphas=(2.0,),
        )
        changed, _, _ = M.fit_two_stage_ridge(
            self.X, self.y, self.Xdev, -1000.0 * self.ydev, alphas=(2.0,),
        )

        assert_array_equal(original.coef_, changed.coef_)
        self.assertEqual(original.intercept_, changed.intercept_)

    def test_two_stage_rejects_layout_changes(self):
        for dimension in (14, 205, 207, 398):
            with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                M.fit_two_stage_ridge(
                    np.zeros((72, dimension)), self.y,
                    np.zeros((24, dimension)), self.ydev,
                    alphas=(1.0,),
                )

    def test_fold_rejects_incorrect_stage_feature_counts(self):
        acoustic = Ridge().fit(self.X[:, :13], self.y)
        embedding = Ridge().fit(self.X[:, 14:], self.y)
        with self.assertRaises(ValueError):
            M.fold_two_stage(acoustic, embedding)

    def test_invalid_sample_weights_fail_before_training(self):
        invalid_weights = {
            "all_zero": np.zeros(len(self.y)),
            "negative": np.r_[-1.0, np.ones(len(self.y) - 1)],
            "nan": np.r_[np.nan, np.ones(len(self.y) - 1)],
            "infinite": np.r_[np.inf, np.ones(len(self.y) - 1)],
            "wrong_row_count": np.ones(len(self.y) - 1),
            "wrong_rank": np.ones((len(self.y), 1)),
        }
        for name, weights in invalid_weights.items():
            with self.subTest(weights=name), self.assertRaises(ValueError):
                M.fit_two_stage_ridge(
                    self.X, self.y, self.Xdev, self.ydev,
                    sample_weight=weights, alphas=(1.0,),
                )


class TwoStageIndicatorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(438)
        def observations(n, location):
            return np.column_stack([
                rng.normal(location, 1.0, size=(n, 206)),
                rng.integers(0, 2, size=(n, 14)),
            ])
        self.X = observations(84, 0.6)
        self.Xdev = observations(28, -0.3)
        self.Xunseen = observations(19, 1.2)
        coefficients = np.zeros(220)
        coefficients[[0, 5, 14, 17, 206, 215]] = [1.3, -0.7, 0.8, 0.3, 2.1, -1.4]
        self.y = 4.7 + self.X @ coefficients + rng.normal(0, 0.1, len(self.X))
        self.ydev = 4.7 + self.Xdev @ coefficients + rng.normal(0, 0.1, len(self.Xdev))

    @staticmethod
    def acoustic_inputs(X):
        # Assemble independently of modeling.py's indexing implementation.
        return np.column_stack([X[:, :14], X[:, 206:220]])

    def test_weighted_fit_and_fold_match_independent_serial_models_on_unseen_rows(self):
        weights = np.where(np.arange(len(self.y)) % 3 == 0, 2.0, 1.0)
        alpha = 2.75
        acoustic = Ridge(alpha=alpha).fit(self.acoustic_inputs(self.X), self.y,
                                          sample_weight=weights)
        residual = self.y - acoustic.predict(self.acoustic_inputs(self.X))
        embedding = Ridge(alpha=alpha).fit(self.X[:, 14:206], residual,
                                           sample_weight=weights)
        folded, metrics, verification = M.fit_two_stage_ridge(
            self.X, self.y, self.Xdev, self.ydev, sample_weight=weights,
            alphas=(alpha,), missing_indicators=True,
        )
        direct_fold = M.fold_two_stage(acoustic, embedding, missing_indicators=True)

        expected_coefficients = np.r_[acoustic.coef_[:14], embedding.coef_, acoustic.coef_[14:]]
        assert_allclose(folded.coef_, expected_coefficients, atol=1e-12, rtol=1e-12)
        assert_allclose(folded.intercept_, acoustic.intercept_ + embedding.intercept_, atol=1e-12)
        self.assertEqual(folded.n_features_in_, 220)
        self.assertEqual(metrics["alpha_acoustic"], alpha)
        self.assertEqual(metrics["alpha_embedding"], alpha)
        self.assertEqual(verification["n_features"], 220)
        self.assertEqual(verification["stage1_columns"], list(range(14)) + list(range(206, 220)))
        self.assertEqual(verification["stage2_columns"], list(range(14, 206)))
        self.assertTrue(verification["passed"])
        self.assertLess(verification["max_abs_error"], 1e-9)
        restored = pickle.loads(pickle.dumps(folded))
        for estimator in (folded, direct_fold, restored):
            self.assertIsInstance(estimator, Ridge)
            self.assertFalse(any(isinstance(value, BaseEstimator) for value in vars(estimator).values()))
            for X in (self.X, self.Xdev, self.Xunseen):
                serial = acoustic.predict(self.acoustic_inputs(X)) + embedding.predict(X[:, 14:206])
                self.assertLess(float(np.max(np.abs(estimator.predict(X) - serial))), 1e-9)
        assert_array_equal(restored.predict(self.Xunseen), folded.predict(self.Xunseen))

    def test_dev_targets_do_not_fit_either_stage_with_fixed_alpha(self):
        original, _, _ = M.fit_two_stage_ridge(
            self.X, self.y, self.Xdev, self.ydev,
            alphas=(1.5,), missing_indicators=True,
        )
        changed, _, _ = M.fit_two_stage_ridge(
            self.X, self.y, self.Xdev, -1000.0 * self.ydev + 70,
            alphas=(1.5,), missing_indicators=True,
        )
        assert_array_equal(original.coef_, changed.coef_)
        self.assertEqual(original.intercept_, changed.intercept_)
        assert_array_equal(original.predict(self.Xunseen), changed.predict(self.Xunseen))

    def test_220_requires_opt_in_and_opt_in_rejects_every_other_width(self):
        with self.assertRaisesRegex(ValueError, "exactly 206"):
            M.fit_two_stage_ridge(self.X, self.y, self.Xdev, self.ydev, alphas=(1.0,))
        for dimension in (14, 206, 219, 221, 398):
            for split in ("train", "dev"):
                train, dev = self.X, self.Xdev
                if split == "train":
                    train = np.zeros((len(train), dimension))
                else:
                    dev = np.zeros((len(dev), dimension))
                with self.subTest(dimension=dimension, split=split), mock.patch.object(
                    Ridge, "fit", side_effect=AssertionError("invalid layout must fail before fit")
                ), self.assertRaisesRegex(ValueError, "exactly 220"):
                    M.fit_two_stage_ridge(train, self.y, dev, self.ydev,
                                         alphas=(1.0,), missing_indicators=True)

    def test_nonbinary_indicators_and_nonfinite_inputs_fail_before_fit(self):
        corruptions = [(206, -1.0), (219, 0.5), (210, 2.0),
                       (0, np.nan), (15, np.inf), (219, np.nan)]
        for column, value in corruptions:
            for split in ("train", "dev"):
                train, dev = self.X.copy(), self.Xdev.copy()
                (train if split == "train" else dev)[0, column] = value
                with self.subTest(column=column, value=value, split=split), mock.patch.object(
                    Ridge, "fit", side_effect=AssertionError("invalid values must fail before fit")
                ), self.assertRaisesRegex(ValueError, "binary|finite"):
                    M.fit_two_stage_ridge(train, self.y, dev, self.ydev,
                                         alphas=(1.0,), missing_indicators=True)

    def test_fold_layout_is_explicit_and_stage_dimensions_are_strict(self):
        for n_acoustic, n_embedding, indicators in ((28, 192, False), (14, 192, True),
                                                   (27, 192, True), (28, 191, True),
                                                   (28, 206, True)):
            acoustic, embedding = Ridge(), Ridge()
            acoustic.coef_, acoustic.intercept_ = np.zeros(n_acoustic), 0.0
            embedding.coef_, embedding.intercept_ = np.zeros(n_embedding), 0.0
            with self.subTest(acoustic=n_acoustic, embedding=n_embedding, indicators=indicators), self.assertRaises(ValueError):
                M.fold_two_stage(acoustic, embedding, missing_indicators=indicators)


class BoundaryWeightTests(unittest.TestCase):
    def test_boundary_is_inclusive_in_native_cape_v_units_only(self):
        samples = {
            "y_severity": np.array([14.999, 15.0, 40.0, 40.001, 20.0, 3.0, 20.0]),
            "severity_kind": np.array(["cape_v"] * 4 + ["grbas", "grbas", "unknown"]),
        }

        assert_array_equal(
            M.boundary_sample_weights(samples, enabled=True),
            [1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
        )

    def test_disabled_weighting_preserves_sklearn_default(self):
        # Disabled code must not need labels or fit a different weighted path.
        self.assertIsNone(M.boundary_sample_weights({}, enabled=False))


class LeakageTests(unittest.TestCase):
    def test_locked_speakers_are_rejected_on_either_side(self):
        for train, dev in ((["locked"], ["dev"]), (["train"], ["locked"])):
            with self.subTest(train=train, dev=dev), self.assertRaises(SystemExit):
                D.assert_no_leak(np.array(train), np.array(dev), {"locked"})

    def test_train_dev_overlap_is_rejected_even_when_unlocked(self):
        with self.assertRaises(SystemExit):
            D.assert_no_leak(np.array(["shared"]), np.array(["shared"]), set())

    def test_group_split_keeps_all_recordings_and_augmentations_together(self):
        speakers = np.repeat([f"synthetic-{i:02d}" for i in range(32)], 3)
        synthetic = {
            "X": np.arange(len(speakers) * 398).reshape(len(speakers), 398),
            "speaker_id": speakers,
            "y_binary": np.repeat(np.arange(32) % 2, 3),
            "corpus": np.repeat(np.where(np.arange(32) < 16, "a", "b"), 3),
            "clip": np.tile(["vowel", "phrase", "vowel"], 32),
        }

        train, dev = D.speaker_grouped_split(synthetic, dev_frac=0.25, seed=7)

        self.assertFalse(set(train["speaker_id"]) & set(dev["speaker_id"]))
        self.assertEqual(len(train["X"]) + len(dev["X"]), len(speakers))
        for side in (train, dev):
            _, counts = np.unique(side["speaker_id"], return_counts=True)
            assert_array_equal(counts, np.full(len(counts), 3))
            self.assertEqual(set(side["clip"]), {"vowel", "phrase"})
        D.assert_no_leak(train["speaker_id"], dev["speaker_id"], {"synthetic-never-loaded"})


class PipelineGuardTests(unittest.TestCase):
    def test_phase2_requires_declared_input_order_before_preprocessing(self):
        with tempfile.TemporaryDirectory(prefix="vocal-modeling-test-") as directory:
            root = Path(directory)
            path = root / "synthetic_without_layout.npz"
            np.savez(
                path, X=np.zeros((12, 398)),
                speaker_id=np.array([f"synthetic-{i}" for i in range(12)]),
                y_severity=np.arange(12, dtype=float),
                severity_kind=np.full(12, "cape_v"),
                y_binary=np.arange(12) % 2, corpus=np.full(12, "synthetic"),
            )
            lock = root / "synthetic_lock.txt"
            lock.write_text("synthetic-not-in-input\n")
            configurations = [
                ["--models", "ridge_2stage"],
                ["--models", "ridge", "--boundary-weight"],
                ["--models", "ridge", "--task-aware", "--phrase-untrusted-columns", "0"],
                ["--models", "ridge", "--missing-indicators"],
                ["--models", "logreg", "--platt"],
                ["--models", "ridge", "--conformal"],
            ]
            for flags in configurations:
                args = ["pipeline.py", "--data", str(path), "--lock-test", str(lock),
                        "--out", str(root / "bundle"), *flags]
                with self.subTest(flags=flags), mock.patch.object(sys, "argv", args), mock.patch.object(
                    P.F.FeatureProcessor, "fit", side_effect=AssertionError("preprocessing must not fit")
                ) as fit:
                    with self.assertRaisesRegex(ValueError, "layout"):
                        P.main()
                    fit.assert_not_called()
            self.assertFalse((root / "bundle").exists())

    def test_locked_input_fails_before_preprocessing_is_fitted(self):
        with tempfile.TemporaryDirectory(prefix="vocal-modeling-test-") as directory:
            root = Path(directory)
            speakers = np.array(["locked"] + [f"synthetic-{i}" for i in range(11)])
            path = root / "synthetic_train.npz"
            np.savez(
                path, X=np.zeros((12, 398)), speaker_id=speakers,
                y_severity=np.arange(12, dtype=float),
                severity_kind=np.full(12, "cape_v"),
                y_binary=np.arange(12) % 2, corpus=np.full(12, "synthetic"),
            )
            lock = root / "synthetic_lock.txt"
            lock.write_text("locked\n")
            args = ["pipeline.py", "--data", str(path), "--lock-test", str(lock),
                    "--models", "ridge", "--out", str(root / "bundle")]
            with mock.patch.object(sys, "argv", args), mock.patch.object(
                P.F.FeatureProcessor, "fit", side_effect=AssertionError("preprocessing must not fit")
            ) as fit:
                with self.assertRaisesRegex(SystemExit, "LOCKED TEST SPEAKERS"):
                    P.main()
                fit.assert_not_called()
            self.assertFalse((root / "bundle").exists())

    def test_lock_hash_mismatch_fails_before_evaluation_data_is_opened(self):
        with tempfile.TemporaryDirectory(prefix="vocal-modeling-test-") as directory:
            root = Path(directory)
            lock = root / "synthetic_lock.txt"
            lock.write_text("synthetic-evaluation-speaker\n")
            (root / "manifest.json").write_text(json.dumps({"lock_test_sha256": "wrong"}))
            args = ["evaluate.py", "--bundle", str(root), "--model", "ridge",
                    "--lock-test", str(lock), "--test-npz", str(root / "never-open.npz"),
                    "--report", str(root / "never-written.json")]
            with mock.patch.object(sys, "argv", args), mock.patch.object(
                E.D, "load_npz", side_effect=AssertionError("evaluation data must not open")
            ) as load:
                with self.assertRaisesRegex(SystemExit, "lock hash mismatch"):
                    E.main()
                load.assert_not_called()

    def test_model_hash_mismatch_fails_before_evaluation_data_is_opened(self):
        with tempfile.TemporaryDirectory(prefix="vocal-modeling-test-") as directory:
            root = Path(directory)
            lock = root / "synthetic_lock.txt"
            lock.write_text("synthetic-evaluation-speaker\n")
            filenames = ["ridge.pkl", "feature_processor.json", "feature_processor.npz",
                         "scaler.npz", "severity_norm.json"]
            hashes = {}
            for filename in filenames:
                path = root / filename
                path.write_bytes(b"synthetic placeholder: rejected before parsing")
                hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = {
                "schema_version": 2,
                "lock_test_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
                "models": ["ridge"], "deployable_models": ["ridge"],
                "artifact_sha256": hashes,
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "ridge.pkl").write_bytes(b"changed model bytes")
            args = ["evaluate.py", "--bundle", str(root), "--model", "ridge",
                    "--lock-test", str(lock), "--test-npz", str(root / "never-open.npz"),
                    "--report", str(root / "never-written.json")]
            with mock.patch.object(sys, "argv", args), mock.patch.object(
                E.D, "load_npz", side_effect=AssertionError("evaluation data must not open")
            ) as load:
                with self.assertRaisesRegex(ValueError, "artifact hash mismatch: ridge.pkl"):
                    E.main()
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
