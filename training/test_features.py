"""Synthetic-only feature contract checks; no repository NPZ data is opened."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    from . import data as D
    from .features import FeatureProcessor, select_features
except ImportError:
    import data as D
    from features import FeatureProcessor, select_features


def synthetic(n=8, seed=13):
    rng = np.random.default_rng(seed)
    return {
        "X": rng.normal(size=(n, 398)),
        "y_severity": rng.uniform(0, 100, n),
        "severity_kind": np.full(n, "cape_v"),
        "y_binary": np.arange(n) % 2,
        "speaker_id": np.array([f"synthetic-{i}" for i in range(n)]),
        "corpus": np.full(n, "synthetic"),
    }


class SpyImputer:
    """A test-only injected estimator; production deliberately chooses no policy."""

    def __init__(self):
        self.calls = 0

    def fit(self, X):
        self.calls += 1
        self.fit_input = X.copy()
        self.values = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        return np.where(np.isnan(X), self.values, X)


class FeatureProcessorTests(unittest.TestCase):
    def test_default_matches_phase1_exactly_for_every_feature_set(self):
        train, dev = synthetic(), synthetic(5, 22)
        for feature_set, width in (("F1", 398), ("F2", 206), ("F3", 14)):
            with self.subTest(feature_set=feature_set):
                reference = StandardScaler().fit(train["X"][:, :width])
                processor = FeatureProcessor(feature_set).fit(train)
                np.testing.assert_array_equal(processor.scaler.mean_, reference.mean_)
                np.testing.assert_array_equal(processor.transform(dev), reference.transform(dev["X"][:, :width]))

    def test_all_feature_sets_reject_changed_raw_layout(self):
        for feature_set in ("F1", "F2", "F3"):
            for shape in ((5, 206), (5, 400), (398,)):
                with self.subTest(feature_set=feature_set, shape=shape):
                    with self.assertRaisesRegex(ValueError, "398-column"):
                        select_features(np.zeros(shape), feature_set)

    def test_phrase_gating_uses_trusted_train_scaling_and_keeps_unknown_unmapped(self):
        train = synthetic(4)
        train["clip"] = np.array(["vowel", "vowel", "phrase", "unknown"])
        train["X"][:, 2] = [1.0, 3.0, 99999.0, 5.0]
        processor = FeatureProcessor("F2", task_aware=True,
                                     phrase_untrusted_columns=[2], missing_indicators=True).fit(train)
        self.assertEqual(processor.scaler.mean_[2], 3.0)
        output = processor.transform(train)
        self.assertEqual(output.shape, (4, 220))
        self.assertEqual(output[2, 2], 0.0)
        self.assertEqual(output[2, 206 + 2], 0.0)
        self.assertGreater(output[3, 2], 0.0)
        self.assertEqual(output[3, 206 + 2], 1.0)
        self.assertTrue(any("unknown task" in note for note in processor.metadata()["limitations"]))
        np.testing.assert_array_equal(np.unique(output[:, 206:]), [0.0, 1.0])

    def test_task_columns_must_be_explicit_valid_indexes(self):
        for columns in ([], [14], [-1], [True], [1.0], [2, 2]):
            with self.subTest(columns=columns):
                with self.assertRaises(ValueError):
                    FeatureProcessor(task_aware=True, phrase_untrusted_columns=columns)

    def test_real_nan_fails_closed_even_in_suppressed_phrase_column(self):
        train = synthetic()
        train["clip"] = np.full(8, "phrase")
        train["X"][0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "no approved imputer"):
            FeatureProcessor(task_aware=True, phrase_untrusted_columns=[1], missing_indicators=True).fit(train)

    def test_injected_imputer_and_scaler_fit_only_trusted_training_values(self):
        train = synthetic(4)
        train["clip"] = np.array(["vowel", "vowel", "phrase", "unknown"])
        train["X"][:, 1] = [1.0, np.nan, 10000.0, 5.0]
        imputer = SpyImputer()
        processor = FeatureProcessor(task_aware=True, phrase_untrusted_columns=[1],
                                     missing_indicators=True, imputer=imputer).fit(train)
        self.assertTrue(np.isnan(imputer.fit_input[1:3, 1]).all())
        self.assertEqual(imputer.calls, 1)
        self.assertEqual(processor.scaler.mean_[1], 3.0)
        self.assertEqual(imputer.values[1], 3.0)
        dev = synthetic(2, 31)
        dev["X"][:, 1] = [np.nan, 1e10]
        output = processor.transform(dev)
        self.assertTrue(np.isfinite(output).all())
        self.assertEqual(output[0, 207], 0.0)
        self.assertEqual(imputer.calls, 1)
        self.assertEqual(processor.scaler.mean_[1], 3.0)
        self.assertEqual(imputer.values[1], 3.0)

    def test_no_trusted_training_values_fail_closed(self):
        train = synthetic()
        train["clip"] = np.full(8, "phrase")
        with self.assertRaisesRegex(ValueError, "no trusted observed training values"):
            FeatureProcessor(task_aware=True, phrase_untrusted_columns=[1]).fit(train)
        train["X"][:, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "no trusted observed training values"):
            FeatureProcessor(missing_indicators=True, imputer=SpyImputer()).fit(train)

    def test_binary_indicators_respect_explicit_raw_availability(self):
        train = synthetic()
        train["acoustic_available"] = np.ones((8, 14), dtype=bool)
        train["acoustic_available"][0, 3] = False
        with self.assertRaisesRegex(ValueError, "no approved imputer"):
            FeatureProcessor(missing_indicators=True).fit(train)
        train["X"][0, 3] = 1e20
        processor = FeatureProcessor(missing_indicators=True, imputer=SpyImputer()).fit(train)
        output = processor.transform(train)
        self.assertEqual(output[0, 206 + 3], 0.0)
        self.assertEqual(output[0, 3], 0.0)
        self.assertEqual(processor.scaler.mean_[3], np.mean(train["X"][1:, 3]))

    def test_processor_round_trip(self):
        train = synthetic()
        train["clip"] = np.array(["vowel", "phrase"] * 4)
        for kwargs in ({}, {"task_aware": True, "phrase_untrusted_columns": [1, 4], "missing_indicators": True}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                processor = FeatureProcessor(**kwargs).fit(train)
                processor.save(directory)
                loaded = FeatureProcessor.load(directory)
                np.testing.assert_array_equal(loaded.transform(train), processor.transform(train))
                self.assertEqual(loaded.metadata(), processor.metadata())

    def test_imputer_round_trip_requires_external_fitted_estimator(self):
        train = synthetic()
        train["X"][0, 3] = np.nan
        imputer = SpyImputer()
        processor = FeatureProcessor(imputer=imputer, missing_indicators=True).fit(train)
        with tempfile.TemporaryDirectory() as directory:
            processor.save(directory)
            with self.assertRaisesRegex(ValueError, "already-fitted imputer"):
                FeatureProcessor.load(directory)
            loaded = FeatureProcessor.load(directory, imputer=imputer)
            np.testing.assert_array_equal(loaded.transform(train), processor.transform(train))


class DataContractTests(unittest.TestCase):
    def test_layout_declaration_is_optional_only_when_not_required(self):
        data = synthetic()
        self.assertFalse(D.validate_feature_layout(data))
        with self.assertRaisesRegex(ValueError, "feature_layout metadata is required"):
            D.validate_feature_layout(data, required=True)
        data["feature_layout"] = np.full(len(data["X"]), D.FEATURE_LAYOUT)
        self.assertTrue(D.validate_feature_layout(data, required=True))

    def test_declared_reversed_or_mixed_layout_is_rejected(self):
        data = synthetic(2)
        reversed_layout = "embedding_mean192_embedding_std192_acoustic14"
        for declarations in ([reversed_layout] * 2, [D.FEATURE_LAYOUT, reversed_layout]):
            data["feature_layout"] = np.asarray(declarations)
            with self.subTest(declarations=declarations):
                with self.assertRaisesRegex(ValueError, "Refusing to infer or reorder"):
                    D.validate_feature_layout(data)
                with self.assertRaisesRegex(ValueError, "feature_layout"):
                    FeatureProcessor().fit(data)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "synthetic.npz"
                    np.savez(path, **data)
                    with self.assertRaisesRegex(ValueError, "feature_layout"):
                        D.load_npz(str(path))

    def test_layout_declaration_requires_one_label_per_row(self):
        data = synthetic(2)
        for declarations in (np.array(D.FEATURE_LAYOUT), np.full((2, 1), D.FEATURE_LAYOUT),
                             np.full(1, D.FEATURE_LAYOUT)):
            data["feature_layout"] = declarations
            with self.subTest(shape=declarations.shape), self.assertRaisesRegex(ValueError, "must have shape"):
                D.validate_feature_layout(data)

    def test_valid_layout_provenance_survives_loader_and_concat(self):
        data = synthetic(3)
        data["feature_layout"] = np.full(3, D.FEATURE_LAYOUT)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.npz"
            np.savez(path, **data)
            loaded = D.load_npz(str(path))
        combined = D.concat([loaded, loaded])
        self.assertTrue(D.validate_feature_layout(combined, required=True))
        np.testing.assert_array_equal(combined["feature_layout"], np.full(6, D.FEATURE_LAYOUT))

    def test_optional_arrays_survive_loading_and_concat(self):
        a, b = synthetic(2), synthetic(3, 4)
        a["clip_type"] = np.array(["phrase", "vowel"])
        a["acoustic_available"] = np.ones((2, 14), dtype=bool)
        a["acoustic_available"][0, 4] = False
        a["recording_id"] = np.array(["a", "b"])
        b["recording_id"] = np.array(["c", "d", "e"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.npz"
            np.savez(path, **a)
            loaded = D.load_npz(str(path))
            self.assertIn("clip", loaded)
            self.assertNotIn("clip_type", loaded)
            combined = D.concat([loaded, b])
        np.testing.assert_array_equal(combined["clip"], ["phrase", "vowel", "unknown", "unknown", "unknown"])
        np.testing.assert_array_equal(combined["recording_id"], ["a", "b", "c", "d", "e"])
        self.assertFalse(combined["acoustic_available"][0, 4])
        self.assertTrue(combined["acoustic_available"][2:].all())

    def test_conflicting_or_invalid_tasks_are_rejected(self):
        data = synthetic(2)
        data["clip"] = np.array(["phrase", "vowel"])
        data["clip_type"] = np.array(["vowel", "vowel"])
        with self.assertRaisesRegex(ValueError, "disagree"):
            D.clip_labels(data)
        data.pop("clip_type")
        data["clip"][0] = "other"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            D.clip_labels(data)

    def test_nan_loader_opt_in_is_acoustic_only(self):
        data = synthetic()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.npz"
            data["X"][0, 1] = np.nan
            np.savez(path, **data)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                D.load_npz(str(path))
            self.assertTrue(np.isnan(D.load_npz(str(path), allow_acoustic_nan=True)["X"][0, 1]))
            for column, value in ((14, np.nan), (1, np.inf), (397, -np.inf)):
                data = synthetic()
                data["X"][0, column] = value
                np.savez(path, **data)
                with self.subTest(column=column, value=value), self.assertRaisesRegex(ValueError, "forbidden"):
                    D.load_npz(str(path), allow_acoustic_nan=True)


if __name__ == "__main__":
    unittest.main()
