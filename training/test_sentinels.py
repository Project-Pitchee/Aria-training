"""Synthetic-only regression checks for the confirmed core acoustic sentinels.

These tests generate their own rows and artifacts; no corpus or locked-test
files are opened.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    from . import data as D
    from .features import AcousticMedianImputer, FeatureProcessor
except ImportError:
    import data as D
    from features import AcousticMedianImputer, FeatureProcessor


def synthetic(n=8, seed=73):
    """Canonical raw layout with independently valid acoustic observations."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, D.N_RAW_FEATURES))
    X[:, :14] = rng.uniform(0.1, 1.0, size=(n, 14))
    X[:, 0] = 3.0 + np.arange(n)
    X[:, 8] = 100.0 + 10.0 * np.arange(n)
    X[:, 9] = 10.0 + np.arange(n)
    X[:, 10] = 0.05 + 0.01 * np.arange(n)
    X[:, 12] = 60.0 + np.arange(n)
    return {
        "X": X,
        "feature_layout": np.full(n, D.FEATURE_LAYOUT),
        "clip": np.full(n, "vowel", dtype="<U7"),
        "y_severity": rng.uniform(0, 100, n),
        "severity_kind": np.full(n, "cape_v"),
        "y_binary": np.arange(n) % 2,
        "speaker_id": np.array([f"synthetic-sentinel-{i}" for i in range(n)]),
        "corpus": np.full(n, "synthetic"),
    }


def sentinel_rows():
    data = synthetic()
    data["X"][:, 0] = [-200.0, -200.0, -5.0, 0.0, 7.0, 20.0, 11.0, 5.0]
    data["X"][2:4, 8:11] = 0.0
    # A voiced clip can have a constant pitch; these zeros are valid.
    data["X"][4, 9:11] = 0.0
    # Zero perturbation and voiced_fraction values are not core_v1 sentinels.
    data["X"][5, 2:8] = 0.0
    data["X"][5, 11] = 0.0
    return data


def expected_availability(n=8):
    flags = np.ones((n, D.N_ACOUSTIC), dtype=bool)
    flags[:2, 0] = False
    flags[2:4, 8:11] = False
    return flags


def processor(**kwargs):
    options = {
        "feature_set": "F2",
        "sentinel_policy": "core_v1",
        "imputation_strategy": "median",
        "missing_indicators": True,
    }
    options.update(kwargs)
    return FeatureProcessor(**options)


class ExternalMeanImputer:
    """Test-only estimator for the caller-owned imputer compatibility path."""

    def fit(self, X):
        self.fit_input = X.copy()
        self.statistics_ = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        return np.where(np.isnan(X), self.statistics_, X)


class AcousticSentinelTests(unittest.TestCase):
    def test_exact_sentinels_preserve_valid_negative_and_zero_measurements(self):
        train = sentinel_rows()
        original = {key: value.copy() for key, value in train.items()}
        fitted = processor().fit(train)
        output = fitted.transform(train)
        self.assertEqual(output.shape, (8, 220))
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_array_equal(output[:, 206:], expected_availability())
        np.testing.assert_array_equal(np.unique(output[:, 206:]), [0.0, 1.0])
        # Flags must describe the raw measurement before the finite imputation.
        self.assertEqual(output[0, 206], 0.0)
        self.assertEqual(output[2, 206], 1.0)  # HNR -5 is observed.
        self.assertEqual(output[3, 206], 1.0)  # HNR 0 is observed.
        np.testing.assert_array_equal(output[4, 214:217], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(output[5, 208:214], np.ones(6))
        self.assertEqual(output[5, 217], 1.0)
        for key, value in original.items():
            np.testing.assert_array_equal(train[key], value, err_msg=key)

    def test_f0_median_marker_masks_its_entire_dependent_group(self):
        train = sentinel_rows()
        # The group rule depends on median == 0, not an all-zero row heuristic.
        train["X"][2, 9:11] = [77.0, 3.0]
        output = processor().fit(train).transform(train)
        np.testing.assert_array_equal(output[:, 206:], expected_availability())

    def test_nan_f0_median_does_not_invent_a_zero_sentinel_for_other_columns(self):
        train = synthetic()
        train["X"][0, 8:11] = [np.nan, 0.0, 0.0]
        output = processor().fit(train).transform(train)
        flags = np.ones((8, 14))
        flags[0, 8] = 0.0
        np.testing.assert_array_equal(output[:, 206:], flags)

    def test_scaler_and_medians_exclude_sentinels_without_including_imputed_rows(self):
        train = sentinel_rows()
        fitted = processor().fit(train)
        available = expected_availability()
        for column in (0, 8, 9, 10):
            trusted_values = train["X"][available[:, column], column]
            with self.subTest(column=column):
                self.assertAlmostEqual(fitted.scaler.mean_[column], np.mean(trusted_values))
                self.assertAlmostEqual(fitted.scaler.var_[column], np.var(trusted_values))
                self.assertAlmostEqual(fitted.imputer.statistics_[column], np.median(trusted_values))
                self.assertEqual(fitted.scaler.n_samples_seen_[column], len(trusted_values))
        restored = fitted.scaler.inverse_transform(fitted.transform(train)[:, :206])
        for column in (0, 8, 9, 10):
            median = np.median(train["X"][available[:, column], column])
            np.testing.assert_allclose(restored[~available[:, column], column], median)
        np.testing.assert_allclose(restored[:, :14][available], train["X"][:, :14][available], atol=1e-14)

    def test_nan_and_explicit_availability_intersect_with_sentinel_rules(self):
        train = sentinel_rows()
        train["X"][6, 4] = np.nan
        train["X"][7, 6] = 1e20
        train["acoustic_available"] = np.ones((8, 14), dtype=bool)
        train["acoustic_available"][7, 6] = False
        original_X = train["X"].copy()
        original_mask = train["acoustic_available"].copy()
        fitted = processor().fit(train)
        output = fitted.transform(train)
        flags = expected_availability()
        flags[6, 4] = flags[7, 6] = False
        np.testing.assert_array_equal(output[:, 206:], flags)
        np.testing.assert_array_equal(train["X"], original_X)
        np.testing.assert_array_equal(train["acoustic_available"], original_mask)
        self.assertAlmostEqual(fitted.scaler.mean_[6], train["X"][:7, 6].mean())
        self.assertAlmostEqual(fitted.imputer.statistics_[6], np.median(train["X"][:7, 6]))

    def test_medians_use_only_task_trusted_training_values(self):
        train = synthetic(6)
        train["X"][:, 0] = [-200.0, 1.0, 3.0, 100.0, 5.0, 900.0]
        train["clip"][-1] = "phrase"
        train["acoustic_available"] = np.ones((6, 14), dtype=bool)
        train["acoustic_available"][3, 0] = False
        fitted = processor(task_aware=True, phrase_untrusted_columns=[0]).fit(train)
        self.assertEqual(fitted.scaler.mean_[0], 3.0)
        self.assertAlmostEqual(fitted.scaler.var_[0], np.var([1.0, 3.0, 5.0]))
        self.assertEqual(fitted.imputer.statistics_[0], 3.0)
        output = fitted.transform(train)
        np.testing.assert_array_equal(output[:, 206], [0, 1, 1, 0, 1, 0])
        self.assertEqual(output[-1, 0], 0.0)

    def test_dev_transform_never_refits_or_changes_training_statistics(self):
        train = sentinel_rows()
        fitted = processor().fit(train)
        original_stats = fitted.imputer.statistics_.copy()
        original_scaler = {name: getattr(fitted.scaler, name).copy()
                           for name in ("mean_", "var_", "scale_", "n_samples_seen_")}
        original_metadata = fitted.metadata()
        dev = synthetic(3, seed=900)
        dev["X"][:, :14] = 1e12
        dev["X"][0, 0] = -200.0
        dev["X"][0, 8:11] = 0.0
        dev["X"][1, 4] = np.nan
        for _ in range(2):
            output = fitted.transform(dev)
            self.assertTrue(np.isfinite(output).all())
            np.testing.assert_array_equal(fitted.imputer.statistics_, original_stats)
            for name, value in original_scaler.items():
                np.testing.assert_array_equal(getattr(fitted.scaler, name), value)
        self.assertEqual(fitted.metadata(), original_metadata)
        restored = fitted.scaler.inverse_transform(output[:, :206])
        self.assertAlmostEqual(restored[0, 0], original_stats[0])
        self.assertAlmostEqual(restored[1, 4], original_stats[4])

    def test_sentinel_processing_is_opt_in_and_legacy_scaling_is_exact(self):
        train = sentinel_rows()
        train.pop("feature_layout")
        reference = StandardScaler().fit(train["X"][:, :206])
        fitted = FeatureProcessor().fit(train)
        np.testing.assert_array_equal(fitted.scaler.mean_, reference.mean_)
        np.testing.assert_array_equal(fitted.transform(train), reference.transform(train["X"][:, :206]))

    def test_core_policy_fails_without_imputation_even_before_missing_rows_arrive(self):
        for data in (synthetic(), sentinel_rows()):
            with self.subTest(has_sentinels=bool(np.any(data["X"][:, 0] == -200))):
                with self.assertRaisesRegex(ValueError, "imput"):
                    FeatureProcessor(sentinel_policy="core_v1").fit(data)

    def test_core_policy_requires_declared_canonical_layout_at_fit_and_transform(self):
        fitted = processor().fit(sentinel_rows())
        for declaration in (None, np.full(8, "embedding_mean192_embedding_std192_acoustic14"),
                            np.full((8, 1), D.FEATURE_LAYOUT), np.full(7, D.FEATURE_LAYOUT)):
            data = sentinel_rows()
            if declaration is None:
                data.pop("feature_layout")
            else:
                data["feature_layout"] = declaration
            with self.subTest(declaration=declaration), self.assertRaisesRegex(ValueError, "feature_layout"):
                processor().fit(data)
            with self.subTest(transform_declaration=declaration), self.assertRaisesRegex(ValueError, "feature_layout"):
                fitted.transform(data)

    def test_all_feature_sets_require_raw_398_columns_and_append_14_flags(self):
        train = sentinel_rows()
        for feature_set, width in (("F1", 398), ("F2", 206), ("F3", 14)):
            with self.subTest(feature_set=feature_set):
                fitted = processor(feature_set=feature_set).fit(train)
                self.assertEqual(fitted.transform(train).shape, (8, width + 14))
                for bad_width in (14, 206, 397, 399):
                    invalid = synthetic()
                    invalid["X"] = np.zeros((8, bad_width))
                    with self.subTest(bad_width=bad_width), self.assertRaisesRegex(ValueError, "398"):
                        fitted.transform(invalid)

    def test_no_observed_training_value_in_any_acoustic_column_fails(self):
        for kind in ("hnr", "f0", "nan", "mask"):
            train = synthetic()
            if kind == "hnr":
                train["X"][:, 0] = -200.0
            elif kind == "f0":
                train["X"][:, 8:11] = 0.0
            elif kind == "nan":
                train["X"][:, 2] = np.nan
            else:
                train["acoustic_available"] = np.ones((8, 14), dtype=bool)
                train["acoustic_available"][:, 4] = False
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "no trusted observed training values"):
                processor().fit(train)

    def test_infinity_or_nonfinite_embeddings_are_never_imputed(self):
        for column, value in ((0, np.inf), (8, -np.inf), (14, np.nan), (397, np.inf)):
            train = sentinel_rows()
            train["X"][0, column] = value
            with self.subTest(column=column, value=value), self.assertRaisesRegex(ValueError, "forbidden"):
                processor().fit(train)

    def test_saved_median_and_policy_round_trip_without_external_imputer(self):
        train, dev = sentinel_rows(), synthetic(3, 99)
        dev["X"][0, 0] = -200.0
        dev["X"][1, 8:11] = 0.0
        fitted = processor().fit(train)
        expected = fitted.transform(dev)
        with tempfile.TemporaryDirectory() as directory:
            metadata = fitted.save(directory)
            self.assertEqual(metadata["sentinel_policy"], "core_v1")
            self.assertEqual(metadata["imputation_strategy"], "median")
            self.assertFalse(metadata["imputer_required_at_load"])
            self.assertEqual(metadata["n_features_in"], 398)
            self.assertEqual(metadata["n_features_out"], 220)
            self.assertTrue(metadata["sentinel_rules"])
            self.assertEqual(metadata["train_sentinel_counts"]["hnr_minus_200"], 2)
            self.assertEqual(metadata["train_sentinel_counts"]["f0_median_zero"], 2)
            np.testing.assert_array_equal(metadata["train_sentinel_counts"]["by_acoustic_column"],
                                          (~expected_availability()).sum(axis=0))
            with (patch.object(FeatureProcessor, "fit", side_effect=AssertionError("must not fit on reload")),
                  patch.object(StandardScaler, "fit", side_effect=AssertionError("must not refit scaler")),
                  patch.object(AcousticMedianImputer, "fit", side_effect=AssertionError("must not refit median"))):
                loaded = FeatureProcessor.load(directory)
                self.assertEqual(loaded.metadata(), metadata)
                np.testing.assert_array_equal(loaded.transform(dev), expected)
                np.testing.assert_array_equal(loaded.imputer.statistics_, fitted.imputer.statistics_)
            with self.assertRaisesRegex(ValueError, "external imputer"):
                FeatureProcessor.load(directory, imputer=ExternalMeanImputer())

    def test_corrupt_saved_median_or_scaler_shape_and_nan_fail_closed(self):
        fitted = processor().fit(sentinel_rows())
        for field in ("imputer_statistics_", "mean_", "var_", "scale_"):
            for corruption in ("shape", "nan"):
                with self.subTest(field=field, corruption=corruption), tempfile.TemporaryDirectory() as directory:
                    fitted.save(directory)
                    path = Path(directory) / "feature_processor.npz"
                    with np.load(path, allow_pickle=False) as saved:
                        state = {name: saved[name].copy() for name in saved.files}
                    if corruption == "shape":
                        state[field] = state[field][:-1]
                    else:
                        state[field][0] = np.nan
                    np.savez(path, **state)
                    with self.assertRaises(ValueError):
                        FeatureProcessor.load(directory)

    def test_missing_saved_medians_fail_closed(self):
        fitted = processor().fit(sentinel_rows())
        with tempfile.TemporaryDirectory() as directory:
            fitted.save(directory)
            path = Path(directory) / "feature_processor.npz"
            with np.load(path, allow_pickle=False) as saved:
                state = {name: saved[name].copy() for name in saved.files if name != "imputer_statistics_"}
            np.savez(path, **state)
            with self.assertRaisesRegex(ValueError, "missing imputer_statistics_"):
                FeatureProcessor.load(directory)

    def test_saved_version_policy_and_imputer_config_must_agree(self):
        fitted = processor().fit(sentinel_rows())
        changes = (
            {"version": 999},
            {"version": 1},
            {"sentinel_policy": "unknown_policy"},
            {"sentinel_policy": "none"},
            {"sentinel_rules": {}},
            {"imputation_strategy": "mean"},
            {"imputation_strategy": None},
            {"imputer_required_at_load": True},
            {"imputer_class": "untrusted.CustomImputer"},
        )
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                fitted.save(directory)
                path = Path(directory) / "feature_processor.json"
                metadata = json.loads(path.read_text())
                metadata.update(change)
                path.write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):
                    FeatureProcessor.load(directory)

    def test_corrupt_saved_dimension_metadata_fails_closed(self):
        fitted = processor().fit(sentinel_rows())
        for field, value in (("n_features_in", 206), ("n_features_out", 206)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                fitted.save(directory)
                path = Path(directory) / "feature_processor.json"
                metadata = json.loads(path.read_text())
                metadata[field] = value
                path.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, "layout"):
                    FeatureProcessor.load(directory)

    def test_external_imputer_contract_is_preserved_for_sentinel_policy(self):
        train = sentinel_rows()
        imputer = ExternalMeanImputer()
        fitted = processor(imputation_strategy=None, imputer=imputer).fit(train)
        np.testing.assert_array_equal(np.isnan(imputer.fit_input), ~expected_availability())
        with tempfile.TemporaryDirectory() as directory:
            fitted.save(directory)
            with self.assertRaisesRegex(ValueError, "already-fitted imputer"):
                FeatureProcessor.load(directory)
            loaded = FeatureProcessor.load(directory, imputer=imputer)
            np.testing.assert_array_equal(loaded.transform(train), fitted.transform(train))
        with self.assertRaises(ValueError):
            processor(imputer=ExternalMeanImputer())


if __name__ == "__main__":
    unittest.main()
