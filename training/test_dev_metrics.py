"""Dev metric contract: absolute error is reported on the native severity scale.

The regression heads are fitted on the per-corpus standardised target, so a
prediction is in standard deviations until it is inverted. Spearman and Pearson
are scale-invariant and hide the difference; MAE does not. A MAE accidentally
computed on the harmonized prediction is a plausible-looking number in the
wrong unit, which is exactly the kind of value that reaches a paper.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from . import harmonize as H
    from .pipeline import dev_metrics_fn, native_error_fields, to_native_severity
except ImportError:
    import harmonize as H
    from pipeline import dev_metrics_fn, native_error_fields, to_native_severity


def rows(severity, corpus="pvqd", kind="cape_v"):
    severity = np.asarray(severity, dtype=float)
    return {
        "y_severity": severity,
        "corpus": np.array([corpus] * len(severity)),
        "severity_kind": np.array([kind] * len(severity)),
    }


class NativeInversionTests(unittest.TestCase):
    def test_inversion_round_trips_the_training_normalisation(self):
        data = rows([10.0, 30.0, 55.0, 80.0, 95.0, 20.0])
        norm = H.fit_severity_norm(
            {**data, "speaker_id": np.array(list("abcdef"))})
        harmonized = H.apply_severity_norm(data, norm)
        recovered = to_native_severity(harmonized, data, norm)
        np.testing.assert_allclose(recovered, data["y_severity"], atol=1e-9)

    def test_each_row_is_inverted_with_its_own_corpus_constants(self):
        data = {
            "y_severity": np.array([10.0, 90.0, 1.0, 3.0]),
            "corpus": np.array(["pvqd", "pvqd", "svd", "svd"]),
            "severity_kind": np.array(["cape_v", "cape_v", "grbas", "grbas"]),
        }
        norm = {"pvqd": (50.0, 20.0), "svd": (2.0, 0.5)}
        native = to_native_severity(np.array([0.0, 1.0, 0.0, -1.0]), data, norm)
        # A corpus-blind inversion would put the SVD rows on the PVQD scale.
        np.testing.assert_allclose(native, [50.0, 70.0, 2.0, 1.5], atol=1e-9)

    def test_unknown_corpus_is_refused_rather_than_left_as_nan(self):
        data = rows([10.0, 20.0], corpus="unknown_corpus")
        with self.assertRaises(ValueError):
            to_native_severity(np.zeros(2), data, {"pvqd": (50.0, 20.0)})


class AbsoluteErrorTests(unittest.TestCase):
    def test_error_is_measured_in_native_units_not_standard_deviations(self):
        data = rows([40.0, 60.0])
        norm = {"pvqd": (50.0, 20.0)}
        # Predict one standard deviation above each true value.
        harmonized_truth = (data["y_severity"] - 50.0) / 20.0
        native = to_native_severity(harmonized_truth + 1.0, data, norm)
        fields = native_error_fields(data["y_severity"], native,
                                     H.native_units_label(data))
        # One sd is 20 native units; a harmonized-space MAE would report 1.0.
        self.assertAlmostEqual(fields["mae_native"], 20.0, places=9)
        self.assertEqual(fields["native_units"], "cape_v_mm_0_100")

    def test_exact_prediction_has_zero_error(self):
        data = rows([12.5, 48.0, 91.0])
        fields = native_error_fields(data["y_severity"], data["y_severity"].copy())
        self.assertEqual(fields["mae_native"], 0.0)
        self.assertEqual(fields["max_abs_error_native"], 0.0)

    def test_median_and_max_separate_a_single_large_error(self):
        truth = np.array([10.0, 20.0, 30.0, 40.0])
        prediction = np.array([11.0, 21.0, 31.0, 100.0])
        fields = native_error_fields(truth, prediction)
        self.assertAlmostEqual(fields["mae_native"], (1 + 1 + 1 + 60) / 4)
        self.assertAlmostEqual(fields["median_abs_error_native"], 1.0)
        self.assertAlmostEqual(fields["max_abs_error_native"], 60.0)

    def test_head_without_a_severity_scale_reports_no_number(self):
        fields = native_error_fields(np.array([10.0, 20.0]), None)
        self.assertIsNone(fields["mae_native"])
        self.assertIn("not defined", fields["mae_note"])
        self.assertNotIn("median_abs_error_native", fields)

    def test_non_finite_prediction_fails_instead_of_producing_nan_mae(self):
        with self.assertRaises(ValueError):
            native_error_fields(np.array([10.0, 20.0]), np.array([10.0, np.nan]))


class DevMetricsTests(unittest.TestCase):
    def test_adding_absolute_error_does_not_disturb_the_rank_metrics(self):
        truth = np.array([5.0, 25.0, 45.0, 65.0, 85.0])
        harmonized = np.array([-1.2, -0.5, 0.1, 0.7, 1.4])
        native = to_native_severity(harmonized, rows(truth), {"pvqd": (45.0, 28.0)})
        without = dev_metrics_fn(truth, harmonized)
        with_error = dev_metrics_fn(truth, harmonized, native_pred=native,
                                    native_units="cape_v_mm_0_100")
        for key in ("spearman_r", "spearman_p", "pearson_r", "pearson_p", "n"):
            self.assertEqual(without[key], with_error[key], key)
        self.assertIsNone(without["mae_native"])
        self.assertIsNotNone(with_error["mae_native"])

    def test_selection_metric_is_unchanged_by_the_error_block(self):
        # Dev selection reads spearman_r; the error block must not shadow it.
        truth = np.array([10.0, 20.0, 30.0, 40.0])
        metrics = dev_metrics_fn(truth, np.array([0.1, 0.2, 0.3, 0.4]),
                                 native_pred=np.array([99.0, 99.0, 99.0, 99.0]))
        self.assertAlmostEqual(metrics["spearman_r"], 1.0)
        self.assertGreater(metrics["mae_native"], 50.0)


if __name__ == "__main__":
    unittest.main()
