"""The layout gate: what every NPZ read must prove before anything slices it.

Synthetic-only. No corpus, no locked speaker list and no file from the
workspace is opened here; every fixture is built in memory or in a temporary
directory.

Feature selection is positional -- F2 is `X[:, :206]` -- and the only thing
any other gate checks is the 398-column width. So a file in the legacy
`[emb-mean | emb-std | acoustic]` producer order trains and evaluates without
a single error, on 192 embedding-mean columns plus the first 14 embedding-std
columns, with every acoustic feature dropped. These tests pin the two halves
of the contract that make that impossible:

  declaration   a per-row `feature_layout` naming the canonical order
  proof         structural evidence from the data itself, at full score

Neither alone is sufficient. A file with no declaration has undeclared
provenance; a file whose declaration contradicts the evidence is worse, since
the claim would otherwise be taken at face value. Both are refused, with no
bypass flag anywhere.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import data as D
import evaluate as E
import features as F
import permute_locked_test_layout as L
import pipeline as P
import verify_layout as V
from make_smoke_data import to_canonical_layout


def canonical_rows(n=40, seed=3, prefix="synthetic", dtype=np.float64):
    """A structurally honest canonical fixture: the three blocks on their scales."""
    rng = np.random.default_rng(seed)
    X = to_canonical_layout(rng.normal(size=(n, D.N_RAW_FEATURES))).astype(dtype)
    return {
        "X": X,
        "y_severity": np.linspace(5.0, 95.0, n),
        "severity_kind": np.full(n, "cape_v"),
        "y_binary": (np.arange(n) % 2).astype(np.int64),
        "speaker_id": np.array([f"{prefix}-{i:04d}" for i in range(n)]),
        "corpus": np.full(n, "pvqd"),
        "feature_layout": np.full(n, D.FEATURE_LAYOUT),
    }


def to_legacy_order(X):
    """Rewrite a canonical matrix in the legacy producer order."""
    return np.concatenate([X[:, D.N_ACOUSTIC:], X[:, :D.N_ACOUSTIC]], axis=1)


def write_npz(path, rows, drop=()):
    np.savez(path, **{k: v for k, v in rows.items() if k not in drop})
    return path


class EvidenceTests(unittest.TestCase):
    def test_canonical_fixture_proves_itself_with_the_full_score(self):
        rows = canonical_rows()
        evidence = V.assert_canonical_input(rows, label="fixture")
        self.assertEqual(evidence["layout"], V.ACOUSTIC_FIRST)
        self.assertEqual(evidence["score"], V.MAX_LAYOUT_SCORE)
        self.assertTrue(evidence["checks"]["std_block_nonnegative"])
        self.assertTrue(evidence["checks"]["mean_block_straddles_zero"])
        self.assertTrue(evidence["checks"]["acoustic_block_leaves_unit_box"])

    def test_legacy_order_is_named_in_the_refusal(self):
        rows = canonical_rows()
        rows["X"] = to_legacy_order(rows["X"])
        with self.assertRaises(V.LayoutError) as caught:
            V.assert_canonical_input(rows, label="legacy fixture")
        message = str(caught.exception)
        self.assertIn(V.EMBEDDING_FIRST, message)
        self.assertIn("legacy fixture", message)
        # The refusal has to say what to do about it, not only that it failed.
        self.assertIn("permute_locked_test_layout.py", message)

    def test_a_declaration_the_evidence_contradicts_is_refused(self):
        """The worst case: legacy columns carrying a canonical declaration."""
        rows = canonical_rows()
        rows["X"] = to_legacy_order(rows["X"])
        self.assertTrue(D.validate_feature_layout(rows, required=True))
        with self.assertRaises(V.LayoutError):
            V.assert_canonical_input(rows)

    def test_a_declaration_without_evidence_is_refused(self):
        """A Gaussian matrix leaves the candidates tied and proves nothing."""
        rng = np.random.default_rng(11)
        rows = canonical_rows()
        rows["X"] = rng.normal(size=rows["X"].shape)
        with self.assertRaisesRegex(V.LayoutError, "does not single out"):
            V.assert_canonical_input(rows)

    def test_a_missing_declaration_is_refused_even_when_provable(self):
        rows = canonical_rows()
        rows.pop("feature_layout")
        with self.assertRaisesRegex(ValueError, "feature_layout"):
            V.assert_canonical_input(rows)
        # ... and the evidence alone would have been conclusive.
        self.assertEqual(V.assert_layout(rows["X"])["score"], V.MAX_LAYOUT_SCORE)

    def test_a_partial_score_is_not_a_proof(self):
        rows = canonical_rows()
        # One negative entry in the std block: no block of 192 per-dimension
        # standard deviations contains one, so this is no longer that block.
        rows["X"][0, D.N_ACOUSTIC + D.N_EMB] = -0.5
        with self.assertRaisesRegex(V.LayoutError, "incomplete"):
            V.assert_canonical_input(rows)

    def test_a_wrong_width_cannot_be_verified_at_all(self):
        rows = canonical_rows()
        rows["X"] = rows["X"][:, :206]
        with self.assertRaisesRegex(V.LayoutError, "398"):
            V.assert_layout(rows["X"])


class PipelineGateTests(unittest.TestCase):
    def test_legacy_training_input_fails_before_preprocessing_is_fitted(self):
        rows = canonical_rows(n=60, prefix="train")
        rows["X"] = to_legacy_order(rows["X"])
        with tempfile.TemporaryDirectory(prefix="vocal-layout-gate-") as directory:
            root = Path(directory)
            path = write_npz(root / "legacy_train.npz", rows)
            lock = root / "synthetic_lock.txt"
            lock.write_text("synthetic-not-in-input\n")
            args = ["pipeline.py", "--data", str(path), "--lock-test", str(lock),
                    "--models", "ridge", "--feature-set", "F2", "--out", str(root / "bundle")]
            with mock.patch.object(sys, "argv", args), mock.patch.object(
                    P.F.FeatureProcessor, "fit",
                    side_effect=AssertionError("preprocessing must not fit")) as fit:
                with self.assertRaises(V.LayoutError):
                    P.main()
                fit.assert_not_called()
            self.assertFalse((root / "bundle").exists())

    def test_the_gate_is_not_conditional_on_phase_2_options(self):
        """The Phase 1 default path slices F2 by position too."""
        rows = canonical_rows(n=60, prefix="train")
        rows.pop("feature_layout")
        with tempfile.TemporaryDirectory(prefix="vocal-layout-gate-") as directory:
            root = Path(directory)
            path = write_npz(root / "undeclared_train.npz", rows)
            lock = root / "synthetic_lock.txt"
            lock.write_text("synthetic-not-in-input\n")
            args = ["pipeline.py", "--data", str(path), "--lock-test", str(lock),
                    "--models", "ridge", "--out", str(root / "bundle")]
            with mock.patch.object(sys, "argv", args), mock.patch.object(
                    P.F.FeatureProcessor, "fit",
                    side_effect=AssertionError("preprocessing must not fit")) as fit:
                with self.assertRaisesRegex(ValueError, "feature_layout"):
                    P.main()
                fit.assert_not_called()

    def test_a_gated_run_records_the_attestation_evaluation_requires(self):
        self.assertEqual(V.LAYOUT_PROVENANCE, "declared_in_npz_and_structurally_verified")
        source = Path(P.__file__).read_text()
        self.assertIn('"input_layout_provenance": V.LAYOUT_PROVENANCE', source)


class EvaluateGateTests(unittest.TestCase):
    """evaluate.py spends the one-shot locked test set, so both sides are checked."""

    def _bundle(self, root, speakers, provenance=V.LAYOUT_PROVENANCE):
        lock = root / "synthetic_lock.txt"
        lock.write_text("\n".join(speakers) + "\n")
        manifest = {"schema_version": 2, "feature_set": "F2",
                    "models": ["ridge"], "deployable_models": ["ridge"],
                    "lock_test_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
                    "artifact_sha256": {}}
        if provenance is not None:
            manifest["input_layout_provenance"] = provenance
        (root / "manifest.json").write_text(json.dumps(manifest))
        return lock, manifest

    def test_a_pre_gate_bundle_is_refused_before_evaluation_data_is_opened(self):
        """A bundle fitted before the gate cannot acquire the attestation later."""
        for provenance in (None, "declared_in_npz", "legacy_phase1_assumption_unverified"):
            with self.subTest(provenance=provenance), tempfile.TemporaryDirectory(
                    prefix="vocal-layout-gate-") as directory:
                root = Path(directory)
                lock, _ = self._bundle(root, ["synthetic-0000"], provenance)
                args = ["evaluate.py", "--bundle", str(root), "--model", "ridge",
                        "--lock-test", str(lock), "--test-npz", str(root / "never-open.npz"),
                        "--report", str(root / "never-written.json")]
                with mock.patch.object(sys, "argv", args), \
                        mock.patch.object(E, "verify_bundle_artifacts"), \
                        mock.patch.object(E.D, "load_npz",
                                          side_effect=AssertionError("must not open")) as load:
                    with self.assertRaisesRegex(SystemExit, "input_layout_provenance"):
                        E.main()
                    load.assert_not_called()
                self.assertFalse((root / "never-written.json").exists())

    def test_legacy_test_input_fails_before_any_feature_is_selected(self):
        rows = canonical_rows(n=58, prefix="locked")
        rows["X"] = to_legacy_order(rows["X"])
        speakers = list(map(str, rows["speaker_id"]))
        with tempfile.TemporaryDirectory(prefix="vocal-layout-gate-") as directory:
            root = Path(directory)
            lock, _ = self._bundle(root, speakers)
            args = ["evaluate.py", "--bundle", str(root), "--model", "ridge",
                    "--lock-test", str(lock), "--test-npz", "synthetic_fixture",
                    "--report", str(root / "never-written.json")]
            with mock.patch.object(sys, "argv", args), \
                    mock.patch.object(E, "verify_bundle_artifacts"), \
                    mock.patch.object(E, "load_calibration", return_value=(None, {})), \
                    mock.patch.object(E.D, "load_npz", return_value=rows), \
                    mock.patch.object(E.F, "select_features",
                                      side_effect=AssertionError("must not select")) as select:
                with self.assertRaises(V.LayoutError):
                    E.main()
                select.assert_not_called()
            self.assertFalse((root / "never-written.json").exists())


class LockedTestPermutationTests(unittest.TestCase):
    """The one-time authorized permutation of the locked test file."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="vocal-locked-permute-")
        self.root = Path(self.directory.name)
        self.rows = canonical_rows(n=58, prefix="LOCKED", dtype=np.float32)
        self.canonical_X = self.rows["X"].copy()
        self.rows["X"] = to_legacy_order(self.rows["X"])
        self.rows.pop("feature_layout")
        self.speakers = list(map(str, self.rows["speaker_id"]))
        self.source = write_npz(self.root / "locked_test.npz", self.rows)
        self.lock = self.root / "lock.txt"
        self.lock.write_text("\n".join(self.speakers) + "\n")
        self.source_sha = L._sha256(self.source)

    def tearDown(self):
        self.directory.cleanup()

    def convert(self, out="canonical.npz", **kwargs):
        options = {"source": self.source, "destination": self.root / out,
                   "lock_path": self.lock, "source_layout": V.EMBEDDING_FIRST,
                   "approved_by": "Synthetic Owner", "approval_date": "2026-10-11"}
        options.update(kwargs)
        return L.permute_locked_test(**options)

    def test_the_conversion_is_verified_and_the_source_is_untouched(self):
        provenance = self.convert()
        destination = self.root / "canonical.npz"
        self.assertEqual(L._sha256(self.source), self.source_sha)
        self.assertTrue(provenance["source_unmodified"])
        self.assertEqual(provenance["source_sha256_after_conversion"], self.source_sha)
        self.assertEqual(provenance["verification"]["output_layout_proof"]["score"],
                         V.MAX_LAYOUT_SCORE)
        for block in provenance["verification"]["blocks"].values():
            self.assertTrue(block["elementwise_identical"])
        with np.load(destination, allow_pickle=False) as got:
            # A pure permutation: same dtype, and each block bit-identical to
            # the columns it came from.
            self.assertEqual(got["X"].dtype, self.rows["X"].dtype)
            np.testing.assert_array_equal(got["X"][:, :14], self.rows["X"][:, 384:])
            np.testing.assert_array_equal(got["X"][:, 14:206], self.rows["X"][:, :192])
            np.testing.assert_array_equal(got["X"][:, 206:], self.rows["X"][:, 192:384])
            np.testing.assert_array_equal(got["y_severity"], self.rows["y_severity"])
            np.testing.assert_array_equal(got["speaker_id"], self.rows["speaker_id"])
            self.assertEqual(set(np.asarray(got["feature_layout"], dtype=str)),
                             {D.FEATURE_LAYOUT})
        # The product of the whole exercise: the copy passes the gate that the
        # original cannot, and the evaluation path can finally read it.
        V.assert_canonical_input(D.load_npz(str(destination)), label=str(destination))

    def test_the_provenance_records_that_no_label_was_read(self):
        provenance = self.convert()
        self.assertFalse(provenance["test_labels_read"])
        self.assertTrue(provenance["test_feature_columns_read"])
        self.assertFalse(provenance["preprocessing_fit"])
        self.assertFalse(provenance["model_or_hyperparameter_decision"])
        self.assertEqual(provenance["authorization"]["approved_by"], "Synthetic Owner")
        self.assertEqual(provenance["dtype"], "float32")
        self.assertEqual(provenance["columns"], [[384, 398], [0, 192], [192, 384]])

    def test_a_file_that_is_not_the_locked_test_file_is_refused(self):
        """Mirror image of prepare_v2_data.py: this tool takes only the locked set."""
        partial = self.root / "partial_lock.txt"
        partial.write_text("\n".join(self.speakers[:30]) + "\n")
        with self.assertRaisesRegex(ValueError, "locked test file and nothing else"):
            self.convert(out="never.npz", lock_path=partial)
        extra = self.root / "extra_lock.txt"
        extra.write_text("\n".join(self.speakers + ["someone-else"]) + "\n")
        with self.assertRaisesRegex(ValueError, "locked test file and nothing else"):
            self.convert(out="never2.npz", lock_path=extra)
        self.assertFalse((self.root / "never.npz").exists())
        self.assertFalse((self.root / "never2.npz").exists())

    def test_an_unprovable_or_already_canonical_source_is_refused(self):
        rng = np.random.default_rng(4)
        gaussian = dict(self.rows, X=rng.normal(size=self.rows["X"].shape).astype(np.float32))
        unprovable = write_npz(self.root / "unprovable.npz", gaussian)
        with self.assertRaisesRegex(V.LayoutError, "does not single out"):
            self.convert(out="never3.npz", source=unprovable)
        already = write_npz(self.root / "already_canonical.npz",
                            dict(self.rows, X=self.canonical_X))
        with self.assertRaisesRegex(V.LayoutError, "not the expected"):
            self.convert(out="never4.npz", source=already)
        self.assertFalse((self.root / "never3.npz").exists())
        self.assertFalse((self.root / "never4.npz").exists())

    def test_an_existing_output_is_never_overwritten(self):
        self.convert()
        with self.assertRaises(FileExistsError):
            self.convert()
        # A dangling symlink is an existing path too, and must not be followed.
        link = self.root / "dangling.npz"
        link.symlink_to(self.root / "missing.npz")
        with self.assertRaises(FileExistsError):
            self.convert(out="dangling.npz")

    def test_the_authorization_flag_is_required_on_the_command_line(self):
        args = ["permute_locked_test_layout.py", "--data", str(self.source),
                "--source-layout", V.EMBEDDING_FIRST, "--lock-test", str(self.lock),
                "--out", str(self.root / "never5.npz"),
                "--approved-by", "Synthetic Owner", "--approval-date", "2026-10-11"]
        with mock.patch.object(sys, "argv", args), \
                mock.patch.object(sys, "stderr", io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as caught:
                L.main()
        self.assertNotEqual(caught.exception.code, 0)
        self.assertIn("--authorize-locked-test-permutation", stderr.getvalue())
        self.assertFalse((self.root / "never5.npz").exists())


if __name__ == "__main__":
    unittest.main()
