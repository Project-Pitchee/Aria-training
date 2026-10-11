"""Synthetic-only tests for the explicit, training-only packed-NPZ migration.

Run with ``python -m unittest discover -s training -p 'test_prepare_v2_data.py'``.
Every archive and lock list opened by these tests is created in a temporary
directory. No workspace data, evaluation archive, or real lock list is read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from numpy.testing import assert_array_equal

try:
    from . import data as D, prepare_v2_data as P
except ImportError:
    import data as D
    import prepare_v2_data as P


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GuardedArchive:
    """An NpzFile-like mapping that fails if a rejected input is materialized."""

    def __init__(self, arrays: dict, allowed: set[str]):
        self.arrays = arrays
        self.files = list(arrays)
        self.allowed = allowed
        self.reads = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __contains__(self, key):
        return key in self.arrays

    def __getitem__(self, key):
        self.reads.append(key)
        if key not in self.allowed:
            raise AssertionError(f"Rejected input must not read {key!r}")
        return self.arrays[key]

    def keys(self):
        return self.arrays.keys()

    def close(self):
        pass


class TrainingDataMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vocal-migration-synthetic-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "synthetic_train.npz"
        self.destination = self.root / "synthetic_migrated.npz"
        self.report = self.destination.with_suffix(".provenance.json")
        self.lock = self.root / "synthetic_lock.txt"
        self.lock.write_text("synthetic-locked-only\n", encoding="utf-8")

    def arrays(self, *, canonical=False):
        # Each value identifies its row, block, and position within the block.
        offsets = np.arange(4, dtype=np.float32)[:, None] * 10_000
        means = offsets + 1000 + np.arange(192, dtype=np.float32)
        stds = offsets + 2000 + np.arange(192, dtype=np.float32)
        acoustics = offsets + 3000 + np.arange(14, dtype=np.float32)
        blocks = (acoustics, means, stds) if canonical else (means, stds, acoustics)
        return {
            "X": np.concatenate(blocks, axis=1),
            "y_severity": np.array([11.5, 29.25, 11.5, 29.25], dtype=np.float32),
            "severity_kind": np.full(4, "cape_v"),
            "y_binary": np.array([0, 1, 0, 1], dtype=np.int64),
            "speaker_id": np.array(["synthetic-a", "synthetic-b", "synthetic-a", "synthetic-b"]),
            "corpus": np.full(4, "synthetic"),
            "clip": np.array(["vowel", "phrase", "vowel", "phrase"]),
            "recording_id": np.array(["synthetic-recording-a", "synthetic-recording-b"] * 2),
            "is_augmented": np.array([False, False, True, True]),
            "acoustic_available": np.tile(np.arange(14) % 3 != 0, (4, 1)),
            "split": np.full(4, "train"),
        }

    def write_source(self, arrays):
        np.savez_compressed(self.source, **arrays)

    def migrate(self, *, layout=None, report_path=None):
        return P.prepare_training_data(
            source=self.source,
            destination=self.destination,
            lock_path=self.lock,
            source_layout=P.SOURCE_ECAPA_FIRST if layout is None else layout,
            report_path=report_path,
        )

    def assert_no_result(self):
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.report.exists())

    def test_explicit_legacy_rotation_preserves_row_data_and_source_bytes(self):
        original = self.arrays()
        self.write_source(original)
        source_bytes = self.source.read_bytes()
        lock_bytes = self.lock.read_bytes()

        self.migrate()

        with np.load(self.destination, allow_pickle=False) as migrated:
            self.assertEqual(set(migrated.files), set(original) | {"feature_layout"})
            expected = self.arrays(canonical=True)["X"]
            assert_array_equal(migrated["X"], expected)
            self.assertEqual(migrated["X"].dtype, original["X"].dtype)
            for key in set(original) - {"X"}:
                with self.subTest(array=key):
                    assert_array_equal(migrated[key], original[key])
                    self.assertEqual(migrated[key].dtype, original[key].dtype)
            assert_array_equal(migrated["feature_layout"], np.full(4, D.FEATURE_LAYOUT))
        self.assertEqual(self.source.read_bytes(), source_bytes)
        self.assertEqual(self.lock.read_bytes(), lock_bytes)

    def test_canonical_input_is_not_reordered_again(self):
        for declared in (False, True):
            with self.subTest(declared=declared):
                original = self.arrays(canonical=True)
                if declared:
                    original["feature_layout"] = np.full(4, D.FEATURE_LAYOUT)
                self.write_source(original)
                self.migrate(layout=D.FEATURE_LAYOUT)
                with np.load(self.destination, allow_pickle=False) as migrated:
                    assert_array_equal(migrated["X"], original["X"])
                    assert_array_equal(migrated["feature_layout"], np.full(4, D.FEATURE_LAYOUT))
                self.destination.unlink()
                self.report.unlink()

    def test_matching_legacy_declaration_is_replaced_with_canonical_declaration(self):
        original = self.arrays()
        original["feature_layout"] = np.full(4, P.SOURCE_ECAPA_FIRST)
        self.write_source(original)

        self.migrate()

        with np.load(self.destination, allow_pickle=False) as migrated:
            assert_array_equal(migrated["X"], self.arrays(canonical=True)["X"])
            assert_array_equal(migrated["feature_layout"], np.full(4, D.FEATURE_LAYOUT))

    def test_provenance_hashes_identify_exact_files_and_record_both_layouts(self):
        self.write_source(self.arrays())
        source_hash = sha256(self.source)
        lock_hash = sha256(self.lock)

        returned = self.migrate()

        self.assertIsInstance(returned, dict)
        report = json.loads(self.report.read_text(encoding="utf-8"))
        self.assertEqual(report, returned)
        self.assertEqual(report["source_sha256"], source_hash)
        self.assertEqual(report["output_sha256"], sha256(self.destination))
        self.assertEqual(report["lock_test_sha256"], lock_hash)
        self.assertEqual(report["source_layout"], P.SOURCE_ECAPA_FIRST)
        self.assertEqual(report["target_layout"], D.FEATURE_LAYOUT)
        self.assertEqual(report["n_clips"], 4)
        self.assertEqual(report["n_speakers"], 2)
        self.assertEqual(report["acoustic_nan_count"], 0)
        self.assertEqual(report["transform"], "explicit_column_permutation_only")
        self.assertFalse(report["test_data_read"])
        self.assertEqual(sha256(self.source), source_hash)
        self.assertEqual(sha256(self.lock), lock_hash)

    def test_custom_provenance_path_is_used(self):
        self.write_source(self.arrays())
        custom = self.root / "synthetic_audit.json"

        returned = self.migrate(report_path=custom)

        self.assertEqual(json.loads(custom.read_text(encoding="utf-8")), returned)
        self.assertFalse(self.report.exists())

    def test_acoustic_nan_and_existing_availability_mask_are_preserved(self):
        for canonical in (False, True):
            with self.subTest(canonical=canonical):
                original = self.arrays(canonical=canonical)
                acoustic_index = 2 if canonical else 386
                original["X"][1, acoustic_index] = np.nan
                original["acoustic_available"][1, 2] = False
                self.write_source(original)
                layout = D.FEATURE_LAYOUT if canonical else P.SOURCE_ECAPA_FIRST

                self.migrate(layout=layout)

                with np.load(self.destination, allow_pickle=False) as migrated:
                    self.assertTrue(np.isnan(migrated["X"][1, 2]))
                    self.assertEqual(int(np.isnan(migrated["X"]).sum()), 1)
                    assert_array_equal(migrated["acoustic_available"], original["acoustic_available"])
                    expected = self.arrays(canonical=True)["X"]
                    expected[1, 2] = np.nan
                    assert_array_equal(migrated["X"], expected)
                self.destination.unlink()
                self.report.unlink()

    def test_contradictory_or_malformed_layout_metadata_is_rejected(self):
        cases = [
            (P.SOURCE_ECAPA_FIRST, np.full(4, D.FEATURE_LAYOUT)),
            (D.FEATURE_LAYOUT, np.full(4, P.SOURCE_ECAPA_FIRST)),
            (P.SOURCE_ECAPA_FIRST, np.array([P.SOURCE_ECAPA_FIRST] * 3 + [D.FEATURE_LAYOUT])),
            (P.SOURCE_ECAPA_FIRST, np.full((4, 1), P.SOURCE_ECAPA_FIRST)),
            (P.SOURCE_ECAPA_FIRST, np.array(P.SOURCE_ECAPA_FIRST)),
        ]
        for layout, declaration in cases:
            with self.subTest(layout=layout, shape=declaration.shape):
                original = self.arrays()
                original["feature_layout"] = declaration
                self.write_source(original)
                before = sha256(self.source)
                with self.assertRaises(ValueError):
                    self.migrate(layout=layout)
                self.assert_no_result()
                self.assertEqual(sha256(self.source), before)

    def test_wrong_feature_width_or_rank_is_rejected(self):
        for shape in ((4, 397), (4, 399), (4, 206), (4,), (4, 1, 398)):
            with self.subTest(shape=shape):
                original = self.arrays()
                original["X"] = np.zeros(shape)
                self.write_source(original)
                with self.assertRaises(ValueError):
                    self.migrate()
                self.assert_no_result()

    def test_nonfinite_embeddings_are_rejected_in_either_layout(self):
        for canonical in (False, True):
            for block_offset in (0, 192):
                for value in (np.nan, np.inf, -np.inf):
                    with self.subTest(canonical=canonical, block=block_offset, value=value):
                        original = self.arrays(canonical=canonical)
                        original["X"][0, block_offset + (14 if canonical else 0)] = value
                        self.write_source(original)
                        layout = D.FEATURE_LAYOUT if canonical else P.SOURCE_ECAPA_FIRST
                        with self.assertRaises(ValueError):
                            self.migrate(layout=layout)
                        self.assert_no_result()

    def test_acoustic_infinity_is_rejected(self):
        original = self.arrays()
        original["X"][0, 384] = np.inf
        self.write_source(original)
        with self.assertRaises(ValueError):
            self.migrate()
        self.assert_no_result()

    def test_missing_required_array_is_rejected(self):
        for key in D.REQUIRED_ARRAYS:
            with self.subTest(missing=key):
                original = self.arrays()
                del original[key]
                self.write_source(original)
                with self.assertRaises(ValueError):
                    self.migrate()
                self.assert_no_result()

    def test_locked_speaker_is_rejected_before_features_or_labels_are_loaded(self):
        original = self.arrays()
        original["speaker_id"] = np.array(["synthetic-a", "synthetic-b", "synthetic-locked-only", "synthetic-b"])
        self.write_source(original)
        before = sha256(self.source)
        archive = GuardedArchive(original, allowed={"speaker_id"})

        with mock.patch.object(P.np, "load", return_value=archive):
            with self.assertRaises((ValueError, SystemExit)):
                self.migrate()

        self.assertEqual(archive.reads, ["speaker_id"])
        self.assert_no_result()
        self.assertEqual(sha256(self.source), before)

    def test_nontraining_split_is_rejected_before_features_or_labels_are_loaded(self):
        for split in ("test", "dev", "validation", "unknown", "TRAIN"):
            with self.subTest(split=split):
                original = self.arrays()
                original["split"] = np.array(["train", "train", split, "train"])
                self.write_source(original)
                archive = GuardedArchive(original, allowed={"speaker_id", "split"})

                with mock.patch.object(P.np, "load", return_value=archive):
                    with self.assertRaises((ValueError, SystemExit)):
                        self.migrate()

                self.assertEqual(archive.reads[0], "speaker_id")
                self.assertIn("split", archive.reads)
                self.assert_no_result()

    def test_raw_or_mixed_feature_archives_are_rejected_without_loading_features(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                original = self.arrays()
                original.update(ecapa=np.zeros((4, 384)), acoustic=np.zeros((4, 14)))
                if not packed:
                    del original["X"]
                self.write_source(original)
                archive = GuardedArchive(original, allowed={"speaker_id", "split"})

                with mock.patch.object(P.np, "load", return_value=archive):
                    with self.assertRaises((ValueError, SystemExit)):
                        self.migrate()

                self.assertFalse(set(archive.reads) & {"X", "ecapa", "acoustic", "y_severity", "y_binary"})
                self.assert_no_result()

    def test_existing_output_is_not_overwritten(self):
        self.write_source(self.arrays())
        sentinel = b"synthetic existing output must survive"
        self.destination.write_bytes(sentinel)
        with self.assertRaises((FileExistsError, ValueError)):
            self.migrate()
        self.assertEqual(self.destination.read_bytes(), sentinel)
        self.assertFalse(self.report.exists())

    def test_existing_report_is_not_overwritten_or_accompanied_by_new_output(self):
        self.write_source(self.arrays())
        for path in (self.report, self.root / "synthetic_existing_report.json"):
            with self.subTest(report=path.name):
                sentinel = b"synthetic existing report must survive"
                path.write_bytes(sentinel)
                with self.assertRaises((FileExistsError, ValueError)):
                    self.migrate(report_path=None if path == self.report else path)
                self.assertEqual(path.read_bytes(), sentinel)
                self.assertFalse(self.destination.exists())
                path.unlink()

    def test_source_cannot_be_its_own_destination(self):
        self.write_source(self.arrays())
        before = self.source.read_bytes()
        with self.assertRaises((FileExistsError, ValueError)):
            P.prepare_training_data(self.source, self.source, self.lock, P.SOURCE_ECAPA_FIRST)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertFalse(self.source.with_suffix(".provenance.json").exists())

    def test_unsupported_source_layout_is_rejected_before_source_load(self):
        self.write_source(self.arrays())
        with mock.patch.object(P.np, "load", side_effect=AssertionError("Unsupported layout must not open NPZ")) as load:
            with self.assertRaises(ValueError):
                self.migrate(layout="infer-from-width")
        load.assert_not_called()
        self.assert_no_result()

    def test_cli_requires_explicit_data_output_lock_and_source_layout(self):
        values = {
            "--data": str(self.source),
            "--out": str(self.destination),
            "--lock-test": str(self.lock),
            "--source-layout": P.SOURCE_ECAPA_FIRST,
        }
        for missing in values:
            args = ["prepare_v2_data.py"]
            for flag, value in values.items():
                if flag != missing:
                    args.extend((flag, value))
            with self.subTest(missing=missing), mock.patch.object(sys, "argv", args), mock.patch(
                "sys.stderr"
            ), mock.patch.object(P.np, "load", side_effect=AssertionError("Incomplete CLI must not open NPZ")) as load:
                with self.assertRaises(SystemExit) as caught:
                    P.main()
                self.assertEqual(caught.exception.code, 2)
                load.assert_not_called()
                self.assert_no_result()


if __name__ == "__main__":
    unittest.main()
