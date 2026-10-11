"""Synthetic-only checks for clean provenance and guarded, exclusive output."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from numpy.testing import assert_array_equal

try:
    from . import data as D, prepare_clean_training as C
    from .test_prepare_v2_data import GuardedArchive
except ImportError:
    import data as D
    import prepare_clean_training as C
    from test_prepare_v2_data import GuardedArchive


class CleanTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="vocal-clean-synthetic-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "packed.npz"
        self.clean = self.root / "producer_clean.npz"
        self.augmented = self.root / "producer_augmented.npz"
        self.lock = self.root / "lock.txt"
        self.output = self.root / "clean.npz"
        self.report = self.root / "provenance.json"
        self.lock.write_text("excluded\n", encoding="utf-8")
        rng = np.random.default_rng(91)
        emb = rng.normal(size=(4, 384)).astype(np.float32)
        acoustic = rng.normal(size=(4, 14)).astype(np.float32)
        acoustic[0, 0] = np.nan
        self.rows = {
            "X": np.concatenate([acoustic, emb], axis=1),
            "speaker_id": np.array(["a", "b", "a", "b"]),
            "y_binary": np.array([0, 1, 0, 1], dtype=np.int32),
            "y_severity": np.array([12.5, 78.25, 12.5, 78.25], dtype=np.float32),
            "severity_kind": np.full(4, "cape_v"),
            "corpus": np.full(4, "synthetic"),
            "feature_layout": np.full(4, D.FEATURE_LAYOUT),
            "recording_metadata": np.arange(8, dtype=np.int16).reshape(4, 2),
            "acoustic_available": np.isfinite(acoustic),
        }
        self.clean_arrays = self.producer([0, 1], ["a", "b"])
        # A nontraining row collides with a real augmented row. A correct
        # speaker filter must never consider this row's feature/target values.
        for key, value in list(self.clean_arrays.items()):
            extra = self.producer([2], ["excluded"])[key]
            self.clean_arrays[key] = np.concatenate([value, extra])
        self.clean_arrays["speaker_id"][-1] = "excluded"
        self.clean_arrays["split"][-1] = "other"
        self.augmented_arrays = self.producer([2, 3], ["a_phone_wide", "b_phone_narrow"])
        self.save()

    def producer(self, indices, clips):
        return {
            "ecapa": self.rows["X"][indices, 14:],
            "acoustic": self.rows["X"][indices, :14],
            "speaker_id": self.rows["speaker_id"][indices].astype("<U20"),
            "clip_id": np.asarray(clips, dtype="<U30"),
            "split": np.full(len(indices), "train", dtype="<U8"),
            "label": self.rows["y_binary"][indices],
            "severity": self.rows["y_severity"][indices],
        }

    def save(self):
        for path, rows in ((self.source, self.rows), (self.clean, self.clean_arrays),
                           (self.augmented, self.augmented_arrays)):
            np.savez_compressed(path, **rows)

    def prepare(self):
        return C.prepare_clean_training(self.source, self.clean, self.augmented,
                                        self.lock, self.output, self.report)

    def test_subset_preserves_raw_arrays_and_provenance(self):
        initial = {p: p.read_bytes() for p in (self.source, self.clean, self.augmented, self.lock)}
        report = self.prepare()
        with np.load(self.output, allow_pickle=False) as result:
            for key, value in self.rows.items():
                assert_array_equal(result[key], value[:2])
                self.assertEqual(result[key].dtype, value.dtype)
            assert_array_equal(result["origin_variant"], ["clean", "clean"])
            assert_array_equal(result["packed_source_row_index"], [0, 1])
            assert_array_equal(result["source_row_index"], [0, 1])
        self.assertEqual(report["variant_counts"], {"clean": 2, "phone_narrow": 1, "phone_wide": 1})
        self.assertEqual(report["producer_counts"]["clean"]["excluded_speaker_rows"], 1)
        self.assertEqual(report["label_checks"], {"binary": 4, "native_severity": 4, "parent_inheritance": 2})
        self.assertEqual(report["paired_embedding_shift"]["phone_wide"]["F2_embedding_mean192"]["n_pairs"], 1)
        self.assertEqual(json.loads(self.report.read_text()), report)
        for path, content in initial.items():
            self.assertEqual(path.read_bytes(), content)

    def test_guard_precedes_train_features_and_all_producer_access(self):
        guarded_rows = dict(self.rows)
        guarded_rows["speaker_id"] = np.array(["excluded"] * 4)
        archive = GuardedArchive(guarded_rows, {"speaker_id"})
        with mock.patch.object(C.P.np, "load", return_value=archive) as load:
            with mock.patch.object(C, "_producer_rows", side_effect=AssertionError("producer accessed")):
                with self.assertRaisesRegex(SystemExit, "LOCKED TEST SPEAKERS"):
                    self.prepare()
        self.assertEqual(load.call_count, 1)
        self.assertEqual(archive.reads, ["speaker_id"])
        self.assertFalse(self.output.exists())

    def test_only_selected_producer_feature_rows_are_decoded(self):
        with mock.patch.object(C.np, "frombuffer", wraps=np.frombuffer) as decode:
            selected = C._selected_member(self.clean, "ecapa", np.array([0, 1]), 3)
        assert_array_equal(selected, self.clean_arrays["ecapa"][:2])
        self.assertEqual(decode.call_count, 2)
        self.assertTrue(all(len(call.args[0]) == 384 * 4 for call in decode.call_args_list))

    def test_clean_mask_is_recovered_after_packed_rows_are_shuffled(self):
        order = np.array([2, 1, 3, 0])
        self.rows = {key: value[order] for key, value in self.rows.items()}
        self.save()
        report = self.prepare()
        self.assertEqual(report["clean_packed_row_indices"], [1, 3])
        self.assertEqual(report["clean_producer_row_indices"], [1, 0])
        with np.load(self.output, allow_pickle=False) as result:
            assert_array_equal(result["X"], self.rows["X"][[1, 3]])
            assert_array_equal(result["speaker_id"], ["b", "a"])

    def test_exact_embedding_collision_is_rejected(self):
        self.augmented_arrays["ecapa"][0] = self.clean_arrays["ecapa"][0]
        self.save()
        with self.assertRaisesRegex(ValueError, "embedding collision"):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report.exists())

    def test_unmatched_embedding_is_rejected(self):
        self.rows["X"][0, 14] += 1
        self.save()
        with self.assertRaisesRegex(ValueError, "no exact producer"):
            self.prepare()

    def test_acoustic_speaker_and_label_mismatches_are_rejected(self):
        mutations = [
            ("acoustic", (0, 1), 99, "acoustic mismatch"),
            ("speaker_id", 0, "b", "speaker mismatch"),
            ("label", 0, 1, "binary label mismatch"),
            ("severity", 0, 99, "native severity label mismatch"),
        ]
        for key, index, value, error in mutations:
            with self.subTest(key=key):
                previous = self.clean_arrays[key][index].copy()
                self.clean_arrays[key][index] = value
                self.save()
                with self.assertRaisesRegex(ValueError, error):
                    self.prepare()
                self.clean_arrays[key][index] = previous
                self.assertFalse(self.output.exists())

    def test_parent_labels_must_be_inherited_even_if_both_sources_agree(self):
        self.rows["y_severity"][2] = 20
        self.augmented_arrays["severity"][0] = 20
        self.save()
        with self.assertRaisesRegex(ValueError, "parent inheritance mismatch"):
            self.prepare()

    def test_existing_output_or_report_is_never_overwritten(self):
        for path in (self.output, self.report):
            with self.subTest(path=path.name):
                path.write_bytes(b"existing artifact")
                with self.assertRaises(FileExistsError):
                    self.prepare()
                self.assertEqual(path.read_bytes(), b"existing artifact")
                path.unlink()

    def test_atomic_publish_rolls_back_if_report_path_appears(self):
        original_link = C.os.link
        def racing_link(source, destination):
            if Path(destination) == self.report:
                self.report.write_bytes(b"someone else's report")
            original_link(source, destination)
        with mock.patch.object(C.os, "link", side_effect=racing_link):
            with self.assertRaises(FileExistsError):
                self.prepare()
        self.assertFalse(self.output.exists())
        self.assertEqual(self.report.read_bytes(), b"someone else's report")
        self.assertEqual(list(self.root.glob(".clean.npz.*")), [])


if __name__ == "__main__":
    unittest.main()
