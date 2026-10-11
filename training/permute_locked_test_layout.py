"""ONE-TIME, explicitly authorized column permutation of the LOCKED test file.

Why this script exists
----------------------
`data/pvqd_test.npz` is in the legacy producer order
`[emb-mean 0:192 | emb-std 192:384 | acoustic 384:398]`, proven structurally by
`verify_layout.py` (5/5 against 1/5 for the alternative). The modeling pipeline
slices by position after checking only the 398-column width, so evaluating a
canonically trained model on that file would select 192 embedding-mean columns
plus the first 14 embedding-std columns, drop every acoustic feature, and pass
every lock, manifest and width check on the way. The one-shot locked test set
would be spent on a number that measures nothing.

`prepare_v2_data.py` cannot do this: it rejects any file containing locked
speakers by design and has no test output path. That guard is deliberately
left untouched -- it is what keeps training paths away from the locked set.
This separate script is the recorded exception instead, approved explicitly by
the pipeline owner for exactly one file (see `--approved-by`). Running it
leaves a provenance JSON naming the approver, the hashes on both sides, the
permutation, and what was and was not read.

What it does, and does not do
-----------------------------
It is a pure column permutation of X. Nothing is fitted, imputed, filled,
dropped, rescaled or recomputed; the dtype is preserved, so every value in the
output is bit-identical to a value in the input. Label arrays are copied
verbatim and are never inspected: no label value enters any check, statistic,
threshold or decision here, and the only post-write comparison they take part
in is an identity check against the source. `speaker_id` is read, because the
guard below has to prove this really is the locked test file.

The original file is opened read-only and never written. Its hash is taken
before and after and must match, and the output is published exclusively: an
existing destination is never followed or overwritten.

Usage (single authorized invocation)
------------------------------------
    python permute_locked_test_layout.py \
        --data ../data/pvqd_test.npz \
        --source-layout embedding_mean192_embedding_std192_acoustic14 \
        --lock-test ../test_speakers.txt \
        --out ../data/pvqd_test_acoustic_first.npz \
        --approved-by "Ryo" --approval-date 2026-10-11 \
        --authorize-locked-test-permutation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

try:
    from . import data as D, verify_layout as V
except ImportError:
    import data as D
    import verify_layout as V

N_ACOUSTIC, N_EMB = D.N_ACOUSTIC, D.N_EMB
SOURCE_LAYOUT = V.EMBEDDING_FIRST

# Read [acoustic | emb-mean | emb-std] out of the legacy [mean | std | acoustic]
# order. Half-open column ranges, concatenated left to right.
COLUMN_RANGES = [[2 * N_EMB, 2 * N_EMB + N_ACOUSTIC], [0, N_EMB], [N_EMB, 2 * N_EMB]]

# Copied verbatim, never inspected. Keeping the list explicit is what makes the
# "no label value was read" claim in the provenance checkable.
LABEL_ARRAYS = ("y_severity", "y_binary", "severity_kind")

# Acoustic columns whose physical units make a transcription error obvious:
# a dB value, a frequency in Hz and a level in dB cannot be mistaken for the
# +-0.01 embedding columns they sit next to in the legacy order.
SPOT_CHECK_COLUMNS = {"cpps_db": 1, "f0_median_hz": 8, "intensity_median_db": 12}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(path):
    stat = Path(path).stat()
    # Access time changes on a read and does not indicate a content change.
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _stage_file(destination, writer):
    """Prepare a complete sibling file; the caller publishes it exclusively."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _read_locked_test_rows(source, locked, source_layout):
    """Read the locked test file, proving first that it is exactly that file.

    The speaker guard is the mirror image of the one in `prepare_v2_data.py`:
    that tool refuses a file containing ANY locked speaker, this one refuses a
    file containing anything else. A training file cannot be fed to it by
    mistake, and neither tool can be aimed at the other's input.
    """
    if source_layout != SOURCE_LAYOUT:
        raise ValueError(
            f"this script only converts {SOURCE_LAYOUT!r}; a file already in "
            f"{D.FEATURE_LAYOUT!r} needs no permutation and must not be rewritten")
    with np.load(source, allow_pickle=False) as archive:
        files = set(archive.files)
        if files & {"ecapa", "acoustic"}:
            raise ValueError("raw or mixed feature archives are not accepted; provide packed rows")
        missing = set(D.REQUIRED_ARRAYS) - files
        if missing:
            raise ValueError(f"input must be a packed NPZ; missing {sorted(missing)}")
        speakers = np.asarray(archive["speaker_id"], dtype=str)
        if speakers.ndim != 1 or not len(speakers) or np.any(np.char.str_len(speakers) == 0):
            raise ValueError("speaker_id must be a nonempty one-dimensional array of nonempty IDs")
        present = set(map(str, speakers))
        if present != locked:
            raise ValueError(
                "this script permutes the locked test file and nothing else, so its "
                "speakers must be exactly the locked list: "
                f"{len(sorted(locked - present))} locked speaker(s) missing from the input, "
                f"{len(sorted(present - locked))} non-locked speaker(s) present. Use "
                "prepare_v2_data.py for training files.")
        if "feature_layout" in files:
            declared = np.asarray(archive["feature_layout"], dtype=str)
            if declared.shape != (len(speakers),) or not np.all(declared == source_layout):
                raise ValueError(
                    "the input already declares a feature_layout that contradicts "
                    "--source-layout; refusing to reorder a file whose producer disagrees")
        rows = {key: archive[key] for key in archive.files}
    n = len(speakers)
    for key, value in rows.items():
        if np.ndim(value) == 0 or len(value) != n:
            raise ValueError(f"{key} must be a row array with {n} entries")
    X = rows["X"]
    if X.shape != (n, D.N_RAW_FEATURES) or not np.issubdtype(X.dtype, np.floating):
        raise ValueError(f"X must be {D.N_RAW_FEATURES} real floating-point columns; "
                         "no layout adaptation or dtype change is allowed")
    if np.isinf(X).any():
        raise ValueError("infinite features are forbidden; acoustic NaNs are preserved as they are")
    if not np.isfinite(X[:, :2 * N_EMB]).all():
        raise ValueError("non-finite embedding features are forbidden")
    return rows


def permute_locked_test(source, destination, lock_path, source_layout,
                        approved_by, approval_date, report_path=None):
    """Permute the locked test file into the canonical layout, once.

    Both directions are gated structurally: the input must prove it is in the
    legacy layout before a single column moves, and the output must prove it is
    canonical before it is published. The three blocks are then compared
    element by element against the source, so the permutation is verified
    rather than assumed.
    """
    source, destination, lock_path = map(Path, (source, destination, lock_path))
    source = source.resolve()
    destination = destination.absolute()
    lock_path = lock_path.resolve()
    report_path = (Path(report_path).absolute() if report_path is not None
                   else destination.with_suffix(".provenance.json"))
    if destination.suffix != ".npz":
        raise ValueError("destination must have an explicit .npz suffix")
    paths = [source, destination.resolve(), lock_path, report_path.resolve()]
    if len(set(paths)) != len(paths):
        raise ValueError("source, output, lock and report paths must be distinct")
    if not approved_by.strip() or not approval_date.strip():
        raise ValueError("the authorizing person and date are recorded in the provenance; both are required")
    # Includes dangling symlinks: never follow an existing destination for writes.
    for path in [destination, report_path]:
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    lock_bytes = lock_path.read_bytes()
    locked = {line.strip() for line in lock_bytes.decode("utf-8").splitlines() if line.strip()}
    if not locked:
        raise ValueError("lock list must not be empty")
    lock_hash = hashlib.sha256(lock_bytes).hexdigest()

    source_signature = _signature(source)
    source_hash = _sha256(source)
    rows = _read_locked_test_rows(source, locked, source_layout)
    X = rows["X"]

    # Gate one: the input is provably in the legacy layout. Asserting this
    # before any column moves is what makes the permutation below the right
    # one; the --source-layout argument alone would only be a claim.
    source_evidence = V.assert_layout(X, SOURCE_LAYOUT, label=str(source))

    converted = dict(rows)
    converted["X"] = np.concatenate([X[:, start:end] for start, end in COLUMN_RANGES], axis=1)
    if converted["X"].dtype != X.dtype or converted["X"].shape != X.shape:
        raise RuntimeError("a column permutation must preserve the shape and dtype of X")
    converted["feature_layout"] = np.full(len(X), D.FEATURE_LAYOUT)

    # Gate two: the result is provably canonical, declaration included. The
    # same gate every reader of an NPZ now has to pass before slicing.
    output_evidence = V.assert_canonical_input(converted, label=str(destination))
    D.clip_labels(converted)
    D.acoustic_availability(converted)

    acoustic = np.asarray(converted["X"][:, :N_ACOUSTIC], dtype=np.float64)
    provenance = {
        "schema_version": 1,
        "transform": "explicit_column_permutation_only",
        "operation": "one_time_authorized_locked_test_permutation",
        "authorization": {
            "approved_by": approved_by.strip(),
            "approval_date": approval_date.strip(),
            "required_flag": "--authorize-locked-test-permutation",
            "scope": "exactly one read-only permutation of the locked test file into the canonical layout",
            "why_not_prepare_v2_data": (
                "prepare_v2_data.py refuses any file containing locked speakers by design and has "
                "no test output path. That guard is unchanged. This separate, explicitly authorized "
                "script is the recorded exception, so the exception is auditable instead of being "
                "carved into the guard that keeps training away from the locked set."),
        },
        "source_path": str(source), "output_path": str(destination),
        "source_sha256": source_hash, "lock_test_sha256": lock_hash,
        "source_layout": source_layout, "target_layout": D.FEATURE_LAYOUT,
        "layout_authority": (
            f"structural proof by verify_layout.assert_layout on both sides "
            f"({source_evidence['score']}/{V.MAX_LAYOUT_SCORE} for the source layout, "
            f"{output_evidence['score']}/{V.MAX_LAYOUT_SCORE} for the canonical output), "
            f"corroborating the explicit --source-layout declaration"),
        "layout_evidence": {
            "source": {"layout": source_evidence["layout"], "score": source_evidence["score"],
                       "checks": source_evidence["checks"]},
            "output": {"layout": output_evidence["layout"], "score": output_evidence["score"],
                       "checks": output_evidence["checks"]},
        },
        "columns": COLUMN_RANGES,
        "column_rule": ("output[:, 0:14] = input[:, 384:398]; output[:, 14:206] = input[:, 0:192]; "
                        "output[:, 206:398] = input[:, 192:384]"),
        "dtype": str(X.dtype),
        "n_clips": int(len(X)), "n_speakers": int(len(set(map(str, rows["speaker_id"])))),
        "acoustic_nan_count": int(np.isnan(acoustic).sum()),
        "availability_source": ("preserved_input_mask" if "acoustic_available" in rows
                                else "not_supplied_not_inferred"),
        "preserved_arrays": sorted(set(rows) - {"X", "feature_layout"}),
        "test_feature_columns_read": True,
        "test_labels_read": False,
        "label_arrays": list(LABEL_ARRAYS),
        "data_access_note": (
            "X was read to prove the source layout and to permute it. speaker_id was read to "
            "prove the input is exactly the locked test file. Label arrays "
            f"({', '.join(LABEL_ARRAYS)}) were copied verbatim and never inspected: no label "
            "value entered any check, statistic, threshold or decision, and the only comparison "
            "they take part in is the post-write identity check against the source."),
        "preprocessing_fit": False,
        "model_or_hyperparameter_decision": False,
        "limitations": [
            "No model, imputer, scaler or norm is fitted. Existing model bundles cannot be reused "
            "with permuted inputs; a bundle fitted on the canonical training copy is required.",
            "Finite inputs cannot reveal missing values filled upstream; no availability mask is "
            "invented, and the -200.0 HNR sentinel is left exactly as the producer wrote it.",
            "A proven column order says nothing about the acoustic regime difference between the "
            "augmented training rows and these clean test rows (reports/layout_and_sentinel_check.md).",
        ],
    }
    provenance["dimension_spot_check"] = {
        name: {"column": column, "mean": float(np.nanmean(acoustic[:, column])),
               "min": float(np.nanmin(acoustic[:, column])),
               "max": float(np.nanmax(acoustic[:, column]))}
        for name, column in SPOT_CHECK_COLUMNS.items()
    }

    temporary_npz = temporary_report = None
    published = []
    try:
        temporary_npz = _stage_file(destination, lambda f: np.savez_compressed(f, **converted))
        provenance["output_sha256"] = _sha256(temporary_npz)
        provenance["verification"] = _verify_against_source(temporary_npz, rows, converted)
        # The point of the whole exercise is that the locked original is still
        # the file it was, so this is recorded inside the provenance itself.
        provenance["source_sha256_after_conversion"] = _sha256(source)
        provenance["source_unmodified"] = (
            provenance["source_sha256_after_conversion"] == source_hash
            and _signature(source) == source_signature)
        if not provenance["source_unmodified"]:
            raise RuntimeError("the locked test file changed while it was being read; nothing published")
        payload = (json.dumps(provenance, indent=2, allow_nan=False) + "\n").encode()
        temporary_report = _stage_file(report_path, lambda f: f.write(payload))
        if _sha256(lock_path) != lock_hash:
            raise RuntimeError("the lock list changed during preparation; nothing published")
        # Hard links publish atomically and fail if a destination now exists.
        os.link(temporary_npz, destination)
        published.append(destination)
        os.link(temporary_report, report_path)
        published.append(report_path)
    except BaseException:
        for path in published:
            path.unlink()
        raise
    finally:
        for temporary in [temporary_npz, temporary_report]:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if _sha256(source) != source_hash or _signature(source) != source_signature:
        raise RuntimeError("the locked test file changed after the output was published")
    return provenance


def _identical(left, right):
    """Element-wise identity, treating NaN as equal to NaN, for any dtype.

    Used for the row arrays, including the labels: this comparison is the only
    thing done with them, and it reads no value for any decision.
    """
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.floating):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _verify_against_source(output_path, source_rows, converted):
    """Re-read the written file and prove it is the permutation it claims to be.

    Values are compared in their stored dtype, so the three blocks must match
    element for element, not approximately. Row arrays are compared for
    identity only, which is also the entire extent to which labels are touched.
    """
    with np.load(output_path, allow_pickle=False) as archive:
        written = {key: archive[key] for key in archive.files}
    if set(written) != set(converted):
        raise RuntimeError("the written file does not have the same row arrays as the conversion")
    source_X, written_X = source_rows["X"], written["X"]
    if written_X.dtype != source_X.dtype:
        raise RuntimeError("the written X changed dtype; this is not a pure permutation")
    blocks = {}
    for name, (start, end), (source_start, source_end) in (
            ("acoustic14", (0, N_ACOUSTIC), COLUMN_RANGES[0]),
            ("embedding_mean192", (N_ACOUSTIC, N_ACOUSTIC + N_EMB), COLUMN_RANGES[1]),
            ("embedding_std192", (N_ACOUSTIC + N_EMB, D.N_RAW_FEATURES), COLUMN_RANGES[2])):
        equal = bool(np.array_equal(written_X[:, start:end],
                                    source_X[:, source_start:source_end], equal_nan=True))
        if not equal:
            raise RuntimeError(f"block {name} is not element-wise identical to the source columns")
        blocks[name] = {"output_columns": [start, end],
                        "source_columns": [source_start, source_end],
                        "elementwise_identical": equal}
    unchanged = {}
    for key in sorted(set(written) - {"X", "feature_layout"}):
        identical = _identical(written[key], source_rows[key])
        if not identical:
            raise RuntimeError(f"row array {key} is not identical to the source")
        unchanged[key] = identical
    if not np.array_equal(np.sort(written_X, axis=None), np.sort(source_X, axis=None), equal_nan=True):
        raise RuntimeError("the written X is not a permutation of the source values")
    reloaded = V.assert_canonical_input(written, label=str(output_path))
    return {
        "reread_from_disk": True,
        "blocks": blocks,
        "row_arrays_identical": unchanged,
        "multiset_of_values_unchanged": True,
        "output_layout_proof": {"layout": reloaded["layout"], "score": reloaded["score"],
                                "max_score": V.MAX_LAYOUT_SCORE, "checks": reloaded["checks"]},
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True,
                        help="the locked test NPZ, opened read-only and never modified")
    parser.add_argument("--source-layout", choices=[SOURCE_LAYOUT], required=True,
                        help="the producer order of the input; also proven structurally before use")
    parser.add_argument("--lock-test", type=Path, required=True,
                        help="locked speaker list; the input must contain exactly these speakers")
    parser.add_argument("--out", type=Path, required=True,
                        help="new canonical .npz file; an existing path is never overwritten")
    parser.add_argument("--approved-by", required=True,
                        help="the person authorizing this one-time permutation; recorded in the provenance")
    parser.add_argument("--approval-date", required=True, help="date of that authorization, ISO 8601")
    parser.add_argument("--authorize-locked-test-permutation", action="store_true", required=True,
                        help="explicit acknowledgement that this writes a copy of the LOCKED test set")
    parser.add_argument("--report", type=Path, help="new provenance JSON, default <out>.provenance.json")
    args = parser.parse_args()
    provenance = permute_locked_test(args.data, args.out, args.lock_test, args.source_layout,
                                     args.approved_by, args.approval_date, args.report)
    print(json.dumps(provenance, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
