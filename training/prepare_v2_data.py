"""Explicit, train-only migration into the v2 acoustic-first feature layout.

Inputs must already be packed TRAIN-ONLY NPZ files with the data.py row
contract. Combined raw archives are rejected, not split here. There is no test
output path, no default data path, and no automatic guess of column order.

Example after the producer confirms BOTH the ECAPA mean/std order and the
14-column acoustic order:
    python prepare_v2_data.py --data verified_legacy_train.npz \
        --source-layout embedding_mean192_embedding_std192_acoustic14 \
        --lock-test synthetic_lock.txt --out migrated_train.npz

This is a column permutation, not imputation or a model/scaler migration.
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
    from . import data as D
except ImportError:
    import data as D

SOURCE_ECAPA_FIRST = "embedding_mean192_embedding_std192_acoustic14"
SOURCE_LAYOUTS = (SOURCE_ECAPA_FIRST, D.FEATURE_LAYOUT)


def _source_signature(path):
    stat = Path(path).stat()
    # Access times may change on reads; they do not indicate content changes.
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_training_rows(source, locked, source_layout):
    """Inspect IDs before any feature/target arrays, then validate all rows."""
    if source_layout not in SOURCE_LAYOUTS:
        raise ValueError("source_layout must explicitly name a supported, producer-verified layout")
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) & {"ecapa", "acoustic"}:
            raise ValueError("raw or mixed feature archives are not accepted; provide only packed train-only rows")
        missing = set(D.REQUIRED_ARRAYS) - set(archive.files)
        if missing:
            raise ValueError(
                f"input must be a packed train-only NPZ; missing {sorted(missing)}. "
                "Combined raw archives are not accepted or split by this tool")
        speakers = np.asarray(archive["speaker_id"], dtype=str)
        if speakers.ndim != 1 or not len(speakers) or np.any(np.char.str_len(speakers) == 0):
            raise ValueError("speaker_id must be a nonempty one-dimensional array of nonempty IDs")
        D.assert_no_leak(speakers, np.array([], dtype=str), locked)
        n = len(speakers)
        if "split" in archive.files:
            split = np.asarray(archive["split"], dtype=str)
            if split.shape != (n,) or not np.all(split == "train"):
                raise ValueError("only rows explicitly marked train are accepted; do not pass mixed split inputs")
        if "feature_layout" in archive.files:
            declared = np.asarray(archive["feature_layout"], dtype=str)
            if declared.shape != (n,) or not np.all(declared == source_layout):
                raise ValueError("input feature_layout contradicts the explicit source_layout; refusing to reorder")
        # Reaching this line establishes no locked speaker is in the input.
        rows = {key: archive[key] for key in archive.files}
    for key, value in rows.items():
        if np.ndim(value) == 0 or len(value) != n:
            raise ValueError(f"{key} must be a row array with {n} entries")
    X = np.asarray(rows["X"])
    if X.shape != (n, D.N_RAW_FEATURES) or not np.issubdtype(X.dtype, np.number):
        raise ValueError("X must have exactly 398 numeric columns; no layout adaptation is allowed")
    if not np.issubdtype(X.dtype, np.floating):
        raise ValueError("X must contain real floating-point features")
    if np.isinf(X).any():
        raise ValueError("infinite features are forbidden; acoustic NaNs are preserved without filling")
    embedding = X[:, :384] if source_layout == SOURCE_ECAPA_FIRST else X[:, 14:]
    if not np.isfinite(embedding).all():
        raise ValueError("non-finite embedding features are forbidden")
    for key in ["y_severity", "severity_kind", "y_binary", "speaker_id", "corpus"]:
        if np.shape(rows[key]) != (n,):
            raise ValueError(f"{key} must be a one-dimensional row array")
    if not np.all(np.isfinite(rows["y_severity"])):
        raise ValueError("train-only severity labels must be finite; targets are never filled")
    if not np.all(np.isin(rows["y_binary"], [0, 1])):
        raise ValueError("y_binary must contain only 0 and 1")
    if any(np.any(np.char.str_len(np.asarray(rows[key], dtype=str)) == 0)
           for key in ["corpus", "severity_kind"]):
        raise ValueError("corpus and severity_kind must be nonempty")
    return rows


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


def prepare_training_data(source, destination, lock_path, source_layout, report_path=None):
    """Reorder verified train-only rows into a new NPZ and write provenance.

    No inference of missingness is made from finite values. Existing masks and
    all row metadata are preserved; true acoustic NaNs remain NaNs. The explicit
    layout argument is a producer declaration, not proof inferred from width.
    """
    source, destination, lock_path = map(Path, (source, destination, lock_path))
    source = source.resolve()
    destination = destination.absolute()
    lock_path = lock_path.resolve()
    report_path = Path(report_path).absolute() if report_path is not None else destination.with_suffix(".provenance.json")
    if destination.suffix != ".npz":
        raise ValueError("destination must have an explicit .npz suffix")
    paths = [source, destination.resolve(), lock_path, report_path.resolve()]
    if len(set(paths)) != len(paths):
        raise ValueError("source, output, lock, and report paths must be distinct")
    # Includes dangling symlinks; never follow an existing destination for writes.
    for path in [destination, report_path]:
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
    lock_bytes = lock_path.read_bytes()
    locked = {line.strip() for line in lock_bytes.decode("utf-8").splitlines() if line.strip()}
    if not locked:
        raise ValueError("lock list must not be empty")
    lock_hash = hashlib.sha256(lock_bytes).hexdigest()
    source_stat = _source_signature(source)
    rows = _read_training_rows(source, locked, source_layout)
    # Hash only after the ID guard; rejected archives' feature bytes are not read.
    source_hash = _sha256(source)
    if _source_signature(source) != source_stat:
        raise RuntimeError("source changed while validating; retry with an immutable training file")
    X = rows["X"]
    columns = [[384, 398], [0, 192], [192, 384]] if source_layout == SOURCE_ECAPA_FIRST else [[0, 398]]
    converted = dict(rows)
    converted["X"] = np.concatenate([X[:, start:end] for start, end in columns], axis=1)
    converted["feature_layout"] = np.full(len(X), D.FEATURE_LAYOUT)
    D.validate_feature_layout(converted, required=True)
    D.clip_labels(converted)
    D.acoustic_availability(converted)
    provenance = {
        "schema_version": 1,
        "transform": "explicit_column_permutation_only",
        "source_path": str(source), "output_path": str(destination),
        "source_sha256": source_hash, "lock_test_sha256": lock_hash,
        "source_layout": source_layout, "target_layout": D.FEATURE_LAYOUT,
        "layout_authority": "explicit caller declaration; verify producer order before real-data use",
        "columns": columns, "n_clips": len(X),
        "n_speakers": len(np.unique(rows["speaker_id"])),
        "acoustic_nan_count": int(np.isnan(converted["X"][:, :14]).sum()),
        "availability_source": "preserved_input_mask" if "acoustic_available" in rows else "not_supplied_not_inferred",
        "preserved_arrays": sorted(set(rows) - {"X", "feature_layout"}),
        "test_data_read": False,
        "preprocessing_fit": False,
        "limitations": [
            "No model, imputer, scaler, or norm is fitted. Existing model bundles cannot be reused with permuted inputs.",
            "Finite inputs cannot reveal missing values filled upstream; no availability mask is invented.",
        ],
    }
    temporary_npz = temporary_report = None
    published = []
    try:
        temporary_npz = _stage_file(destination, lambda f: np.savez_compressed(f, **converted))
        provenance["output_sha256"] = _sha256(temporary_npz)
        payload = (json.dumps(provenance, indent=2, allow_nan=False) + "\n").encode()
        temporary_report = _stage_file(report_path, lambda f: f.write(payload))
        if _sha256(lock_path) != lock_hash or _source_signature(source) != source_stat:
            raise RuntimeError("source or lock changed during preparation; no output published")
        # Hard links publish atomically and fail if a destination now exists.
        # Both staged files are on their respective destination filesystems.
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
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="packed, verified train-only NPZ")
    parser.add_argument("--source-layout", choices=SOURCE_LAYOUTS, required=True,
                        help="producer-verified source order; never inferred from 398 columns")
    parser.add_argument("--lock-test", type=Path, required=True, help="speaker exclusion list only")
    parser.add_argument("--out", type=Path, required=True, help="new train-only .npz file; never overwrite")
    parser.add_argument("--report", type=Path, help="new provenance JSON, default <out>.provenance.json")
    args = parser.parse_args()
    report = prepare_training_data(args.data, args.out, args.lock_test, args.source_layout, args.report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
