"""Recover verified clean training rows without fitting or touching test inputs.

The packed input must declare [14 acoustic | 192 mean | 192 std]. Producer
speaker IDs are inspected first; only members belonging to those already
guarded training speakers are decoded from producer feature/label arrays.
Raw archive hashes cover opaque bytes, including excluded rows. No test NPZ
is accepted by this CLI. Both the subset and its provenance must be new files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np

try:
    from . import data as D, prepare_v2_data as P
except ImportError:
    import data as D
    import prepare_v2_data as P


def _selected_member(path, key, indices, n_rows):
    """Decode selected rows only; seek past excluded rows without interpreting.

    This deliberately does not use ``np.load(path)[key][mask]``, which would
    first materialize nontraining feature/label values. ZIP may decompress
    opaque skipped bytes internally; only selected row bytes become arrays.
    """
    with zipfile.ZipFile(path) as archive:
        member = f"{key}.npy"
        if archive.namelist().count(member) != 1:
            raise ValueError(f"{path}: missing or duplicated NPZ member {key!r}")
        with archive.open(member) as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"{path}: unsupported NPY header version {version}")
            if not shape or shape[0] != n_rows or dtype.hasobject:
                raise ValueError(f"{path}: {key} must be a non-object row array with {n_rows} rows")
            if fortran and len(shape) > 1:
                raise ValueError(f"{path}: {key} must be stored in C row order for selective decoding")
            row_values = int(np.prod(shape[1:], dtype=np.int64))
            row_bytes = row_values * dtype.itemsize
            start = handle.tell()
            selected = np.empty((len(indices), *shape[1:]), dtype=dtype)
            for dest, source in enumerate(indices):
                handle.seek(start + int(source) * row_bytes)
                payload = handle.read(row_bytes)
                if len(payload) != row_bytes:
                    raise ValueError(f"{path}: truncated {key} row {source}")
                selected[dest] = np.frombuffer(payload, dtype=dtype).reshape(shape[1:])
    return selected


def _producer_rows(path, training_speakers):
    with np.load(path, allow_pickle=False) as archive:
        required = {"speaker_id", "clip_id", "ecapa", "acoustic"}
        if not required <= set(archive.files):
            raise ValueError(f"{path}: producer missing {sorted(required - set(archive.files))}")
        # Speaker identifiers alone are allowed before the producer row filter.
        speakers = np.asarray(archive["speaker_id"], dtype=str)
        if speakers.ndim != 1:
            raise ValueError(f"{path}: producer speaker_id must be one-dimensional")
        keys = required | (set(archive.files) & {"split", "label", "severity", "grbas_grade", "valid"})
    indices = np.flatnonzero(np.isin(speakers, list(training_speakers)))
    result = {"speaker_id": speakers[indices]}
    for key in sorted(keys - {"speaker_id"}):
        result[key] = _selected_member(path, key, indices, len(speakers))
    n = len(indices)
    if result["ecapa"].shape != (n, 384) or result["acoustic"].shape != (n, 14):
        raise ValueError(f"{path}: producer requires ecapa384 and acoustic14")
    for key in keys - {"ecapa", "acoustic", "speaker_id"}:
        if result[key].shape != (n,):
            raise ValueError(f"{path}: {key} must be a one-dimensional row array")
    if "split" in result and not np.all(result["split"].astype(str) == "train"):
        raise ValueError(f"{path}: a matched training speaker has a non-train producer split")
    return result, indices, len(speakers)


def _embedding_key(row):
    row = np.asarray(row)
    if not np.issubdtype(row.dtype, np.floating):
        raise ValueError("producer embeddings must contain real floating-point values")
    if not np.isfinite(row).all():
        return None
    encoded = row.astype("<f4")
    if not np.isfinite(encoded).all():
        raise ValueError("embedding cannot be represented as finite float32")
    return encoded.tobytes()


def _pair_summary(clean, augmented):
    delta = np.linalg.norm(augmented - clean, axis=1)
    norm = np.linalg.norm(clean, axis=1)
    other_norm = np.linalg.norm(augmented, axis=1)
    cosine = 1.0 - np.sum(clean * augmented, axis=1) / np.maximum(norm * other_norm, 1e-12)
    return {
        "n_pairs": len(clean),
        "exact_equal_pairs": int(np.all(clean == augmented, axis=1).sum()),
        "median_euclidean": float(np.median(delta)),
        "median_cosine_distance": float(np.median(cosine)),
        "mean_cosine_distance": float(np.mean(cosine)),
        "median_shift_over_clean_norm": float(np.median(delta / np.maximum(norm, 1e-12))),
    }


def recover_clean_rows(rows, clean_path, augmented_path):
    """Exact float32 provenance, plus independent acoustic/ID/label checks."""
    speakers = set(np.asarray(rows["speaker_id"], dtype=str))
    candidates = {}
    producer_counts = {}
    for origin, path in (("clean", clean_path), ("augmented", augmented_path)):
        producer, original_indices, total_rows = _producer_rows(path, speakers)
        skipped = 0
        seen_clips = set()
        for index, source_index in enumerate(original_indices):
            clip = str(producer["clip_id"][index])
            if not clip or clip in seen_clips:
                raise ValueError(f"{origin}: empty or duplicated producer clip_id {clip!r}")
            seen_clips.add(clip)
            variant = "clean"
            if origin == "augmented":
                suffixes = [v for v in ("phone_wide", "phone_narrow") if clip.endswith("_" + v)]
                if len(suffixes) != 1:
                    raise ValueError(f"unrecognized augmented variant in clip_id {clip!r}")
                variant = suffixes[0]
            key = _embedding_key(producer["ecapa"][index])
            if key is None:
                skipped += 1
                continue
            if key in candidates:
                raise ValueError("ambiguous provenance: exact embedding collision across producer rows")
            candidates[key] = (variant, int(source_index), index, clip, producer)
        producer_counts[origin] = {
            "archive_rows": total_rows,
            "training_speaker_rows": len(original_indices),
            "excluded_speaker_rows": total_rows - len(original_indices),
            "nonfinite_embeddings_skipped": skipped,
        }

    variants, source_indices, clips = [], [], []
    used_sources = set()
    label_checks = {"binary": 0, "native_severity": 0, "parent_inheritance": 0}
    for packed_index, embedding in enumerate(rows["X"][:, 14:]):
        hit = candidates.get(_embedding_key(embedding))
        if hit is None:
            raise ValueError(f"packed row {packed_index}: no exact producer embedding match")
        variant, source_index, index, clip, producer = hit
        identity = ("clean" if variant == "clean" else "augmented", source_index)
        if identity in used_sources:
            raise ValueError(f"packed row {packed_index}: duplicate use of one producer row")
        used_sources.add(identity)
        if str(producer["speaker_id"][index]) != str(rows["speaker_id"][packed_index]):
            raise ValueError(f"packed row {packed_index}: speaker mismatch")
        if not np.array_equal(producer["acoustic"][index], rows["X"][packed_index, :14], equal_nan=True):
            raise ValueError(f"packed row {packed_index}: acoustic mismatch")
        if "valid" in producer and not bool(producer["valid"][index]):
            raise ValueError(f"packed row {packed_index}: producer marks matched row invalid")
        if "label" in producer:
            if producer["label"][index] != rows["y_binary"][packed_index]:
                raise ValueError(f"packed row {packed_index}: binary label mismatch")
            label_checks["binary"] += 1
        severity_key = {"cape_v": "severity", "grbas": "grbas_grade"}.get(str(rows["severity_kind"][packed_index]))
        if severity_key in producer:
            if producer[severity_key][index] != rows["y_severity"][packed_index]:
                raise ValueError(f"packed row {packed_index}: native severity label mismatch")
            label_checks["native_severity"] += 1
        variants.append(variant)
        source_indices.append(source_index)
        clips.append(clip)

    variants = np.asarray(variants)
    indices = np.flatnonzero(variants == "clean")
    if not len(indices):
        raise ValueError("no verified clean rows in packed training input")
    parent_rows = {clips[r]: int(r) for r in indices}
    paired_shift = {}
    for variant in ("phone_wide", "phone_narrow"):
        pairs = []
        for row in np.flatnonzero(variants == variant):
            base = clips[row].removesuffix("_" + variant)
            if base not in parent_rows:
                raise ValueError(f"augmented clip {clips[row]!r} has no matched clean parent in packed input")
            parent = parent_rows[base]
            for key in ("speaker_id", "y_binary", "y_severity", "severity_kind"):
                if rows[key][parent] != rows[key][row]:
                    raise ValueError(f"augmented row {row}: parent inheritance mismatch for {key}")
            label_checks["parent_inheritance"] += 1
            pairs.append((parent, row))
        if pairs:
            paired = np.asarray(pairs)
            clean = rows["X"][paired[:, 0], 14:].astype(np.float64)
            augmented = rows["X"][paired[:, 1], 14:].astype(np.float64)
            paired_shift[variant] = {
                "embedding384": _pair_summary(clean, augmented),
                "F2_embedding_mean192": _pair_summary(clean[:, :192], augmented[:, :192]),
            }

    clean_rows = {key: value[indices] for key, value in rows.items()}
    additions = {
        "origin_variant": variants[indices],
        "source_row_index": np.asarray(source_indices, dtype=np.int64)[indices],
        "packed_source_row_index": indices.astype(np.int64),
    }
    for key, value in additions.items():
        if key in clean_rows and not np.array_equal(clean_rows[key], value):
            raise ValueError(f"existing row metadata {key!r} conflicts with verified provenance")
        clean_rows.setdefault(key, value)
    counts = {key: int(value) for key, value in zip(*np.unique(variants, return_counts=True))}
    return clean_rows, {
        "matching_method": "exact little-endian float32 bytes of all 384 embedding columns",
        "producer_access": "speaker_id inspected first; only training-speaker feature/label rows decoded",
        "producer_counts": producer_counts,
        "variant_counts": counts,
        "unmatched_rows": 0,
        "embedding_collisions": 0,
        "acoustic_mismatches": 0,
        "speaker_mismatches": 0,
        "label_checks": label_checks,
        "clean_packed_row_indices": indices.tolist(),
        "clean_producer_row_indices": additions["source_row_index"].tolist(),
        "clean_index_sha256_int64_le": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
        "clean_row_count": len(indices),
        "clean_speaker_count": len(np.unique(clean_rows["speaker_id"])),
        "previously_stated_clean_count": 216,
        "clean_count_correction": {
            "previously_stated": 216,
            "verified": len(indices),
            "reason": "Use all exactly matched clean rows; no arbitrary exclusion to force the previously stated count.",
        },
        "paired_embedding_shift": paired_shift,
    }


def prepare_clean_training(source, clean_archive, augmented_archive, lock_path, destination, report_path):
    source, clean_archive, augmented_archive, lock_path = [
        Path(path).resolve() for path in (source, clean_archive, augmented_archive, lock_path)]
    destination, report_path = Path(destination).absolute(), Path(report_path).absolute()
    paths = [source, clean_archive, augmented_archive, lock_path, destination.resolve(), report_path.resolve()]
    if len(set(paths)) != len(paths):
        raise ValueError("input, producer, lock, output, and report paths must all be distinct")
    if destination.suffix != ".npz" or report_path.suffix != ".json":
        raise ValueError("output must use .npz and report must use .json")
    for path in (destination, report_path):
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
    lock_bytes = lock_path.read_bytes()
    locked = {line.strip() for line in lock_bytes.decode("utf-8").splitlines() if line.strip()}
    if not locked:
        raise ValueError("lock list must not be empty")
    inputs = {"packed_train": source, "clean_producer": clean_archive,
              "augmented_producer": augmented_archive, "lock": lock_path}
    signatures = {key: P._source_signature(path) for key, path in inputs.items()}
    # This guard completes before either producer's features or labels load.
    rows = P._read_training_rows(source, locked, D.FEATURE_LAYOUT)
    D.validate_feature_layout(rows, required=True)
    D.clip_labels(rows)
    D.acoustic_availability(rows)
    clean_rows, provenance = recover_clean_rows(rows, clean_archive, augmented_archive)
    hashes = {key: P._sha256(path) for key, path in inputs.items()}
    if hashes["lock"] != hashlib.sha256(lock_bytes).hexdigest():
        raise RuntimeError("lock changed during preparation")
    provenance.update({
        "schema_version": 1,
        "transform": "verified_clean_row_subset_only",
        "branch": "A1",
        "input_paths": {key: str(value) for key, value in inputs.items()},
        "input_sha256": hashes,
        "output_path": str(destination),
        "feature_layout": D.FEATURE_LAYOUT,
        "input_row_count": len(rows["X"]),
        "input_speaker_count": len(np.unique(rows["speaker_id"])),
        "preserved_arrays": sorted(rows),
        "preserved_input_dtypes": {key: str(value.dtype) for key, value in rows.items()},
        "test_npz_read": False,
        "nontraining_producer_feature_or_label_rows_decoded": False,
        "preprocessing_fit": False,
        "model_fit": False,
        "inputs_unchanged": True,
    })
    temporary_npz = temporary_report = None
    published = []
    try:
        temporary_npz = P._stage_file(destination, lambda handle: np.savez_compressed(handle, **clean_rows))
        provenance["output_sha256"] = P._sha256(temporary_npz)
        payload = (json.dumps(provenance, indent=2, allow_nan=False) + "\n").encode("utf-8")
        temporary_report = P._stage_file(report_path, lambda handle: handle.write(payload))
        for key, path in inputs.items():
            if P._source_signature(path) != signatures[key] or P._sha256(path) != hashes[key]:
                raise RuntimeError(f"{key} changed during preparation; no output published")
        os.link(temporary_npz, destination)
        published.append(destination)
        os.link(temporary_report, report_path)
        published.append(report_path)
    except BaseException:
        for path in published:
            path.unlink()
        raise
    finally:
        for path in (temporary_npz, temporary_report):
            if path is not None:
                path.unlink(missing_ok=True)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="packed acoustic-first training NPZ")
    parser.add_argument("--clean-archive", type=Path, required=True)
    parser.add_argument("--augmented-archive", type=Path, required=True)
    parser.add_argument("--lock-test", type=Path, required=True, help="speaker exclusion IDs only")
    parser.add_argument("--out", type=Path, required=True, help="fresh clean-only .npz")
    parser.add_argument("--report", type=Path, required=True, help="fresh provenance .json")
    args = parser.parse_args()
    result = prepare_clean_training(args.data, args.clean_archive, args.augmented_archive,
                                    args.lock_test, args.out, args.report)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
