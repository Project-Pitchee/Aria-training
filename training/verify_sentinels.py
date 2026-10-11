"""Verify the confirmed sentinel policy on TRAIN-ONLY inputs before model fits.

Checks exclusion IDs before reading feature/label arrays, reproduces the
speaker-grouped train/dev split, and fits only the feature processor on train.
There is no evaluation input or model fitting. The resulting report can be
compared with the manifest of a subsequent pipeline.py run using the same seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

try:
    from . import data as D, features as F, prepare_v2_data as P, verify_layout as V
    from .acoustic_missingness import ACOUSTIC_FEATURES, sentinel_mask
except ImportError:
    import data as D
    import features as F
    import prepare_v2_data as P
    import verify_layout as V
    from acoustic_missingness import ACOUSTIC_FEATURES, sentinel_mask


def verify_training_sentinels(data_path, lock_path, dev_frac=0.15, seed=7):
    if not 0 < dev_frac < 1:
        raise ValueError("dev_frac must lie strictly between 0 and 1")
    lock_bytes = Path(lock_path).read_bytes()
    locked = {line.strip() for line in lock_bytes.decode("utf-8").splitlines() if line.strip()}
    if not locked:
        raise ValueError("lock list must not be empty")
    lock_hash = hashlib.sha256(lock_bytes).hexdigest()
    source_signature = P._source_signature(data_path)
    rows = P._read_training_rows(Path(data_path), locked, D.FEATURE_LAYOUT)
    # THE GATE, before the processor is fitted: declared canonical per row and
    # structurally proven. The sentinel rules below address acoustic columns by
    # index (HNR at 0, F0 median at 8), so a file whose column order was never
    # proven would have the rules applied to embedding columns instead.
    V.assert_canonical_input(rows, label=str(data_path))
    source_hash = P._sha256(data_path)
    if P._source_signature(data_path) != source_signature:
        raise RuntimeError("training input changed while reading; no verification result produced")
    train, dev = D.speaker_grouped_split(rows, dev_frac, seed)
    D.assert_no_leak(train["speaker_id"], dev["speaker_id"], locked)
    if not len(dev["X"]):
        raise ValueError("verification needs nonempty train and dev splits")
    processor = F.FeatureProcessor(
        "F2", sentinel_policy="core_v1", imputation_strategy="median",
        missing_indicators=True).fit(train)
    Xt = processor.transform(train)
    initial_medians = processor.imputer.statistics_.copy()
    initial_mean = processor.scaler.mean_.copy()
    Xd = processor.transform(dev)
    np.testing.assert_array_equal(initial_medians, processor.imputer.statistics_)
    np.testing.assert_array_equal(initial_mean, processor.scaler.mean_)
    acoustic = np.asarray(train["X"][:, :14], dtype=float)
    trusted = D.acoustic_availability(train) & ~sentinel_mask(train["X"], "core_v1")
    observed = np.where(trusted, acoustic, np.nan)
    np.testing.assert_allclose(processor.imputer.statistics_, np.nanmedian(observed, axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(processor.scaler.mean_[:14], np.nanmean(observed, axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(processor.scaler.var_[:14], np.nanvar(observed, axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(Xt[:, 206:], trusted.astype(float))
    dev_trusted = D.acoustic_availability(dev) & ~sentinel_mask(dev["X"], "core_v1")
    np.testing.assert_array_equal(Xd[:, 206:], dev_trusted.astype(float))
    with tempfile.TemporaryDirectory(prefix="vocal-sentinel-verify-") as temporary:
        processor.save(temporary)
        loaded = F.FeatureProcessor.load(temporary)
        np.testing.assert_array_equal(loaded.transform(train), Xt)
        np.testing.assert_array_equal(loaded.transform(dev), Xd)
    if (P._sha256(data_path) != source_hash or P._sha256(lock_path) != lock_hash
            or P._source_signature(data_path) != source_signature):
        raise RuntimeError("training input or lock changed during verification")

    counts = {}
    for name, split in [("all_training_input", rows), ("train", train), ("dev", dev)]:
        X = np.asarray(split["X"])
        mask = sentinel_mask(X, "core_v1")
        counts[name] = {
            "n_clips": len(X), "n_speakers": len(np.unique(split["speaker_id"])),
            "hnr_minus_200": int(mask[:, 0].sum()),
            "f0_median_zero": int(mask[:, 8].sum()),
            "hnr_and_f0_failure": int((mask[:, 0] & mask[:, 8]).sum()),
            "valid_median_with_zero_iqr_cv": int(((X[:, 8] > 0) & (X[:, 9] == 0) & (X[:, 10] == 0)).sum()),
        }
    return {
        "status": "passed", "seed": seed, "dev_frac": dev_frac,
        "input_sha256": source_hash, "lock_test_sha256": lock_hash,
        "test_features_or_labels_read": False, "model_fitting_performed": False,
        "n_features_out": processor.n_features_out,
        "counts": counts,
        "checks": {"train_only_medians": True, "train_only_observed_scaler": True,
                   "pre_imputation_availability_flags": True, "saved_processor_round_trip": True,
                   "source_and_lock_unchanged": True},
        "training_statistics": [
            {"index": i, "feature": name, "n_available": int(trusted[:, i].sum()),
             "raw_mean_including_sentinels": float(np.nanmean(acoustic[:, i])),
             "observed_mean_excluding_unavailable": float(processor.scaler.mean_[i]),
             "impute_median": float(processor.imputer.statistics_[i])}
            for i, name in enumerate(ACOUSTIC_FEATURES)
        ],
        "feature_processing": processor.metadata(),
        "next_step": "Claude: verify passed, then rerun section 2.4 on this same input with --acoustic-sentinels core_v1 --impute median --missing-indicators; use a fresh output bundle. No locked evaluation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--lock-test", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dev-frac", type=float, default=0.15)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("verification report must be a new file")
    report = verify_training_sentinels(args.data, args.lock_test, args.dev_frac, args.seed)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
