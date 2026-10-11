"""Prove the column layout of a feature NPZ before anything trains on it.

Why this exists
---------------
`features.select_features` validates only the *width* of X (398) and then
slices by position: F2 is `X[:, :206]`. Position is the entire contract, and
398 columns do not prove their own order. The two layouts in circulation are

    acoustic14_embedding_mean192_embedding_std192   <- what the pipeline assumes
    embedding_mean192_embedding_std192_acoustic14   <- the legacy producer order

If a legacy-order file is trained with --feature-set F2, the 206 selected
columns are 192 embedding-mean columns plus the first 14 embedding-*std*
columns: every acoustic feature is silently dropped and replaced by noise
columns. Nothing in the pipeline raises. The dev metrics look plausible. The
pre-registered primary comparison (ridge+F2 vs v1) becomes meaningless.

What this proves, and how
-------------------------
The checks are structural invariants of the data itself, not guesses:

  std block      A block of 192 per-dimension standard deviations is
                 non-negative in every entry. A 192-wide block containing a
                 negative value cannot be the std block. This alone separates
                 the two candidate layouts.
  mean block     ECAPA embedding means are centred near zero and roughly
                 symmetric, with both signs well represented.
  acoustic block The 14 acoustic features are physical quantities on
                 incompatible scales (dB, Hz, dimensionless ratios). Their
                 spread covers orders of magnitude and leaves the [-1, 1] box
                 that both embedding blocks live in.

A declared `feature_layout` array is checked for agreement with the evidence
rather than trusted: a wrong declaration is worse than none.

Mandatory gate (not optional, not advisory)
-------------------------------------------
`assert_canonical_input(data, label=...)` is the gate. Every entry point that
reads an NPZ must call it as the FIRST thing it does with the loaded arrays,
before any selection, scaling, fitting, prediction or reporting. It raises
`LayoutError` unless the file BOTH declares the canonical layout per row and
proves it structurally with the full score. Current call sites:

    pipeline.py          training input, before the processor is fitted
    evaluate.py          locked test input, before any feature is selected
    verify_sentinels.py  training input, before the processor is fitted
    environment_check.py both checked-in inputs

A declaration alone is a claim; the structural score is the proof. Neither is
sufficient, so the gate requires both and has no bypass flag: a file that
cannot prove its column order has no safe use, and an opt-out would only move
the failure somewhere quieter. `permute_locked_test_layout.py` uses the same
gate in the other direction, asserting the legacy layout of its input before it
permutes anything.

Sentinels
---------
The acoustic block may carry sentinel values standing in for "not measurable"
(an unvoiced clip has no jitter). Fed to a scaler as real numbers they poison
the fit. This tool reports sentinel candidates it finds, and verifies them
against a producer declaration when one is supplied with --sentinels.

Test-set handling
-----------------
Structural checks read `X` only. No label, and no row of a locked-test file,
is used for any model or hyperparameter decision; this tool makes no decision
beyond accept/reject of the file's column order. Value-level checks are
restricted to the training file.

Usage
-----
    python verify_layout.py --data ../data/pvqd_train.npz
    python verify_layout.py --data ../data/pvqd_train.npz ../data/pvqd_test.npz \
        --sentinels ../data/sentinels.json --report ../reports/layout_check.json

Exit status is 0 only when every file provably carries the layout the pipeline
assumes, with no unresolved sentinel candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from . import data as D
except ImportError:
    import data as D

N_ACOUSTIC, N_EMB = D.N_ACOUSTIC, D.N_EMB
ACOUSTIC_FIRST = D.FEATURE_LAYOUT
EMBEDDING_FIRST = "embedding_mean192_embedding_std192_acoustic14"

# Full marks from score_layout: std block non-negative (2) + mean block
# straddling zero (1) + acoustic block outside the unit box (2).
MAX_LAYOUT_SCORE = 5

# Written into a bundle manifest by pipeline.py only after the gate has passed
# on its training input, and required by evaluate.py before the locked test set
# is opened. An older bundle has no such attestation and cannot acquire one.
LAYOUT_PROVENANCE = "declared_in_npz_and_structurally_verified"


class LayoutError(ValueError):
    """The column layout of a feature matrix is not proven to be the expected one.

    A ValueError subclass: callers that already treat a bad data contract as a
    ValueError keep working, and callers that want to name this failure can.
    """

# Column order of the 14 acoustic features as the deployed scorer produces them
# (pitchee_voice_health_features_t / voice_health.cpp result.acoustic[0..13]),
# with the range each one can physically occupy. Used to check that an
# identified acoustic block is in the expected order, not merely present.
ACOUSTIC_SPEC = (
    ("hnr_db", -60.0, 60.0),
    ("cpps_db", 60.0, 140.0),        # emitted as 100.0 + cpps
    ("jitter_local", 0.0, 2.0),
    ("jitter_rap", 0.0, 2.0),
    ("jitter_ppq5", 0.0, 2.0),
    ("shimmer_local", 0.0, 2.0),
    ("shimmer_apq3", 0.0, 2.0),
    ("shimmer_apq5", 0.0, 2.0),
    ("f0_median", 0.0, 600.0),
    ("f0_iqr", 0.0, 600.0),
    ("f0_cv", 0.0, 10.0),
    ("voiced_fraction", 0.0, 1.0),
    ("intensity_median", -120.0, 120.0),
    ("intensity_iqr", 0.0, 120.0),
)

LAYOUT_BLOCKS = {
    # layout -> (acoustic slice, embedding-mean slice, embedding-std slice)
    ACOUSTIC_FIRST: ((0, N_ACOUSTIC),
                     (N_ACOUSTIC, N_ACOUSTIC + N_EMB),
                     (N_ACOUSTIC + N_EMB, N_ACOUSTIC + 2 * N_EMB)),
    EMBEDDING_FIRST: ((2 * N_EMB, 2 * N_EMB + N_ACOUSTIC),
                      (0, N_EMB),
                      (N_EMB, 2 * N_EMB)),
}


def block(X, bounds):
    start, end = bounds
    return X[:, start:end]


def finite(values):
    return values[np.isfinite(values)]


def describe(values):
    present = finite(values)
    if not present.size:
        return {"n_finite": 0}
    return {
        "n_finite": int(present.size),
        "n_nonfinite": int(values.size - present.size),
        "min": float(present.min()),
        "max": float(present.max()),
        "mean": float(present.mean()),
        "fraction_negative": float((present < 0).mean()),
    }


def score_layout(X, layout):
    """Collect the evidence for one candidate layout. Higher score is better."""
    acoustic_b, mean_b, std_b = LAYOUT_BLOCKS[layout]
    acoustic, means, stds = block(X, acoustic_b), block(X, mean_b), block(X, std_b)
    std_present, mean_present = finite(stds), finite(means)

    std_negative = int((std_present < 0).sum())
    std_nonnegative = std_negative == 0
    # Embedding means straddle zero; a block that is all one sign is not one.
    mean_symmetric = bool(
        mean_present.size
        and 0.2 <= float((mean_present < 0).mean()) <= 0.8
        and abs(float(mean_present.mean())) < 0.5
    )
    # The acoustic block holds physical quantities, so it must leave the unit
    # box that both embedding blocks occupy.
    acoustic_present = finite(acoustic)
    acoustic_wide = bool(
        acoustic_present.size
        and float(np.abs(acoustic_present).max()) > 2.0
    )
    in_range = []
    for index, (name, low, high) in enumerate(ACOUSTIC_SPEC):
        column = finite(acoustic[:, index])
        ok = bool(column.size and column.min() >= low and column.max() <= high)
        in_range.append({"index": index, "name": name, "within_physical_range": ok,
                         **describe(acoustic[:, index])})
    checks = {
        "std_block_nonnegative": std_nonnegative,
        "std_block_negative_entries": std_negative,
        "mean_block_straddles_zero": mean_symmetric,
        "acoustic_block_leaves_unit_box": acoustic_wide,
        "acoustic_columns_in_physical_range":
            sum(1 for row in in_range if row["within_physical_range"]),
    }
    score = (
        (2 if std_nonnegative else 0)
        + (1 if mean_symmetric else 0)
        + (2 if acoustic_wide else 0)
    )
    return {
        "layout": layout,
        "score": score,
        "checks": checks,
        "blocks": {
            "acoustic": {"slice": list(acoustic_b), **describe(acoustic)},
            "embedding_mean": {"slice": list(mean_b), **describe(means)},
            "embedding_std": {"slice": list(std_b), **describe(stds)},
        },
        "acoustic_columns": in_range,
    }


def layout_evidence(X):
    """Score every candidate layout for X, strongest evidence first."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != D.N_RAW_FEATURES:
        raise LayoutError(
            f"a column layout can only be verified on the fixed {D.N_RAW_FEATURES}-column "
            f"[{N_ACOUSTIC} acoustic|{N_EMB} emb-mean|{N_EMB} emb-std] matrix, got shape {X.shape}")
    evidence = [score_layout(X, layout) for layout in (ACOUSTIC_FIRST, EMBEDDING_FIRST)]
    evidence.sort(key=lambda row: -row["score"])
    return evidence


def assert_layout(X, expected=ACOUSTIC_FIRST, *, label=None, min_score=MAX_LAYOUT_SCORE):
    """Raise unless the structural evidence proves X is in the `expected` layout.

    Proof means the full score and strict dominance over the alternative: the
    std block has no negative entry, the mean block straddles zero, and the
    acoustic block leaves the unit box the two embedding blocks live in. Each
    check pins one block, so a file that misses any of them has not established
    its column order and must not be sliced by position.
    """
    if expected not in LAYOUT_BLOCKS:
        raise ValueError(f"unknown layout: {expected!r}")
    evidence = layout_evidence(X)
    best, runner_up = evidence[0], evidence[1]
    where = f"{label}: " if label else ""
    checks = {row["layout"]: row["checks"] for row in evidence}
    if best["score"] <= runner_up["score"]:
        raise LayoutError(
            f"{where}the structural evidence does not single out a column layout "
            f"({best['layout']}={best['score']}, {runner_up['layout']}={runner_up['score']}). "
            f"Nothing may slice this file by position until the feature producer "
            f"declares and the evidence confirms its column order. Checks: {checks}")
    if best["layout"] != expected:
        raise LayoutError(
            f"{where}this file is in {best['layout']!r} ({best['score']}/{MAX_LAYOUT_SCORE}), "
            f"not the expected {expected!r} ({runner_up['score']}/{MAX_LAYOUT_SCORE}). "
            f"Every feature-set gate checks only the {D.N_RAW_FEATURES}-column width and then "
            f"slices by position, so using it as it is would silently select the wrong "
            f"columns: F2 would take 192 embedding-mean columns plus the first 14 "
            f"embedding-std columns and drop every acoustic feature. Permute it into "
            f"{ACOUSTIC_FIRST!r} first (prepare_v2_data.py for training files, "
            f"permute_locked_test_layout.py for the locked test file). Checks: {checks}")
    if best["score"] < min_score:
        raise LayoutError(
            f"{where}the evidence for {expected!r} is incomplete "
            f"({best['score']}/{MAX_LAYOUT_SCORE}, required {min_score}): {best['checks']}. "
            f"A partially supported layout is not a proven layout; resolve it with the "
            f"feature producer before anything reads these columns by position.")
    return best


def assert_canonical_input(data, *, label=None):
    """THE MANDATORY GATE for every NPZ read. Call it before anything else.

    `data` is a loaded row dict (``data.load_npz`` output or equivalent). The
    file must BOTH declare the canonical layout for every row and prove it
    structurally. A declaration is the producer's claim; the score is the
    evidence. There is no bypass: see this module's docstring.
    """
    D.validate_feature_layout(data, required=True)
    return assert_layout(data["X"], ACOUSTIC_FIRST, label=label)


def read_matrix(path):
    """Read X alone from an NPZ, for callers that only need the layout proof."""
    with np.load(path, allow_pickle=False) as archive:
        if "X" not in archive.files:
            raise LayoutError(f"{path}: no X array; this is not a feature NPZ")
        return np.asarray(archive["X"], dtype=np.float64)


def sentinel_candidates(acoustic, min_repeats=3):
    """Values repeated across rows in a column that cannot be real measurements.

    A real jitter or HNR estimate is a continuous quantity: the same value
    recurring in many rows of one column is a placeholder, not a measurement.
    Zero is reported separately because it is both a plausible value and a
    common filler.
    """
    found = []
    for index in range(acoustic.shape[1]):
        column = acoustic[:, index]
        present = finite(column)
        if not present.size:
            continue
        values, counts = np.unique(present, return_counts=True)
        for value, count in zip(values, counts):
            if count < min_repeats:
                continue
            extreme = abs(float(value)) >= 100.0
            if not (extreme or float(value) == 0.0):
                continue
            found.append({
                "column": index,
                "name": ACOUSTIC_SPEC[index][0],
                "value": float(value),
                "rows": int(count),
                "fraction_of_rows": float(count / column.size),
                "kind": "extreme_repeated" if extreme else "zero_repeated",
            })
    found.sort(key=lambda row: (-row["rows"], row["column"]))
    return found



def within_speaker_regime(acoustic, speakers):
    """Flag acoustic columns whose value moves more within a speaker than across.

    Augmented copies share their original's speaker_id, so a column where the
    typical within-speaker spread exceeds the whole population's spread is not
    measuring a speaker property any more: the augmentation has moved it into a
    different regime. A model fitted on such a column learns the augmentation,
    and the clean test rows are then out of distribution.
    """
    rows = []
    speakers = np.asarray(speakers, dtype=str)
    unique, inverse = np.unique(speakers, return_inverse=True)
    groups = [np.where(inverse == index)[0] for index in range(len(unique))]
    multi = [group for group in groups if len(group) > 1]
    if not multi:
        return rows
    for column in range(acoustic.shape[1]):
        values = acoustic[:, column]
        present = finite(values)
        if present.size < 2:
            continue
        population = float(np.percentile(present, 75) - np.percentile(present, 25))
        spreads = []
        for group in multi:
            group_values = finite(values[group])
            if group_values.size > 1:
                spreads.append(float(group_values.max() - group_values.min()))
        if not spreads or population <= 0:
            continue
        median_spread = float(np.median(spreads))
        rows.append({
            "column": column,
            "name": ACOUSTIC_SPEC[column][0],
            "median_within_speaker_range": median_spread,
            "population_iqr": population,
            "ratio": median_spread / population,
            "dominated_by_within_speaker_variation": bool(median_spread > population),
        })
    return rows


def first_row_per_speaker(speakers):
    """Index of the first row of each speaker group, in file order.

    Augmented copies share their original's speaker_id, so this subset is one
    row per speaker. Which row is the unaugmented original is not recorded in
    the NPZ; the caller must corroborate the choice against the data before
    relying on it (see augmentation_signature).
    """
    speakers = np.asarray(speakers, dtype=str)
    seen, picked = set(), []
    for index, speaker in enumerate(speakers):
        if speaker not in seen:
            seen.add(speaker)
            picked.append(index)
    return np.asarray(picked, dtype=int)


def augmentation_signature(acoustic, speakers):
    """Per-row-position summary for speakers with equally sized groups.

    If the rows of a speaker group were produced by different processing, the
    position summaries separate. That is the evidence for treating position 0
    as the unaugmented original rather than assuming it.
    """
    speakers = np.asarray(speakers, dtype=str)
    unique, inverse = np.unique(speakers, return_inverse=True)
    groups = [np.where(inverse == index)[0] for index in range(len(unique))]
    sizes = [len(group) for group in groups]
    if not sizes:
        return []
    common = max(set(sizes), key=sizes.count)
    selected = [group for group in groups if len(group) == common]
    if common < 2 or len(selected) < 5:
        return []
    rows = []
    for position in range(common):
        indexes = [group[position] for group in selected]
        values = acoustic[indexes, :]
        rows.append({
            "row_position": position,
            "n_speakers": len(indexes),
            "columns": [
                {"column": column, "name": ACOUSTIC_SPEC[column][0],
                 **describe(values[:, column])}
                for column in range(values.shape[1])
            ],
        })
    return rows


def compare_acoustic_regimes(reference, candidate, layout, reference_rows=None):
    """Compare the acoustic block of two files column by column.

    A one-shot evaluation is only meaningful if the test rows were measured on
    the same scale as the training rows. An offset or a scale change in a
    column invalidates it no matter how the model was fitted, because the
    scaler was fitted on the other regime.
    """
    acoustic_b = LAYOUT_BLOCKS[layout][0]
    left, right = block(reference, acoustic_b), block(candidate, acoustic_b)
    if reference_rows is not None:
        left = left[reference_rows, :]
    rows = []
    for column in range(left.shape[1]):
        a, b = finite(left[:, column]), finite(right[:, column])
        if a.size < 2 or b.size < 2:
            continue
        pooled = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0))
        shift = abs(float(a.mean() - b.mean()))
        standardised = shift / pooled if pooled > 0 else float("inf")
        a_iqr = float(np.percentile(a, 75) - np.percentile(a, 25))
        b_iqr = float(np.percentile(b, 75) - np.percentile(b, 25))
        scale = (max(a_iqr, b_iqr) / min(a_iqr, b_iqr)
                 if min(a_iqr, b_iqr) > 0 else float("inf"))
        a_negative, b_negative = float((a < 0).mean()), float((b < 0).mean())
        rows.append({
            "column": column,
            "name": ACOUSTIC_SPEC[column][0],
            "mean_reference": float(a.mean()),
            "mean_candidate": float(b.mean()),
            "standardised_mean_shift": standardised,
            "iqr_ratio": scale,
            "sign_flip": bool(abs(a_negative - b_negative) > 0.5),
            "regime_mismatch": bool(standardised > 1.0 or scale > 3.0
                                    or abs(a_negative - b_negative) > 0.5),
        })
    return rows


def verify_file(path, sentinels=None, locked=None):
    with np.load(path, allow_pickle=True) as archive:
        files = set(archive.files)
        missing = set(D.REQUIRED_ARRAYS) - files
        if missing:
            return {"path": str(path), "verdict": "FAIL",
                    "errors": [f"missing required arrays: {sorted(missing)}"]}
        X = np.asarray(archive["X"], dtype=np.float64)
        declared = (sorted(set(np.asarray(archive["feature_layout"], dtype=str).tolist()))
                    if "feature_layout" in files else None)
        speakers = np.asarray(archive["speaker_id"], dtype=str)
        n_speakers = int(len(set(map(str, speakers))))

    result = {"path": str(path), "n_clips": int(X.shape[0]), "n_speakers": n_speakers,
              "n_columns": int(X.shape[1]), "dtype": str(X.dtype),
              "declared_feature_layout": declared, "errors": [], "warnings": []}
    n_locked = (len(set(map(str, speakers)) & locked) if locked is not None else None)
    if n_locked is not None:
        result["n_locked_speakers_present"] = n_locked
        result["is_locked_test_file"] = bool(n_locked and n_locked == n_speakers)
    if X.ndim != 2 or X.shape[1] != D.N_RAW_FEATURES:
        result["verdict"] = "FAIL"
        result["errors"].append(
            f"X must have exactly {D.N_RAW_FEATURES} columns, got shape {X.shape}; "
            "no layout can be verified")
        return result

    evidence = [score_layout(X, layout) for layout in (ACOUSTIC_FIRST, EMBEDDING_FIRST)]
    evidence.sort(key=lambda row: -row["score"])
    best, runner_up = evidence[0], evidence[1]
    result["evidence"] = evidence
    conclusive = best["score"] > runner_up["score"] and best["score"] >= 4
    result["inferred_layout"] = best["layout"] if conclusive else None

    if not conclusive:
        result["errors"].append(
            "the structural evidence does not single out a layout "
            f"(scores: {best['layout']}={best['score']}, "
            f"{runner_up['layout']}={runner_up['score']}). Do not train on this "
            "file until the feature producer declares the column order.")
    elif best["layout"] != ACOUSTIC_FIRST:
        message = (
            f"this file is in {best['layout']!r}, but the pipeline's "
            f"--feature-set slices assume {ACOUSTIC_FIRST!r}. Used as it is, F2 "
            "selects 192 embedding-mean columns plus 14 embedding-STD columns "
            "and drops every acoustic feature, silently: only the 398-column "
            "width is checked anywhere.")
        if n_locked:
            message += (
                f" {n_locked} of this file's {n_speakers} speakers are locked "
                "test speakers, and prepare_v2_data.py rejects any file "
                "containing them by design (it has no test output path), so it "
                "cannot permute this file. Agree the route with the pipeline "
                "owner before any evaluation: a model fitted on permuted "
                "training rows and evaluated on these columns measures nothing, "
                "and the locked set is spent either way.")
        else:
            message += ("\n    python prepare_v2_data.py --data " + str(path)
                        + f" --source-layout {best['layout']}"
                        + " --lock-test ../test_speakers.txt --out <new file>")
        result["errors"].append(message)

    if declared is not None:
        if declared != [ACOUSTIC_FIRST]:
            result["errors"].append(
                f"declared feature_layout {declared} is not {ACOUSTIC_FIRST!r}")
        elif conclusive and best["layout"] != ACOUSTIC_FIRST:
            result["errors"].append(
                f"declared feature_layout {declared} contradicts the structural "
                f"evidence ({best['layout']!r}). A wrong declaration is worse "
                "than none: resolve with the feature producer before training.")
    else:
        result["warnings"].append(
            "no feature_layout array: column order is undeclared provenance. "
            "Phase 2 requires the producer to declare it per row "
            "(data.validate_feature_layout(required=True)).")

    # Sentinel scan on whichever block the evidence says is acoustic.
    acoustic = block(X, LAYOUT_BLOCKS[best["layout"]][0]) if conclusive else None
    if acoustic is not None:
        regimes = within_speaker_regime(acoustic, speakers)
        result["within_speaker_regime"] = regimes
        dominated = [row["name"] for row in regimes
                     if row["dominated_by_within_speaker_variation"]]
        if dominated:
            result["errors"].append(
                "acoustic columns whose typical within-speaker range exceeds the "
                f"whole population's IQR: {dominated}. Copies of one recording "
                "are supposed to share a speaker; here the augmentation moves "
                "these columns further than real speaker differences do, so a "
                "model fitted on them learns the augmentation and clean rows "
                "fall out of distribution. Resolve with the feature producer.")
    if acoustic is not None:
        candidates = sentinel_candidates(acoustic)
        result["sentinel_candidates"] = candidates
        extreme = [row for row in candidates if row["kind"] == "extreme_repeated"]
        if sentinels is None:
            if extreme:
                values = sorted({row["value"] for row in extreme})
                result["errors"].append(
                    f"acoustic block carries repeated extreme values {values}, which "
                    "are placeholders, not measurements. Fed to StandardScaler as "
                    "real numbers they move the mean and variance of their column. "
                    "Get the producer's sentinel declaration and re-run with "
                    "--sentinels before training.")
        else:
            declared_values = {float(v) for v in sentinels.get("values", [])}
            found_values = {row["value"] for row in extreme}
            undeclared = sorted(found_values - declared_values)
            unseen = sorted(declared_values - found_values)
            result["sentinels"] = {
                "declared": sorted(declared_values),
                "found_in_acoustic_block": sorted(found_values),
                "undeclared_found": undeclared,
                "declared_not_found": unseen,
            }
            if undeclared:
                result["errors"].append(
                    f"acoustic block carries undeclared extreme values {undeclared}; "
                    "the declaration does not cover what is in the file")
            if unseen:
                result["warnings"].append(
                    f"declared sentinels not present in this file: {unseen}")

    nonfinite = int((~np.isfinite(X)).sum())
    if nonfinite:
        result["warnings"].append(
            f"{nonfinite} non-finite entries in X; the loader only admits them "
            "for the acoustic block under --task-aware/--missing-indicators")

    result["verdict"] = "FAIL" if result["errors"] else "PASS"
    result["_matrix"] = X
    result["_speakers"] = speakers
    result["_layout"] = best["layout"] if conclusive else None
    if acoustic is not None:
        result["augmentation_signature"] = augmentation_signature(acoustic, speakers)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", required=True, type=Path,
                        help="NPZ files to verify")
    parser.add_argument("--lock-test", type=Path,
                        help="locked test speaker list; lets the tool say which "
                             "remediation route applies to each file")
    parser.add_argument("--sentinels", type=Path,
                        help='producer declaration, JSON: {"values": [-200.0, ...]}')
    parser.add_argument("--report", type=Path, help="write the full findings as JSON")
    args = parser.parse_args()

    sentinels = json.loads(args.sentinels.read_text()) if args.sentinels else None
    locked = ({line.strip() for line in args.lock_test.read_text().splitlines() if line.strip()}
              if args.lock_test else None)
    results = [verify_file(path, sentinels, locked) for path in args.data]

    # With more than one file, the first is the reference (the training file):
    # the others must have been measured on the same scale or a one-shot
    # evaluation compares a model against features it was never fitted on.
    if len(results) > 1:
        reference = results[0]
        for candidate in results[1:]:
            if reference.get("_layout") is None or reference["_layout"] != candidate.get("_layout"):
                candidate["errors"].append(
                    f"{Path(candidate['path']).name} and "
                    f"{Path(reference['path']).name} do not have the same provable "
                    "column layout, so one of them must be permuted before either "
                    "is used. Nothing downstream catches this: pipeline.py and "
                    "evaluate.py both slice by position after checking only the "
                    "398-column width, and the lock hash and manifest checks pass "
                    "regardless. Training on one layout and evaluating on the "
                    "other spends the one-shot locked test set on a number that "
                    "measures nothing. Note that prepare_v2_data.py refuses files "
                    "containing locked speakers by design, so it cannot permute "
                    "the test file: agree the route with the pipeline owner.")
                candidate["verdict"] = "FAIL"
                continue
            layout = reference["_layout"]
            full = compare_acoustic_regimes(reference["_matrix"], candidate["_matrix"], layout)
            singles = first_row_per_speaker(reference["_speakers"])
            has_groups = len(singles) < len(reference["_speakers"])
            subset = (compare_acoustic_regimes(reference["_matrix"], candidate["_matrix"],
                                               layout, reference_rows=singles)
                      if has_groups else None)
            candidate["acoustic_regime_vs_reference"] = {
                "reference": reference["path"],
                "all_reference_rows": full,
                "first_row_per_speaker": subset,
                "subset_note": (
                    "The reference file has several rows per speaker, so its "
                    "aggregate statistics mix processing variants. The subset "
                    "takes the first row of each speaker group; corroborate that "
                    "choice against augmentation_signature before relying on it."
                    if has_groups else None),
            }
            full_bad = [row["name"] for row in full if row["regime_mismatch"]]
            subset_bad = ([row["name"] for row in subset if row["regime_mismatch"]]
                          if subset is not None else full_bad)
            if subset_bad:
                candidate["errors"].append(
                    "acoustic columns differ in scale between "
                    f"{Path(reference['path']).name} and "
                    f"{Path(candidate['path']).name} even when the reference is "
                    f"reduced to one row per speaker: {subset_bad}. One row per "
                    "speaker carries no augmented duplicates, so this is not an "
                    "augmentation artefact: the two files were measured "
                    "differently. Stop and have the feature producer check the "
                    "extraction pipeline; a scaler fitted on one of these files "
                    "does not apply to the other.")
                candidate["verdict"] = "FAIL"
            elif full_bad:
                candidate["errors"].append(
                    "acoustic columns whose scale differs from "
                    f"{Path(candidate['path']).name}: {full_bad} -- but only when "
                    f"all rows of {Path(reference['path']).name} are pooled. "
                    "Reduced to one row per speaker the two files agree, so the "
                    "extraction is consistent and the augmented duplicates are "
                    "what sits in a different regime. Since the scaler is fitted "
                    "on all training rows, the augmented majority sets the mean "
                    "and scale that the clean rows are then standardised by. "
                    "Decide with the feature producer whether the augmentation "
                    "should preserve these columns, or whether they should be "
                    "fitted on unaugmented rows only.")
                candidate["verdict"] = "FAIL"

    for result in results:
        print(f"\n=== {result['path']} ===")
        print(f"  {result['n_clips']} clips / {result['n_speakers']} speakers, "
              f"{result['n_columns']} columns, dtype {result['dtype']}")
        if result.get("n_locked_speakers_present") is not None:
            print(f"  locked speakers : {result['n_locked_speakers_present']}"
                  f"{'  (this is the locked test file)' if result.get('is_locked_test_file') else ''}")
        print(f"  declared layout : {result['declared_feature_layout'] or '(none)'}")
        print(f"  inferred layout : {result.get('inferred_layout') or '(inconclusive)'}")
        for row in result.get("evidence", []):
            print(f"    [{row['score']}/5] {row['layout']}")
            for name, value in row["checks"].items():
                print(f"          {name}: {value}")
            for label, info in row["blocks"].items():
                if info.get("n_finite"):
                    print(f"          {label:<15} cols {info['slice'][0]:>3}:"
                          f"{info['slice'][1]:<3} min={info['min']:>10.4f} "
                          f"max={info['max']:>10.4f} neg={info['fraction_negative']:.3f}")
        for row in result.get("within_speaker_regime", []):
            if row["dominated_by_within_speaker_variation"]:
                print(f"    regime! col {row['column']:>2} ({row['name']}): "
                      f"within-speaker range {row['median_within_speaker_range']:.3f} "
                      f"> population IQR {row['population_iqr']:.3f} "
                      f"(x{row['ratio']:.1f})")
        comparison = result.get("acoustic_regime_vs_reference")
        if comparison:
            for label, key in (("all rows", "all_reference_rows"),
                               ("one row per speaker", "first_row_per_speaker")):
                rows = comparison.get(key)
                if rows is None:
                    continue
                bad = [row for row in rows if row["regime_mismatch"]]
                print(f"    acoustic scale vs {Path(comparison['reference']).name} "
                      f"[{label}]: {len(bad)}/{len(rows)} columns mismatch")
                for row in bad:
                    print(f"      MISMATCH col {row['column']:>2} ({row['name']}): "
                          f"mean {row['mean_reference']:.3f} -> {row['mean_candidate']:.3f} "
                          f"(shift {row['standardised_mean_shift']:.2f} sd, "
                          f"iqr x{row['iqr_ratio']:.2f}"
                          f"{', sign flip' if row['sign_flip'] else ''})")
        for row in result.get("sentinel_candidates", [])[:10]:
            print(f"    sentinel? col {row['column']:>2} ({row['name']}): "
                  f"{row['value']} in {row['rows']} rows "
                  f"({row['fraction_of_rows']:.1%}) [{row['kind']}]")
        for warning in result["warnings"]:
            print(f"  WARN  {warning}")
        for error in result["errors"]:
            print(f"  ERROR {error}")
        print(f"  VERDICT: {result['verdict']}")

    for result in results:
        result.pop("_matrix", None)
        result.pop("_speakers", None)
        result.pop("_layout", None)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nreport written to {args.report}")

    failed = [r for r in results if r["verdict"] != "PASS"]
    if failed:
        print(f"\n{len(failed)} of {len(results)} file(s) FAILED. "
              "Do not run pipeline.py on them.")
        return 1
    print(f"\nall {len(results)} file(s) PASS: layout is {ACOUSTIC_FIRST}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
