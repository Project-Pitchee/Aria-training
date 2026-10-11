"""Data contract and speaker-grouped splitting for the vocal-health v2 pipeline.

NPZ CONTRACT (one file per corpus, or one combined file):
    X            (n_clips, n_features) float32/float64 -- model input features
    y_severity   (n_clips,) float32                   -- continuous severity,
                                                       corpus-native units
                                                       (e.g. CAPE-V 0-100 mm,
                                                       GRBAS 0-3)
    severity_kind (n_clips,) str                       -- 'cape_v' | 'grbas' | ...
    y_binary     (n_clips,) int                        -- 1=disordered, 0=healthy
    speaker_id   (n_clips,) str                        -- stable speaker key.
                                                       Augmented copies of one
                                                       recording MUST share the
                                                       speaker_id of the original.
    corpus       (n_clips,) str                        -- 'pvqd' | 'svd' | ...
    clip         (n_clips,) str, optional              -- vowel | phrase | unknown
                                                       ('clip_type' alias accepted)
    acoustic_available (n_clips, 14) bool, optional     -- whether each raw acoustic
                                                       feature was computable
    feature_layout (n_clips,) str, optional            -- exact FEATURE_LAYOUT value;
                                                       provenance, not a reordering hint

SPLIT RULES (non-negotiable):
    * All splits are by speaker_id. A speaker's clips (incl. augmentations)
      never land on both sides of a split.
    * The locked test set (paper's 58 PVQD speakers) is NEVER used for
      training, dev, scaler fitting, or any model/hyperparameter decision.
      pipeline.py hard-fails if a locked speaker appears in train/dev input.
"""

from __future__ import annotations

import numpy as np


N_ACOUSTIC, N_EMB = 14, 192
N_RAW_FEATURES = N_ACOUSTIC + 2 * N_EMB
FEATURE_LAYOUT = "acoustic14_embedding_mean192_embedding_std192"
REQUIRED_ARRAYS = ("X", "y_severity", "severity_kind", "y_binary", "speaker_id", "corpus")


def validate_feature_layout(data: dict, required: bool = False) -> bool:
    """Check declared column provenance, without guessing or reordering columns.

    Return True for an explicit matching declaration. Missing metadata is
    permitted only when ``required=False`` and returns False, indicating that
    the caller is relying on the legacy layout assumption.
    """
    if "feature_layout" not in data:
        if required:
            raise ValueError(
                "feature_layout metadata is required for Phase 2: have the feature producer verify "
                f"and declare {FEATURE_LAYOUT!r} per row; 398 columns alone do not prove their ordering")
        return False
    n = len(data["X"])
    declarations = np.asarray(data["feature_layout"], dtype=str)
    if declarations.shape != (n,):
        raise ValueError(f"feature_layout must have shape ({n},), got {declarations.shape}")
    if not np.all(declarations == FEATURE_LAYOUT):
        raise ValueError(
            f"feature_layout must be exactly {FEATURE_LAYOUT!r} for every row; "
            f"got {sorted(set(declarations))}. Refusing to infer or reorder feature columns")
    if np.ndim(data["X"]) != 2 or np.shape(data["X"])[1] != N_RAW_FEATURES:
        raise ValueError("declared feature_layout requires the fixed 398-column [14|192|192] matrix")
    return True


def clip_labels(data: dict) -> np.ndarray:
    """Read the optional task labels without inferring an unknown clip's task."""
    n = len(data["X"])
    arrays = []
    for key in ("clip", "clip_type"):
        if key not in data:
            continue
        values = np.asarray(data[key], dtype=str)
        if values.shape != (n,):
            raise ValueError(f"{key} must have shape ({n},), got {values.shape}")
        bad = sorted(set(values) - {"vowel", "phrase", "unknown"})
        if bad:
            raise ValueError(f"{key} has unsupported task labels {bad}; use vowel, phrase, or unknown")
        arrays.append(values)
    if len(arrays) == 2 and not np.array_equal(*arrays):
        raise ValueError("clip and clip_type disagree; supply one consistent task label per row")
    return arrays[0] if arrays else np.full(n, "unknown", dtype="<U7")


def acoustic_availability(data: dict) -> np.ndarray:
    """Return raw availability; a non-finite acoustic value is never available."""
    X = np.asarray(data["X"])
    finite = np.isfinite(X[:, :N_ACOUSTIC])
    if "acoustic_available" not in data:
        return finite
    values = np.asarray(data["acoustic_available"])
    if values.shape != (len(X), N_ACOUSTIC):
        raise ValueError(
            f"acoustic_available must have shape ({len(X)}, {N_ACOUSTIC}), got {values.shape}")
    if not np.all(np.isin(values, [False, True])):
        raise ValueError("acoustic_available must contain only binary 0/1 values")
    return values.astype(bool) & finite


def load_npz(path: str, *, allow_acoustic_nan: bool = False) -> dict:
    with np.load(path, allow_pickle=True) as d:
        missing = set(REQUIRED_ARRAYS) - set(d.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        out = {k: d[k] for k in d.files}
    if np.ndim(out["X"]) != 2:
        raise ValueError(f"{path}: X must be a two-dimensional feature matrix")
    n = len(out["X"])
    for k, v in out.items():
        if np.ndim(v) == 0 or len(v) != n:
            raise ValueError(f"{path}: {k} must be a row array with {n} rows, got shape {np.shape(v)}")
    out["X"] = np.asarray(out["X"], dtype=np.float64)
    out["y_severity"] = np.asarray(out["y_severity"], dtype=np.float64)
    out["y_binary"] = np.asarray(out["y_binary"], dtype=np.int64)
    out["speaker_id"] = np.asarray(out["speaker_id"], dtype=str)
    out["corpus"] = np.asarray(out["corpus"], dtype=str)
    out["severity_kind"] = np.asarray(out["severity_kind"], dtype=str)
    if validate_feature_layout(out):
        out["feature_layout"] = np.asarray(out["feature_layout"], dtype=str)
    if allow_acoustic_nan:
        if out["X"].shape[1] != N_RAW_FEATURES:
            raise ValueError(f"{path}: allowing acoustic NaN requires the fixed 398-column [14|192|192] layout")
        if np.isinf(out["X"]).any() or not np.isfinite(out["X"][:, N_ACOUSTIC:]).all():
            raise ValueError(f"{path}: infinity and non-finite embedding values are forbidden; only acoustic NaN is supported")
    elif not np.all(np.isfinite(out["X"])):
        raise ValueError(f"{path}: X contains non-finite values")
    if "clip" in out or "clip_type" in out:
        out["clip"] = clip_labels(out)
        out.pop("clip_type", None)
    if "acoustic_available" in out:
        out["acoustic_available"] = acoustic_availability(out)
    return out


def concat(datasets: list[dict]) -> dict:
    if not datasets:
        raise ValueError("concat needs at least one dataset")
    keys = set().union(*(d.keys() for d in datasets))
    out = {k: np.concatenate([d[k] for d in datasets]) for k in REQUIRED_ARRAYS}
    if keys & {"clip", "clip_type"}:
        out["clip"] = np.concatenate([clip_labels(d) for d in datasets])
    if "acoustic_available" in keys:
        out["acoustic_available"] = np.concatenate([acoustic_availability(d) for d in datasets])
    for key in sorted(keys - set(REQUIRED_ARRAYS) - {"clip", "clip_type", "acoustic_available"}):
        if any(key not in d for d in datasets):
            raise ValueError(f"optional row array {key!r} is missing from a dataset; supply it explicitly")
        out[key] = np.concatenate([d[key] for d in datasets])
    return out


def speaker_grouped_split(data: dict, dev_frac: float = 0.15, seed: int = 7):
    rng = np.random.default_rng(seed)
    spk = data["speaker_id"]
    uniq, inv = np.unique(spk, return_inverse=True)
    strata: dict = {}
    for s in uniq:
        m = inv == np.where(uniq == s)[0][0]
        corp = str(data["corpus"][m][0])
        lab = int(np.round(data["y_binary"][m].mean()))
        strata.setdefault((corp, lab), []).append(s)
    dev_spk: set = set()
    for key, members in strata.items():
        members = list(members)
        rng.shuffle(members)
        k = max(1, int(round(len(members) * dev_frac))) if len(members) > 1 else 0
        k = min(k, len(members) - 1)
        dev_spk.update(members[:k])
    dev_mask = np.isin(spk, list(dev_spk))
    return ({k: v[~dev_mask] for k, v in data.items()},
            {k: v[dev_mask] for k, v in data.items()})


def assert_no_leak(train_spk: np.ndarray, dev_spk: np.ndarray, locked: set) -> None:
    bad_tr = sorted(set(map(str, train_spk)) & locked)
    bad_dv = sorted(set(map(str, dev_spk)) & locked)
    if bad_tr or bad_dv:
        raise SystemExit(
            "LOCKED TEST SPEAKERS IN TRAIN/DEV -- refusing to continue.\n"
            f"  in train: {bad_tr}\n  in dev: {bad_dv}")
    overlap = sorted(set(map(str, train_spk)) & set(map(str, dev_spk)))
    if overlap:
        raise SystemExit(f"SPEAKERS IN BOTH TRAIN AND DEV -- refusing to continue: {overlap}")


def read_locked_speakers(path: str) -> set:
    with open(path) as source:
        return {ln.strip() for ln in source if ln.strip()}
