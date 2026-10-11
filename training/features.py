"""Train-fitted feature processing for the fixed v1 [14 | 192 | 192] layout.

Task suppression and missing-value imputation are separate operations. Known
untrusted phrase acoustics are zeroed *after* standardization, so their linear
contribution is exactly zero. Real acoustic NaNs require an explicit imputation
choice. The opt-in core_v1 sentinel policy and train-fitted median strategy are
persisted with the scaler; default processing keeps the Phase 1 numeric path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    from . import data as D
    from .acoustic_missingness import (AcousticMedianImputer, policy_metadata,
                                       sentinel_mask, validate_sentinel_policy)
except ImportError:
    import data as D
    from acoustic_missingness import (AcousticMedianImputer, policy_metadata,
                                      sentinel_mask, validate_sentinel_policy)

N_ACOUSTIC, N_EMB = D.N_ACOUSTIC, D.N_EMB
N_RAW_FEATURES = D.N_RAW_FEATURES
FEATURE_WIDTHS = {"F1": N_RAW_FEATURES, "F2": N_ACOUSTIC + N_EMB, "F3": N_ACOUSTIC}


def select_features(X, feature_set: str) -> np.ndarray:
    """Select a feature set only after validating the complete raw layout."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != N_RAW_FEATURES:
        raise ValueError(
            f"feature-set {feature_set} requires the fixed 398-column [14 acoustic|192 emb-mean|192 emb-std] "
            f"layout, got shape {X.shape}; fix the input layout before continuing")
    if feature_set not in FEATURE_WIDTHS:
        raise ValueError(f"unknown feature set: {feature_set}")
    return X[:, :FEATURE_WIDTHS[feature_set]]


class FeatureProcessor:
    """Fit on train only, then apply identical processing to any held-out rows.

    ``imputation_strategy='median'`` selects the built-in persisted imputer.
    Alternatively, ``imputer`` is an external estimator with ``fit(acoustic_X)``
    and ``transform(acoustic_X)`` methods. It receives exactly 14 columns, with
    unavailable and task-untrusted train entries represented by NaN. Its fitted
    state is owned by the caller and is never deserialized implicitly by load().
    """

    def __init__(self, feature_set="F2", task_aware=False,
                 phrase_untrusted_columns=(), missing_indicators=False, imputer=None,
                 *, sentinel_policy="none", imputation_strategy=None):
        if feature_set not in FEATURE_WIDTHS:
            raise ValueError(f"unknown feature set: {feature_set}")
        columns = tuple(phrase_untrusted_columns)
        if any(isinstance(c, (bool, np.bool_)) or not isinstance(c, (int, np.integer))
               or c < 0 or c >= N_ACOUSTIC for c in columns):
            raise ValueError("phrase_untrusted_columns must be explicit zero-based acoustic indexes in [0, 13]")
        if len(set(columns)) != len(columns):
            raise ValueError("phrase_untrusted_columns must not contain duplicate indexes")
        if task_aware and not columns:
            raise ValueError("task-aware processing requires an explicit phrase_untrusted_columns list from the feature documentation")
        validate_sentinel_policy(sentinel_policy)
        if imputation_strategy not in (None, "median"):
            raise ValueError("imputation_strategy must be None or 'median'")
        if imputation_strategy is not None and imputer is not None:
            raise ValueError("imputation_strategy and an external imputer are mutually exclusive")
        if imputer is not None and not all(callable(getattr(imputer, name, None)) for name in ("fit", "transform")):
            raise TypeError("imputer must implement fit(acoustic_X) and transform(acoustic_X)")
        if sentinel_policy != "none" and imputation_strategy is None and imputer is None:
            raise ValueError("an acoustic sentinel policy requires imputation_strategy='median' or an explicit imputer")
        self.feature_set = feature_set
        self.task_aware = bool(task_aware)
        self.phrase_untrusted_columns = tuple(map(int, columns))
        self.missing_indicators = bool(missing_indicators)
        self.sentinel_policy = sentinel_policy
        self.imputation_strategy = imputation_strategy
        self.imputer = AcousticMedianImputer() if imputation_strategy == "median" else imputer
        self.scaler = None
        self.train_clip_counts_ = {}
        self.train_unavailable_counts_ = []
        self.train_sentinel_counts_ = {}
        self._unknown_seen = False

    @property
    def n_features_out(self) -> int:
        return FEATURE_WIDTHS[self.feature_set] + (N_ACOUSTIC if self.missing_indicators else 0)

    def _inputs(self, data):
        D.validate_feature_layout(data, required=self.sentinel_policy != "none")
        X = np.asarray(data["X"], dtype=np.float64)
        select_features(X, self.feature_set)
        if np.isinf(X).any() or not np.isfinite(X[:, N_ACOUSTIC:]).all():
            raise ValueError("infinity and non-finite embedding values are forbidden; only acoustic NaN is supported")
        # Sentinel detection precedes imputation and never mutates raw X or a
        # producer-supplied mask. Finite sentinel values are not observations.
        sentinels = sentinel_mask(X, self.sentinel_policy)
        available = D.acoustic_availability(data) & ~sentinels
        if np.isnan(X[:, :N_ACOUSTIC]).any() and self.imputer is None:
            raise ValueError(
                "acoustic X contains NaN but no approved imputer was supplied; select imputation_strategy='median' "
                "or inject an explicit train-fitted imputer. Task gating and missing indicators do not impute real NaNs")
        clip = D.clip_labels(data)
        if (self.task_aware or self.missing_indicators or self.sentinel_policy != "none") and not available.all() and self.imputer is None:
            raise ValueError(
                "raw acoustic availability marks values unavailable but no approved imputer was supplied; "
                "select imputation_strategy='median' or inject an explicit train-fitted imputer")
        suppressed = np.zeros((len(X), N_ACOUSTIC), dtype=bool)
        if self.task_aware:
            suppressed[np.ix_(clip == "phrase", self.phrase_untrusted_columns)] = True
        return X, clip, available, suppressed, sentinels

    def fit(self, train: dict):
        X, clip, available, suppressed, sentinels = self._inputs(train)
        if len(X) == 0:
            raise ValueError("cannot fit feature processing on an empty training set")
        selected = select_features(X, self.feature_set).copy()
        trusted = available & ~suppressed
        enhanced = self.task_aware or self.missing_indicators or self.imputer is not None
        if enhanced:
            bad_columns = np.flatnonzero(~trusted.any(axis=0)).tolist()
            if bad_columns:
                raise ValueError(
                    f"no trusted observed training values for acoustic columns {bad_columns}; "
                    "cannot fit their scaler or imputer without an explicit approved policy")
            selected[:, :N_ACOUSTIC][~trusted] = np.nan
        # With every option disabled this is the exact Phase 1 StandardScaler path.
        self.scaler = StandardScaler().fit(selected)
        if self.imputer is not None:
            acoustic = X[:, :N_ACOUSTIC].copy()
            acoustic[~trusted] = np.nan
            self.imputer.fit(acoustic)
        self.train_clip_counts_ = {name: int(np.sum(clip == name)) for name in ("vowel", "phrase", "unknown")}
        self.train_unavailable_counts_ = (~trusted).sum(axis=0).astype(int).tolist()
        self.train_sentinel_counts_ = {
            "hnr_minus_200": int(sentinels[:, 0].sum()),
            "f0_median_zero": int(sentinels[:, 8].sum()),
            "by_acoustic_column": sentinels.sum(axis=0).astype(int).tolist(),
        }
        self._unknown_seen = bool(np.any(clip == "unknown"))
        # Fail during fit if the configured imputer cannot produce usable features.
        if self.imputer is not None:
            self.transform(train)
        return self

    def transform(self, data: dict) -> np.ndarray:
        if self.scaler is None:
            raise ValueError("FeatureProcessor must be fitted on training data before transform")
        X, clip, available, suppressed, _ = self._inputs(data)
        selected = select_features(X, self.feature_set).copy()
        if self.imputer is not None:
            acoustic = X[:, :N_ACOUSTIC].copy()
            acoustic[~available | suppressed] = np.nan
            transformed = np.asarray(self.imputer.transform(acoustic), dtype=np.float64)
            if transformed.shape != acoustic.shape or not np.isfinite(transformed).all():
                raise ValueError("configured imputer must return a finite acoustic matrix with the same 14-column shape")
            selected[:, :N_ACOUSTIC] = transformed
        if not np.isfinite(selected).all():
            raise ValueError("unresolved non-finite features after the explicitly configured imputer")
        result = self.scaler.transform(selected)
        result[:, :N_ACOUSTIC][suppressed] = 0.0
        if self.missing_indicators:
            # These are availability (1 = usable), not missingness (1 = absent).
            result = np.column_stack([result, (available & ~suppressed).astype(np.float64)])
        if not np.isfinite(result).all():
            raise ValueError("feature processing produced non-finite values")
        self._unknown_seen = self._unknown_seen or bool(np.any(clip == "unknown"))
        return result

    def metadata(self) -> dict:
        limitations = []
        if self.task_aware and self._unknown_seen:
            limitations.append(
                "Some clips have unknown task labels (including unsplit PVQD where applicable); "
                "their acoustic features remain unmapped and no phrase suppression is inferred")
        if self.imputer is None:
            limitations.append("No concrete imputation strategy is selected; real acoustic NaNs and opted-in raw unavailability fail closed")
        return {
            "version": 2,
            "feature_set": self.feature_set,
            "n_features_in": N_RAW_FEATURES,
            "n_features_out": self.n_features_out,
            "layout": {"acoustic": N_ACOUSTIC, "embedding_mean": N_EMB, "embedding_std": N_EMB},
            "task_aware": self.task_aware,
            "phrase_untrusted_columns": list(self.phrase_untrusted_columns),
            "missing_indicators": self.missing_indicators,
            "indicator_definition": "14 unscaled binary availability columns appended after the selected features; 1=available and task-trusted",
            "task_suppression": "configured phrase acoustic columns are exactly zero after standardization",
            "scaler_fit": "train only; opt-in acoustic statistics use observed, task-trusted values before imputation, excluding unavailable/task-untrusted entries",
            "sentinel_policy": self.sentinel_policy,
            "sentinel_rules": policy_metadata(self.sentinel_policy),
            "imputation_strategy": self.imputation_strategy,
            "imputer_fit": ("train only; per-column medians of observed, task-trusted acoustic values" if self.imputation_strategy == "median" else
                            "caller-selected estimator fitted on observed, task-trusted training acoustics" if self.imputer is not None else None),
            "imputer_required_at_load": self.imputer is not None and self.imputation_strategy is None,
            "imputer_class": ("training.acoustic_missingness.AcousticMedianImputer" if self.imputation_strategy == "median" else
                              f"{type(self.imputer).__module__}.{type(self.imputer).__qualname__}" if self.imputer is not None else None),
            "train_clip_counts": self.train_clip_counts_,
            "train_unavailable_counts": self.train_unavailable_counts_,
            "train_sentinel_counts": self.train_sentinel_counts_,
            "unknown_clip_seen": self._unknown_seen,
            "limitations": limitations,
        }

    def save(self, directory) -> dict:
        if self.scaler is None:
            raise ValueError("cannot save an unfitted FeatureProcessor")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = self.metadata()
        state = {"mean_": self.scaler.mean_, "scale_": self.scaler.scale_,
                 "var_": self.scaler.var_, "n_samples_seen_": self.scaler.n_samples_seen_}
        if self.imputation_strategy == "median":
            state["imputer_statistics_"] = AcousticMedianImputer.validate_statistics(self.imputer.statistics_)
        (directory / "feature_processor.json").write_text(json.dumps(metadata, indent=2) + "\n")
        np.savez(directory / "feature_processor.npz", **state)
        return metadata

    @classmethod
    def load(cls, directory, *, imputer=None):
        directory = Path(directory)
        metadata = json.loads((directory / "feature_processor.json").read_text())
        version = metadata.get("version")
        if version not in (1, 2):
            raise ValueError("unsupported feature processor artifact version")
        if version == 1 and (metadata.get("sentinel_policy", "none") != "none" or
                             metadata.get("imputation_strategy") is not None):
            raise ValueError("version 1 cannot contain an active sentinel policy or built-in imputation strategy")
        strategy = metadata.get("imputation_strategy") if version == 2 else None
        sentinel_policy = metadata.get("sentinel_policy", "none") if version == 2 else "none"
        if strategy is None and not metadata.get("imputer_required_at_load") and metadata.get("imputer_class") is not None:
            raise ValueError("saved imputer class is inconsistent with a processor configured without an imputer")
        if strategy == "median" and imputer is not None:
            raise ValueError("cannot supply an external imputer when loading a built-in median imputer")
        if strategy == "median" and metadata.get("imputer_required_at_load"):
            raise ValueError("median imputer artifact cannot require an external imputer")
        if metadata.get("imputer_required_at_load") and imputer is None:
            raise ValueError("this feature processor requires the caller's explicitly supplied, already-fitted imputer")
        if not metadata.get("imputer_required_at_load") and imputer is not None:
            raise ValueError("cannot add an imputer when loading a processor trained without one")
        if imputer is not None:
            name = f"{type(imputer).__module__}.{type(imputer).__qualname__}"
            if metadata.get("imputer_class") != name:
                raise ValueError("supplied imputer class differs from the imputer used to fit this processor")
        obj = cls(metadata["feature_set"], metadata["task_aware"],
                  metadata["phrase_untrusted_columns"], metadata["missing_indicators"], imputer,
                  sentinel_policy=sentinel_policy, imputation_strategy=strategy)
        if version == 2 and metadata.get("sentinel_rules") != policy_metadata(sentinel_policy):
            raise ValueError("saved sentinel rules differ from the named sentinel policy")
        if strategy == "median":
            # The same built-in class is importable as a script or package.
            if metadata.get("imputer_class") not in ("acoustic_missingness.AcousticMedianImputer",
                                                     "training.acoustic_missingness.AcousticMedianImputer"):
                raise ValueError("saved imputer class differs from the built-in median imputer")
        if metadata.get("n_features_in") != N_RAW_FEATURES or metadata.get("n_features_out") != obj.n_features_out:
            raise ValueError("feature processor artifact has an incompatible feature layout")
        obj.scaler = StandardScaler()
        with np.load(directory / "feature_processor.npz", allow_pickle=False) as state:
            if strategy != "median" and "imputer_statistics_" in state:
                raise ValueError("saved median imputer statistics require imputation_strategy='median' and artifact version 2")
            for name in ("mean_", "scale_", "var_"):
                values = state[name]
                if values.shape != (FEATURE_WIDTHS[obj.feature_set],) or not np.isfinite(values).all():
                    raise ValueError(f"invalid saved scaler {name}")
                setattr(obj.scaler, name, values.copy())
            obj.scaler.n_samples_seen_ = state["n_samples_seen_"].copy()
            if strategy == "median":
                if "imputer_statistics_" not in state:
                    raise ValueError("median imputer artifact is missing imputer_statistics_")
                obj.imputer.statistics_ = AcousticMedianImputer.validate_statistics(state["imputer_statistics_"]).copy()
                obj.imputer.n_features_in_ = N_ACOUSTIC
        if (obj.scaler.scale_ <= 0).any() or (obj.scaler.var_ < 0).any():
            raise ValueError("invalid saved scaler variance/scale")
        obj.scaler.n_features_in_ = FEATURE_WIDTHS[obj.feature_set]
        obj.train_clip_counts_ = metadata["train_clip_counts"]
        obj.train_unavailable_counts_ = metadata["train_unavailable_counts"]
        obj.train_sentinel_counts_ = metadata.get("train_sentinel_counts", {})
        obj._unknown_seen = bool(metadata["unknown_clip_seen"])
        return obj
