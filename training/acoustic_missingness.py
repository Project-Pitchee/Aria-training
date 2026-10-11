"""Explicit missingness rules and train-fitted imputation for v1 acoustics.

Column names mirror ``hoarseness/extract_features.py:ACOUSTIC_FEATURES``.
Callers must verify the canonical raw feature layout before applying a policy.
No sentinel policy is selected by default, and no rule treats arbitrary zeros
as unavailable.
"""

from __future__ import annotations

import numpy as np


ACOUSTIC_FEATURES = (
    "hnr_db", "cpps_db", "jitter_local", "jitter_rap", "jitter_ppq5",
    "shimmer_local", "shimmer_apq3", "shimmer_apq5", "f0_median",
    "f0_iqr", "f0_cv", "voiced_fraction", "intensity_median", "intensity_iqr",
)
N_ACOUSTIC = len(ACOUSTIC_FEATURES)
SENTINEL_POLICIES = ("none", "core_v1")
HNR_INDEX = 0
F0_MEDIAN_INDEX = 8
F0_SUMMARY_INDEXES = (8, 9, 10)


def validate_sentinel_policy(policy: str) -> None:
    if policy not in SENTINEL_POLICIES:
        raise ValueError(f"unknown acoustic sentinel policy: {policy!r}; choose none or core_v1")


def sentinel_mask(X, policy: str = "none") -> np.ndarray:
    """Return a fresh 14-column mask; True means a confirmed raw sentinel.

    F0 IQR/CV zeros are unavailable only when the same row's F0 median is zero.
    A constant voiced track can legitimately have zero IQR and CV.
    """
    validate_sentinel_policy(policy)
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] < N_ACOUSTIC:
        raise ValueError("acoustic sentinel detection requires at least 14 acoustic columns")
    mask = np.zeros((len(X), N_ACOUSTIC), dtype=bool)
    if policy == "core_v1":
        mask[:, HNR_INDEX] = X[:, HNR_INDEX] == -200.0
        mask[np.ix_(X[:, F0_MEDIAN_INDEX] == 0.0, F0_SUMMARY_INDEXES)] = True
    return mask


def policy_metadata(policy: str) -> dict:
    """Describe the exact policy without inferring unavailable values."""
    validate_sentinel_policy(policy)
    return {
        "name": policy,
        "acoustic_features": [
            {"index": index, "name": name} for index, name in enumerate(ACOUSTIC_FEATURES)
        ],
        "rules": ([
            {"name": "hnr_minus_200", "condition": "hnr_db == -200.0",
             "unavailable_columns": [HNR_INDEX]},
            {"name": "f0_median_zero", "condition": "f0_median == 0.0",
             "unavailable_columns": list(F0_SUMMARY_INDEXES)},
        ] if policy == "core_v1" else []),
        "other_zeros": "preserved, including f0_iqr/f0_cv zeros when f0_median is nonzero",
        "availability": "raw finite availability AND supplied acoustic_available AND NOT sentinel mask; computed before imputation",
    }


class AcousticMedianImputer:
    """Per-column medians fitted from available training acoustics only.

    The caller represents unavailable or task-untrusted values by NaN before
    ``fit``. ``transform`` only fills NaNs with the fixed 14-element statistics
    and never refits or changes the input matrix.
    """

    @staticmethod
    def _matrix(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != N_ACOUSTIC:
            raise ValueError("AcousticMedianImputer requires a 14-column acoustic matrix")
        if np.isinf(X).any():
            raise ValueError("AcousticMedianImputer forbids infinity; use NaN for unavailable acoustics")
        return X

    @staticmethod
    def validate_statistics(statistics) -> np.ndarray:
        values = np.asarray(statistics, dtype=np.float64)
        if values.shape != (N_ACOUSTIC,) or not np.isfinite(values).all():
            raise ValueError("invalid median imputer statistics: expected 14 finite values")
        return values

    def fit(self, X):
        X = self._matrix(X)
        bad_columns = np.flatnonzero(~np.isfinite(X).any(axis=0)).tolist()
        if bad_columns:
            raise ValueError(
                f"no trusted observed training values for acoustic columns {bad_columns}; "
                "cannot fit median imputation")
        self.statistics_ = self.validate_statistics(np.nanmedian(X, axis=0)).copy()
        self.n_features_in_ = N_ACOUSTIC
        return self

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "statistics_"):
            raise ValueError("AcousticMedianImputer must be fitted on training data before transform")
        X = self._matrix(X)
        statistics = self.validate_statistics(self.statistics_)
        return np.where(np.isnan(X), statistics, X)
