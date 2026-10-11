"""Multi-corpus label harmonization.

Problem: corpora don't share a severity scale (PVQD: CAPE-V 0-100 mm and
GRBAS 0-3; SVD: diagnosis labels only). Training one regression head
on raw mixed units is meaningless.

Solution: per-corpus standardization of the continuous target.
    y_reg = (y_severity - mean_corpus) / std_corpus
computed on TRAIN speakers only, then applied to dev. The fitted
(corpus -> (mean, std)) map is saved with the model so evaluation can
invert it back to native units for reporting (e.g. MCID in CAPE-V mm).

Binary labels are harmonized by rule, not statistics:
    disordered = 1 if any diagnosed pathology OR perceptual grade > 0
    The corpus loader is responsible for producing y_binary under this rule;
    this module only validates it is {0,1}.
"""

from __future__ import annotations

import json

import numpy as np


# Spelled out so a reported absolute error cannot be quoted without its scale.
NATIVE_UNITS = {"cape_v": "cape_v_mm_0_100", "grbas": "grbas_grade_0_3"}


def native_units_label(data: dict) -> str:
    """Name the severity scale the rows are measured on."""
    kinds = sorted(set(map(str, data["severity_kind"])))
    return "+".join(NATIVE_UNITS.get(kind, kind) for kind in kinds)


def fit_severity_norm(data: dict, min_speakers: int = 5):
    """Fit per-corpus (mean, std) of y_severity on TRAIN data only."""
    norm = {}
    for corp in sorted(set(map(str, data["corpus"]))):
        m = data["corpus"] == corp
        n_spk = len(set(map(str, data["speaker_id"][m])))
        if n_spk < min_speakers:
            raise ValueError(
                f"corpus {corp}: only {n_spk} train speakers, "
                f"need >= {min_speakers} for a stable norm")
        y = data["y_severity"][m].astype(float)
        mu, sd = float(y.mean()), float(y.std())
        if sd <= 0:
            raise ValueError(f"corpus {corp}: zero severity variance")
        norm[corp] = (mu, sd)
    return norm


def apply_severity_norm(data: dict, norm) -> np.ndarray:
    y = np.zeros(len(data["y_severity"]))
    for corp, (mu, sd) in norm.items():
        m = data["corpus"] == corp
        if not np.any(m):
            continue
        y[m] = (data["y_severity"][m].astype(float) - mu) / sd
    unknown = set(map(str, data["corpus"])) - set(norm)
    if unknown:
        raise ValueError(f"no severity norm fitted for corpora: {sorted(unknown)}")
    return y


def invert_severity_norm(y_reg: np.ndarray, corpus: str, norm) -> np.ndarray:
    mu, sd = norm[corpus]
    return y_reg * sd + mu


def validate_binary(data: dict) -> None:
    vals = set(np.unique(data["y_binary"]).tolist())
    if vals - {0, 1}:
        raise ValueError(f"y_binary must be {{0,1}}, found {sorted(vals)}")


def norm_to_json(norm) -> str:
    return json.dumps({k: [mu, sd] for k, (mu, sd) in norm.items()})
