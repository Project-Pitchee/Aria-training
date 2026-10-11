"""Development-only probability calibration and speaker-level split conformal.

Every fit function in this module consumes explicitly supplied development data.
Evaluation must only call the apply/report functions with persisted parameters.
Conformal calibration speakers must be disjoint from training and from every
model/hyperparameter-selection split. Its finite-sample statement additionally
requires exchangeable speakers, and is marginal rather than corpus conditional.
"""

from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score


def _vector(values, name, *, allow_empty=False):
    out = np.asarray(values, dtype=np.float64)
    if out.ndim != 1 or (not allow_empty and not len(out)):
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain finite values")
    return out


def _binary(values, n, *, require_both=False):
    y = _vector(values, "y_binary")
    if len(y) != n or not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("y_binary must match predictions and contain only 0 and 1")
    if require_both and len(np.unique(y)) != 2:
        raise ValueError("calibration requires both healthy and disordered labels")
    return y


def _probabilities(values):
    p = _vector(values, "probabilities")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    return p


def fit_platt(logits, y_binary):
    """Fit sigmoid(slope * raw_logit + intercept) with Platt target smoothing.

    The sigmoid targets (n_positive+1)/(n_positive+2) and 1/(n_negative+2)
    avoid an unbounded maximum-likelihood solution for separated small samples.
    Returned parameters are in the original raw-logit coordinates.
    """
    raw = _vector(logits, "logits")
    y = _binary(y_binary, len(raw), require_both=True)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    targets = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
    center = float(raw.mean())
    scale = float(raw.std()) or 1.0
    x = (raw - center) / scale

    def objective(theta):
        z = theta[0] * x + theta[1]
        error = expit(z) - targets
        loss = np.mean(np.logaddexp(0, z) - targets * z)
        gradient = np.array([np.mean(error * x), np.mean(error)])
        return float(loss), gradient

    initial = np.array([0.0, math.log(targets.mean() / (1 - targets.mean()))])
    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-9})
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"Platt optimization failed: {result.message}")
    slope = float(result.x[0] / scale)
    intercept = float(result.x[1] - slope * center)
    return {"slope": slope, "intercept": intercept, "n": int(len(y)),
            "n_positive": n_pos, "n_negative": n_neg,
            "input": "raw_decision_function", "method": "platt_target_smoothing"}


def apply_platt(logits, params):
    """Apply a saved Platt fit; does not estimate any parameters."""
    raw = _vector(logits, "logits")
    slope, intercept = float(params["slope"]), float(params["intercept"])
    if not np.isfinite(slope) or not np.isfinite(intercept):
        raise ValueError("Platt slope/intercept must be finite")
    return expit(slope * raw + intercept)


def fold_platt(model, params):
    """Absorb Platt's affine logit transform into one binary linear head.

    The returned estimator retains the original input dimension and ordinary
    logistic output. No second learned linear layer is required at deployment.
    """
    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        raise ValueError("Platt folding requires a fitted binary linear classifier")
    coef, intercept = np.asarray(model.coef_), np.asarray(model.intercept_)
    if (coef.ndim != 2 or coef.shape[0] != 1 or intercept.shape != (1,)
            or not np.array_equal(getattr(model, "classes_", None), [0, 1])):
        raise ValueError("Platt folding requires a binary {0,1} linear classifier")
    slope, offset = float(params["slope"]), float(params["intercept"])
    if not np.isfinite(slope) or not np.isfinite(offset):
        raise ValueError("Platt slope/intercept must be finite")
    folded = deepcopy(model)
    folded.coef_ = np.asarray(coef, dtype=float) * slope
    folded.intercept_ = np.asarray(intercept, dtype=float) * slope + offset
    return folded


def reliability_report(y_binary, probabilities, n_bins=10):
    """Equal-width reliability bins and proper scores, without fitting anything."""
    p = _probabilities(probabilities)
    y = _binary(y_binary, len(p))
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    ids = np.minimum((p * n_bins).astype(int), n_bins - 1)
    bins = []
    for i in range(n_bins):
        mask = ids == i
        count = int(mask.sum())
        bins.append({"lower": i / n_bins, "upper": (i + 1) / n_bins,
                     "count": count,
                     "mean_predicted_probability": float(p[mask].mean()) if count else None,
                     "fraction_positive": float(y[mask].mean()) if count else None})
    safe = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return {"n": int(len(y)), "brier_score": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(safe) + (1 - y) * np.log1p(-safe))),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "reliability_bins": bins}


def operating_point_report(y_binary, probabilities, threshold):
    """Report performance at a previously chosen finite probability threshold."""
    p = _probabilities(probabilities)
    y = _binary(y_binary, len(p))
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("operating threshold must be a finite probability")
    pred = p >= threshold
    positives, negatives = int(y.sum()), int((y == 0).sum())
    tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
    sensitivity = tp / positives if positives else None
    specificity = (negatives - fp) / negatives if negatives else None
    return {"threshold": threshold, "n": int(len(y)), "true_positive": tp,
            "false_positive": fp, "true_negative": negatives - fp,
            "false_negative": positives - tp, "sensitivity": sensitivity,
            "specificity": specificity,
            "youden_j": sensitivity + specificity - 1
            if sensitivity is not None and specificity is not None else None}


def select_operating_threshold(y_binary, probabilities, min_sensitivity=0.85):
    """Maximize development Youden J subject to sensitivity >= the minimum.

    Candidates are observed probabilities, never the artificial ROC infinity
    sentinel. Ties prefer greater specificity, then the greater threshold.
    """
    p = _probabilities(probabilities)
    y = _binary(y_binary, len(p), require_both=True)
    if not np.isfinite(min_sensitivity) or not 0 <= min_sensitivity <= 1:
        raise ValueError("min_sensitivity must lie in [0, 1]")
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    ends = np.r_[np.flatnonzero(ps[:-1] != ps[1:]), len(ps) - 1]
    tp = np.cumsum(ys)[ends]
    fp = ends + 1 - tp
    sensitivity = tp / y.sum()
    specificity = 1 - fp / np.sum(y == 0)
    eligible = np.flatnonzero(sensitivity >= min_sensitivity)
    best = max(eligible, key=lambda i: (sensitivity[i] + specificity[i] - 1,
                                         specificity[i], ps[ends[i]]))
    report = operating_point_report(y, p, ps[ends[best]])
    report.update({"min_sensitivity": float(min_sensitivity),
                   "selection": "maximum_youden_j_subject_to_minimum_sensitivity",
                   "tie_break": "greater_specificity_then_greater_threshold",
                   "decision_rule": "probability >= threshold"})
    return report


def fit_conformal(y_reg, predictions, speaker_id, coverage=0.90):
    """Fit split conformal using each calibration speaker's maximum residual.

    A rank beyond the available speaker scores produces an infinite interval,
    encoded as q=None and q_is_infinite=True to keep JSON standards compliant.
    Do not replace this conservative order statistic with an interpolated
    quantile. It covers all clips of a new exchangeable speaker as a block.
    """
    y = _vector(y_reg, "y_reg", allow_empty=True)
    pred = _vector(predictions, "predictions", allow_empty=True)
    speakers = np.asarray(speaker_id, dtype=str)
    if len(pred) != len(y) or speakers.ndim != 1 or len(speakers) != len(y):
        raise ValueError("conformal labels, predictions and speakers must match")
    if np.any(np.char.str_len(speakers) == 0):
        raise ValueError("calibration speaker IDs must not be empty")
    if not np.isfinite(coverage) or not 0 < coverage < 1:
        raise ValueError("coverage must lie strictly between 0 and 1")
    unique = np.unique(speakers)
    residual = np.abs(y - pred)
    scores = np.array([residual[speakers == s].max() for s in unique])
    n = len(scores)
    rank = int(math.ceil((n + 1) * coverage))
    unbounded = rank > n
    q = None if unbounded else float(np.partition(scores, rank - 1)[rank - 1])
    return {"q": q, "q_is_infinite": bool(unbounded),
            "status": "unavailable" if not n else "unbounded" if unbounded else "ok",
            "coverage": float(coverage), "rank": rank,
            "n_calibration_clips": int(len(y)), "n_calibration_units": int(n),
            "n_calibration_speakers": int(n), "unit": "speaker_max",
            "residual_units": "normalized_severity",
            "quantile_rule": "ceil((n_speakers + 1) * coverage), without interpolation",
            "coverage_scope": "marginal simultaneous coverage of a new speaker's clips",
            "coverage_assumptions": "Calibration speakers must be disjoint from training and all model selection; speaker blocks must be exchangeable with evaluation speakers. No corpus-conditional or distribution-shift guarantee."}


def conformal_interval(predictions, params):
    """Apply a persisted radius in the same normalized units as predictions."""
    pred = _vector(predictions, "predictions")
    if params.get("status") == "unavailable":
        raise ValueError("conformal calibration unavailable: no calibration speakers")
    if params.get("q_is_infinite"):
        if params.get("q") is not None:
            raise ValueError("infinite conformal radius must be encoded with q=null")
        q = np.inf
    else:
        q = float(params["q"])
        if not np.isfinite(q) or q < 0:
            raise ValueError("conformal radius must be finite and nonnegative")
    return pred - q, pred + q
