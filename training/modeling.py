"""Linear modeling primitives; two stages exist only during training."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge

N_ACOUSTIC, N_EMB = 14, 192
FOLDED_DIM = N_ACOUSTIC + N_EMB
FOLD_TOLERANCE = 1e-9


def boundary_sample_weights(data, enabled=False):
    """Weight native CAPE-V 15–40 mm, inclusively; other scales stay at 1.

    None deliberately preserves the Phase 1 sklearn fitting path when off.
    """
    if not enabled:
        return None
    severity = np.asarray(data["y_severity"], dtype=float)
    kind = np.asarray(data["severity_kind"], dtype=str)
    if not np.all(np.isfinite(severity)):
        raise ValueError("boundary weighting requires finite native severity")
    return np.where((kind == "cape_v") & (severity >= 15) & (severity <= 40),
                    2.0, 1.0)


def fold_two_stage(acoustic_model, embedding_model, *, missing_indicators=False):
    """Return one sklearn linear predictor, with no retained stage models.

    Inputs are the saved scaler's standardized [14 acoustic | 192 mean]
    features. With explicit ``missing_indicators=True``, 14 unscaled availability
    columns follow the embedding block and belong to the acoustic stage. Both
    stages are affine, so their coefficients are placed in the processor's
    column order and their intercepts summed. No refit takes place when folding.
    """
    ac = np.asarray(acoustic_model.coef_, dtype=float)
    emb = np.asarray(embedding_model.coef_, dtype=float)
    n_acoustic = N_ACOUSTIC * (2 if missing_indicators else 1)
    if ac.shape != (n_acoustic,) or emb.shape != (N_EMB,):
        raise ValueError(
            f"folding requires exactly {n_acoustic} acoustic-stage and 192 embedding coefficients")
    folded = Ridge()
    folded.coef_ = (np.concatenate([ac[:N_ACOUSTIC], emb, ac[N_ACOUSTIC:]])
                    if missing_indicators else np.concatenate([ac, emb]))
    folded.intercept_ = float(acoustic_model.intercept_ + embedding_model.intercept_)
    folded.n_features_in_ = FOLDED_DIM + (N_ACOUSTIC if missing_indicators else 0)
    return folded


def fit_two_stage_ridge(X_train, y_train, X_dev, y_dev,
                        sample_weight=None, alphas=(0.1, 1.0, 10.0, 100.0),
                        *, missing_indicators=False):
    """Train acoustics first, then embeddings on train residuals; select on dev.

    y_train is train-normalized severity. y_dev is native severity, matching
    the Phase 1 dev selection metric. Calibration data must not be passed as
    X_dev. Only the folded predictor survives this function. The default input
    contract remains exactly 206 columns. ``missing_indicators=True`` explicitly
    selects [14 acoustic | 192 mean | 14 binary availability] and fits the first
    stage on acoustic values plus availability; only embedding means fit the
    remaining train residual.
    """
    n_features = FOLDED_DIM + (N_ACOUSTIC if missing_indicators else 0)
    for name, X in (("train", X_train), ("dev", X_dev)):
        if np.ndim(X) != 2 or X.shape[1] != n_features:
            raise ValueError(f"ridge_2stage {name} input must have exactly {n_features} features")
        if not np.all(np.isfinite(X)):
            raise ValueError(f"ridge_2stage {name} input must be finite")
        if missing_indicators and not np.isin(X[:, FOLDED_DIM:], (0.0, 1.0)).all():
            raise ValueError(f"ridge_2stage {name} availability indicators must be binary 0/1")
    if len(X_train) != len(y_train) or len(X_dev) != len(y_dev):
        raise ValueError("feature/target row count mismatch")
    if len(y_dev) < 2 or not np.all(np.isfinite(y_train)) or not np.all(np.isfinite(y_dev)):
        raise ValueError("finite targets and at least two dev rows are required")
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if (sample_weight.shape != (len(y_train),)
                or not np.all(np.isfinite(sample_weight))
                or np.any(sample_weight < 0) or not np.any(sample_weight > 0)):
            raise ValueError("sample_weight must be finite, nonnegative, with positive sum")
    if not len(alphas) or any(not np.isfinite(a) or a < 0 for a in alphas):
        raise ValueError("alphas must be a nonempty finite nonnegative grid")

    acoustic_columns = (np.r_[0:N_ACOUSTIC, FOLDED_DIM:n_features]
                        if missing_indicators else slice(0, N_ACOUSTIC))
    embedding_columns = slice(N_ACOUSTIC, FOLDED_DIM)
    acoustic_train, acoustic_dev = X_train[:, acoustic_columns], X_dev[:, acoustic_columns]
    embedding_train, embedding_dev = X_train[:, embedding_columns], X_dev[:, embedding_columns]
    best = None
    for alpha_ac in alphas:
        ac = Ridge(alpha=alpha_ac).fit(acoustic_train, y_train,
                                      sample_weight=sample_weight)
        residual = y_train - ac.predict(acoustic_train)
        for alpha_emb in alphas:
            emb = Ridge(alpha=alpha_emb).fit(embedding_train, residual,
                                            sample_weight=sample_weight)
            serial = ac.predict(acoustic_dev) + emb.predict(embedding_dev)
            rs = stats.spearmanr(serial, y_dev)
            if not np.isfinite(rs.statistic):
                continue
            if best is None or rs.statistic > best[0]:
                best = (float(rs.statistic), ac, emb, float(alpha_ac), float(alpha_emb))
    if best is None:
        raise ValueError("ridge_2stage has no candidate with finite dev Spearman")
    _, ac, emb, alpha_ac, alpha_emb = best
    folded = fold_two_stage(ac, emb, missing_indicators=missing_indicators)
    errors = {}
    for name, X in (("train", X_train), ("dev", X_dev)):
        serial = ac.predict(X[:, acoustic_columns]) + emb.predict(X[:, embedding_columns])
        errors[name] = float(np.max(np.abs(serial - folded.predict(X))))
    max_error = max(errors.values())
    if not np.isfinite(max_error) or max_error >= FOLD_TOLERANCE:
        raise ArithmeticError(f"two-stage folding error {max_error} >= {FOLD_TOLERANCE}")
    pred = folded.predict(X_dev)
    rs, pr = stats.spearmanr(pred, y_dev), stats.pearsonr(pred, y_dev)
    metrics = {
        "spearman_r": float(rs.statistic), "spearman_p": float(rs.pvalue),
        "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
        "n": int(len(y_dev)), "alpha_acoustic": alpha_ac,
        "alpha_embedding": alpha_emb,
    }
    verification = {
        "passed": True, "max_abs_error": max_error,
        "max_abs_error_by_split": errors, "tolerance": FOLD_TOLERANCE,
        "n_features": n_features, "inference": "single_linear_layer",
        "coefficient_space": "standardized_features",
        "formula": "concat(w_ac, w_emb); b_ac + b_emb",
    }
    if missing_indicators:
        verification.update({
            "formula": "concat(w_ac[:14], w_emb, w_ac[14:]); b_ac + b_emb",
            "stage1_columns": list(range(N_ACOUSTIC)) + list(range(FOLDED_DIM, n_features)),
            "stage2_columns": list(range(N_ACOUSTIC, FOLDED_DIM)),
            "indicator_definition": "14 unscaled binary availability columns; 1=available",
        })
    return folded, metrics, verification
