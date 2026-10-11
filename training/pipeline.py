"""Train linear vocal-health heads using train/dev speakers only.

Default Phase 1: ridge+F2 vs v1 (logreg+F1) is the pre-registered comparison.
All Phase 2 options are explicit sensitivity analyses. ridge_2stage is folded
before serialization; LGBM is a development-only upper-bound probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle

import numpy as np
from scipy import stats
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge

import calibration as C
import data as D
import features as F
import harmonize as H
import modeling as M
import verify_layout as V

N_ACOUSTIC, N_EMB = 14, 192
MODEL_NAMES = ["logreg", "ridge", "ridge_2stage", "huber", "lgbm"]


def apply_feature_set(X, feature_set):
    return F.select_features(X, feature_set)


def sha256_of_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def to_native_severity(y_reg, data, norm):
    """Undo the per-corpus severity standardisation, row by row.

    The regression heads are fitted on the harmonized target, so their raw
    predictions are in per-corpus standard deviations. Spearman and Pearson do
    not care, but an absolute error does: MAE is only meaningful in the
    corpus's own units (CAPE-V mm for PVQD). Each row is inverted with its own
    corpus's constants so a multi-corpus dev set stays correct.
    """
    y_reg = np.asarray(y_reg, dtype=float)
    native = np.full(len(y_reg), np.nan)
    for corpus, (mu, sd) in norm.items():
        mask = data["corpus"] == corpus
        native[mask] = y_reg[mask] * sd + mu
    unknown = set(map(str, data["corpus"])) - set(norm)
    if unknown:
        raise ValueError(f"no severity norm to invert for corpora: {sorted(unknown)}")
    return native


def dev_metrics_fn(y_true_sev, y_pred, native_pred=None, native_units=None):
    """Rank/linear agreement, plus absolute error when a native prediction exists.

    ``native_pred`` is the prediction expressed in the corpus's own severity
    units. Pass it for the regression heads; omit it for a head whose output
    has no severity scale (logreg predicts a probability), where an absolute
    error against CAPE-V mm would be a category error rather than a number.
    """
    rs = stats.spearmanr(y_pred, y_true_sev)
    pr = stats.pearsonr(y_pred, y_true_sev)
    metrics = {
        "spearman_r": float(rs.statistic), "spearman_p": float(rs.pvalue),
        "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
        "n": int(len(y_true_sev)),
    }
    metrics.update(native_error_fields(y_true_sev, native_pred, native_units))
    return metrics


def native_error_fields(y_true_sev, native_pred, native_units=None):
    """The absolute-error block, in the corpus's own severity units."""
    if native_pred is None:
        return {"mae_native": None,
                "mae_note": ("head output has no severity scale; "
                             "absolute error is not defined for it")}
    errors = np.abs(np.asarray(native_pred, dtype=float)
                    - np.asarray(y_true_sev, dtype=float))
    if not np.all(np.isfinite(errors)):
        raise ValueError("native predictions must be finite to report MAE")
    return {
        "mae_native": float(errors.mean()),
        "median_abs_error_native": float(np.median(errors)),
        "max_abs_error_native": float(errors.max()),
        "native_units": native_units or "corpus_native",
    }


def split_for_calibration(dev, fraction, seed, locked):
    """Reserve dev speakers before any model selection; never modify train."""
    if not 0 < fraction < 1:
        raise ValueError("--calibration-frac must lie strictly between 0 and 1")
    selection, calibration = D.speaker_grouped_split(dev, fraction, seed)
    D.assert_no_leak(selection["speaker_id"], calibration["speaker_id"], locked)
    if len(selection["X"]) < 2 or len(calibration["X"]) < 2:
        raise ValueError("dev must contain enough speakers for selection and calibration")
    return selection, calibration


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--lock-test", required=True)
    ap.add_argument("--out", required=True, help="new/empty output directory")
    ap.add_argument("--dev-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--feature-set", default="F2", choices=["F1", "F2", "F3"])
    ap.add_argument("--models", nargs="+", choices=MODEL_NAMES,
                    default=["logreg", "ridge", "huber"])
    ap.add_argument("--lgbm-estimators", type=int, default=300)
    ap.add_argument("--task-aware", action="store_true")
    ap.add_argument("--phrase-untrusted-columns", nargs="+", type=int, default=[],
                    help="explicit zero-based acoustic column indexes; no guessed mapping")
    ap.add_argument("--missing-indicators", action="store_true",
                    help="append 14 availability flags, computed before imputation")
    ap.add_argument("--acoustic-sentinels", choices=["none", "core_v1"], default="none",
                    help="explicit HNR=-200/F0-median=0 missingness policy")
    ap.add_argument("--impute", choices=["median"], default=None,
                    help="fit acoustic medians only from available training observations")
    ap.add_argument("--boundary-weight", action="store_true",
                    help="weight native CAPE-V 15..40 mm training rows by 2")
    ap.add_argument("--platt", action="store_true")
    ap.add_argument("--conformal", action="store_true")
    ap.add_argument("--calibration-frac", type=float, default=0.5,
                    help="fraction of dev speakers reserved when calibration is enabled")
    args = ap.parse_args()
    if not 0 < args.dev_frac < 1:
        ap.error("--dev-frac must lie strictly between 0 and 1")
    if args.platt and "logreg" not in args.models:
        ap.error("--platt requires --models logreg")
    if args.conformal and not set(args.models) & {"ridge", "ridge_2stage", "huber"}:
        ap.error("--conformal requires a linear regression head")
    if args.acoustic_sentinels != "none" and (args.impute != "median" or not args.missing_indicators):
        ap.error("--acoustic-sentinels core_v1 requires --impute median --missing-indicators")
    if "ridge_2stage" in args.models and args.feature_set != "F2":
        ap.error("ridge_2stage requires F2: 206 features, or 220 with missing indicators")
    if os.path.isdir(args.out) and os.listdir(args.out):
        ap.error("--out must be empty to prevent mixing stale model/calibration artifacts")

    locked = D.read_locked_speakers(args.lock_test)
    lock_sha = sha256_of_file(args.lock_test)
    data = D.concat([D.load_npz(p, allow_acoustic_nan=args.task_aware or args.missing_indicators or args.impute is not None)
                     for p in args.data])
    H.validate_binary(data)
    if not np.all(np.isfinite(data["y_severity"])):
        raise ValueError("severity must be finite")
    for corpus in np.unique(data["corpus"]):
        kinds = np.unique(data["severity_kind"][data["corpus"] == corpus])
        if len(kinds) != 1:
            raise ValueError(f"corpus {corpus} mixes severity scales {kinds.tolist()}")
    train, original_dev = D.speaker_grouped_split(data, args.dev_frac, args.seed)
    D.assert_no_leak(train["speaker_id"], original_dev["speaker_id"], locked)
    # THE GATE, before any preprocessing is fitted: the training input must
    # declare the canonical layout for every row and prove it structurally.
    # Not conditional on which Phase 2 options are on -- F2 slices `X[:, :206]`
    # on every path, so a legacy file silently trains on 192 embedding-mean
    # columns plus 14 embedding-std columns on every path too. The locked
    # speaker guard above runs first, so a leaking file is still rejected as a
    # leak rather than as a layout problem.
    V.assert_canonical_input(data, label=", ".join(map(str, args.data)))
    if len(original_dev["X"]) < 2:
        raise ValueError("need at least two dev rows")
    dev, calibration = original_dev, None
    if args.platt or args.conformal:
        dev, calibration = split_for_calibration(original_dev, args.calibration_frac,
                                                 args.seed + 1, locked)
        D.assert_no_leak(train["speaker_id"], calibration["speaker_id"], locked)

    processor = F.FeatureProcessor(
        args.feature_set, task_aware=args.task_aware,
        phrase_untrusted_columns=args.phrase_untrusted_columns,
        missing_indicators=args.missing_indicators,
        sentinel_policy=args.acoustic_sentinels, imputation_strategy=args.impute).fit(train)
    Xt, Xd = processor.transform(train), processor.transform(dev)
    norm = H.fit_severity_norm(train)
    y_reg_t = H.apply_severity_norm(train, norm)
    units = H.native_units_label(dev)
    # Calculate weights in native units, never on the harmonized target.
    weights = M.boundary_sample_weights(train, args.boundary_weight)
    models, dev_metrics, fold_verification = {}, {}, {}

    if "logreg" in args.models:
        m = LogisticRegression(max_iter=5000, C=1.0)
        m.fit(Xt, train["y_binary"], sample_weight=weights)
        models["logreg"] = m
        p = m.predict_proba(Xd)[:, 1]
        dev_metrics["logreg"] = {**dev_metrics_fn(dev["y_severity"], p),
                                  "classification": C.reliability_report(dev["y_binary"], p)}

    if "ridge" in args.models or "ridge_2stage" in args.models:
        best = None
        for alpha in [0.1, 1.0, 10.0, 100.0]:
            m = Ridge(alpha=alpha).fit(Xt, y_reg_t, sample_weight=weights)
            predicted = m.predict(Xd)
            met = dev_metrics_fn(
                dev["y_severity"], predicted,
                native_pred=to_native_severity(predicted, dev, norm),
                native_units=units)
            if best is None or met["spearman_r"] > best[1]["spearman_r"]:
                best = (m, met, alpha)
        ridge_metrics = {**best[1], "alpha": best[2]}
        if "ridge" in args.models:
            models["ridge"], dev_metrics["ridge"] = best[0], ridge_metrics

    if "ridge_2stage" in args.models:
        m, met, verification = M.fit_two_stage_ridge(
            Xt, y_reg_t, Xd, dev["y_severity"], sample_weight=weights,
            missing_indicators=args.missing_indicators)
        models["ridge_2stage"] = m
        dev_metrics["ridge_2stage"] = {
            **met,
            **native_error_fields(
                dev["y_severity"],
                to_native_severity(m.predict(Xd), dev, norm), units),
            "single_stage_ridge_spearman_r": ridge_metrics["spearman_r"],
            "single_stage_ridge_alpha": ridge_metrics["alpha"],
            "delta_spearman_r": met["spearman_r"] - ridge_metrics["spearman_r"],
        }
        fold_verification["ridge_2stage"] = verification

    if "huber" in args.models:
        best = None
        for epsilon in [1.35, 1.5]:
            for alpha in [1.0, 10.0]:
                m = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=2000)
                m.fit(Xt, y_reg_t, sample_weight=weights)
                predicted = m.predict(Xd)
                met = dev_metrics_fn(
                    dev["y_severity"], predicted,
                    native_pred=to_native_severity(predicted, dev, norm),
                    native_units=units)
                if best is None or met["spearman_r"] > best[1]["spearman_r"]:
                    best = (m, met, epsilon, alpha)
        models["huber"] = best[0]
        dev_metrics["huber"] = {**best[1], "epsilon": best[2], "alpha": best[3]}

    if "lgbm" in args.models:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(
            n_estimators=args.lgbm_estimators, num_leaves=15,
            min_child_samples=20, feature_fraction=0.7,
            bagging_fraction=0.7, bagging_freq=1,
            lambda_l2=10.0, random_state=args.seed, verbose=-1)
        m.fit(Xt, y_reg_t, sample_weight=weights)
        models["lgbm"] = m
        lgbm_predicted = m.predict(Xd)
        dev_metrics["lgbm"] = {**dev_metrics_fn(
            dev["y_severity"], lgbm_predicted,
            native_pred=to_native_severity(lgbm_predicted, dev, norm),
            native_units=units),
                                 "_note": "UPPER-BOUND PROBE ONLY, never ships"}

    cal_artifact = {"schema_version": 1, "models": {}}
    calibrated_head = None
    if calibration is not None:
        Xc = processor.transform(calibration)
        yc = H.apply_severity_norm(calibration, norm)
        cal_artifact["split"] = {
            "source": "speaker-disjoint subset of original dev",
            "n_selection_speakers": len(np.unique(dev["speaker_id"])),
            "n_calibration_speakers": len(np.unique(calibration["speaker_id"])),
            "used_for_model_selection": False,
            "note": "Calibration-fit diagnostics are in-sample. Do not select configurations using these diagnostics or interval widths.",
        }
        if args.platt:
            m = models["logreg"]
            params = C.fit_platt(m.decision_function(Xc), calibration["y_binary"])
            before = m.predict_proba(Xc)[:, 1]
            after = C.apply_platt(m.decision_function(Xc), params)
            operating = C.select_operating_threshold(calibration["y_binary"], after)
            calibrated_head = C.fold_platt(m, params)
            max_error = max(float(np.max(np.abs(
                calibrated_head.predict_proba(X)[:, 1]
                - C.apply_platt(m.decision_function(X), params)))) for X in [Xt, Xd, Xc])
            if max_error >= M.FOLD_TOLERANCE:
                raise ArithmeticError("Platt linear folding failed")
            fold_verification["logreg_calibrated"] = {
                "passed": True, "max_abs_error": max_error,
                "tolerance": M.FOLD_TOLERANCE, "inference": "single_linear_layer_plus_sigmoid",
                "n_features": int(Xt.shape[1]),
            }
            cal_artifact["models"]["logreg"] = {
                "platt": params, "operating_point": operating,
                "before": C.reliability_report(calibration["y_binary"], before),
                "after": C.reliability_report(calibration["y_binary"], after),
                "diagnostic_split": "calibration_in_sample",
                "folded_artifact": "logreg_calibrated.pkl",
            }
            dev_metrics["logreg"]["platt_calibration"] = cal_artifact["models"]["logreg"]
        if args.conformal:
            for name in sorted(set(models) - {"logreg", "lgbm"}):
                pred = models[name].predict(Xc)
                params = C.fit_conformal(yc, pred, calibration["speaker_id"])
                lower, upper = C.conformal_interval(pred, params)
                cal_artifact["models"].setdefault(name, {})["conformal"] = params
                dev_metrics[name]["conformal_calibration"] = {
                    **params, "diagnostic_split": "calibration_in_sample",
                    "empirical_clip_coverage": float(np.mean((yc >= lower) & (yc <= upper))),
                }

    # Persist only after training/calibration validation, into a fresh bundle.
    os.makedirs(args.out, exist_ok=True)
    processor.save(args.out)
    filenames = ["feature_processor.json", "feature_processor.npz", "scaler.npz",
                 "severity_norm.json", "dev_metrics.json"]
    np.savez(os.path.join(args.out, "scaler.npz"),
             mean_=processor.scaler.mean_, scale_=processor.scaler.scale_,
             feature_set=np.array(args.feature_set))
    with open(os.path.join(args.out, "severity_norm.json"), "w") as f:
        f.write(H.norm_to_json(norm))
    artifacts = dict(models)
    if calibrated_head is not None:
        artifacts["logreg_calibrated"] = calibrated_head
    for name, model in artifacts.items():
        filename = f"{name}.pkl"
        with open(os.path.join(args.out, filename), "wb") as f:
            pickle.dump(model, f)
        filenames.append(filename)
    with open(os.path.join(args.out, "dev_metrics.json"), "w") as f:
        json.dump(dev_metrics, f, indent=2, allow_nan=False)
    if calibration is not None:
        with open(os.path.join(args.out, "calibration.json"), "w") as f:
            json.dump(cal_artifact, f, indent=2, allow_nan=False)
        filenames.append("calibration.json")
    # Detect edits to the lock while fitting, before publishing a manifest.
    if sha256_of_file(args.lock_test) != lock_sha:
        raise RuntimeError("lock file changed during training")
    split_hash = lambda ids: hashlib.sha256("\n".join(sorted(set(map(str, ids)))).encode()).hexdigest()
    manifest = {
        "schema_version": 2, "seed": args.seed, "dev_frac": args.dev_frac,
        "feature_set": args.feature_set,
        "primary_comparison": "ridge+F2 vs v1 (logreg+F1) -- PRE-REGISTERED",
        "phase2_is_sensitivity_analysis": True,
        "lock_test_sha256": lock_sha, "n_lock_speakers": len(locked),
        "n_train_clips": int(len(train["X"])),
        "n_train_speakers": len(np.unique(train["speaker_id"])),
        "n_dev_clips": int(len(dev["X"])),
        "n_dev_speakers": len(np.unique(dev["speaker_id"])),
        "n_original_dev_clips": int(len(original_dev["X"])),
        "n_calibration_clips": 0 if calibration is None else int(len(calibration["X"])),
        "n_calibration_speakers": 0 if calibration is None else len(np.unique(calibration["speaker_id"])),
        "split_speaker_sha256": {
            "train": split_hash(train["speaker_id"]), "selection_dev": split_hash(dev["speaker_id"]),
            "calibration": None if calibration is None else split_hash(calibration["speaker_id"]),
        },
        "n_features_in": int(train["X"].shape[1]), "n_features_used": int(Xt.shape[1]),
        "input_layout_provenance": V.LAYOUT_PROVENANCE,
        "feature_processing": processor.metadata(),
        "boundary_weight": {"enabled": args.boundary_weight, "severity_kind": "cape_v",
                            "native_interval_inclusive": [15.0, 40.0], "weight": 2.0,
                            "n_upweighted_train": 0 if weights is None else int(np.sum(weights == 2))},
        "calibration": {"platt": args.platt, "conformal": args.conformal,
                        "fraction_of_dev": args.calibration_frac if calibration is not None else None},
        "fold_verification": fold_verification,
        "corpora": sorted(set(map(str, train["corpus"]))),
        "models": sorted(models),
        "deployable_models": sorted(set(models) - {"lgbm"}),
        "model_roles": {name: "dev_only_upper_bound_probe" if name == "lgbm" else "linear_head"
                        for name in models},
        "calibrated_head_artifact": "logreg_calibrated.pkl" if calibrated_head is not None else None,
        "artifact_sha256": {name: sha256_of_file(os.path.join(args.out, name)) for name in filenames},
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
    print(json.dumps({"dev_metrics": dev_metrics, "manifest": manifest}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
