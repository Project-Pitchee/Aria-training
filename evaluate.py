"""ONE-SHOT evaluation on the locked test set.

This is the ONLY script that may read the locked test speakers,
and it may ONLY evaluate -- never train, never tune, never select.

Column layout: both sides must be canonical, and both are checked
--------------------------------------------------------------------
Feature selection is positional: F2 is `X[:, :206]`. That is the whole
contract, and 398 columns do not prove their own order, so this script does not
branch on layout -- it refuses anything that is not provably canonical:

  * the bundle must record that its training input passed the layout gate
    (`input_layout_provenance`); a bundle fitted on an unverified file cannot
    acquire that attestation retroactively, it has to be refitted.
  * the test NPZ must declare `acoustic14_embedding_mean192_embedding_std192`
    for every row AND prove it structurally, which is the first thing done with
    the loaded arrays (`verify_layout.assert_canonical_input`).

With both sides canonical, the F2 slice is taken directly and provably means
`[14 acoustic | 192 embedding mean]`. Fed a legacy
`[emb-mean | emb-std | acoustic]` file, this script fails loudly instead of
selecting 192 embedding-mean columns plus 14 embedding-std columns, dropping
every acoustic feature, and spending the one-shot test set on a meaningless
number.

v1-vs-v2 comparison (pain point 11): with n=58, comparing two Spearman point
estimates needs a gap > ~0.156 for p<0.05. Instead, --v1-predictions takes
the v1 per-speaker predictions CSV and runs a PAIRED Wilcoxon signed-rank
test on per-speaker absolute residuals |severity - pred|.
Spearman is reported as point estimate + CI only; the 'v2 beats v1' claim
rests on the paired test.

The v1 comparator is v1's own frozen CAPE-V severity head, which outputs
native CAPE-V mm and so is directly comparable with no mapping. The former
`100 - health_score` proxy is refused; see load_v1_predictions and
reports/V1_SEVERITY_MAPPING_CONFIRMATION.md.

Usage:
    python evaluate.py --bundle ../models/v2 --model ridge \
        --test-npz ../data/pvqd_test_acoustic_first.npz \
        --lock-test ../test_speakers.txt \
        --report ../reports/v2_ridge.json --native-corpus pvqd \
        --v1-predictions ../data/v1_test_severity_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys

# Allow `python evaluate.py` from the research repo root while keeping the
# training modules isolated under training/.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "training"))

import numpy as np
from scipy import stats
from scipy.special import expit

try:
    from . import calibration as C, data as D, features as F, harmonize as H, verify_layout as V
except ImportError:
    import calibration as C
    import data as D
    import features as F
    import harmonize as H
    import verify_layout as V


def require_gated_canonical_training_input(manifest):
    """Refuse a bundle whose training input never passed the layout gate.

    Run as part of the bundle preflight, before any evaluation data is opened.
    The locked test file is canonical and gated on the way in; a model fitted
    on columns nobody proved is not comparable with it, and no amount of hash
    checking downstream notices, because every gate past this point verifies
    only the 398-column width.
    """
    provenance = manifest.get("input_layout_provenance")
    if provenance != V.LAYOUT_PROVENANCE:
        raise SystemExit(
            "this bundle does not record that its training input passed the layout gate "
            f"(input_layout_provenance={provenance!r}, required {V.LAYOUT_PROVENANCE!r}). "
            "Refusing: a bundle fitted before the gate, or on a file whose column order "
            "was never proven, cannot acquire that attestation retroactively. Refit it "
            "with pipeline.py from a training file that passes verify_layout.py, then "
            "evaluate.")


def verify_bundle_artifacts(bundle, manifest, model_name):
    """Validate the chosen model and its exact preprocessing/calibration state.

    Run this preflight before opening any evaluation data. Old Phase 1 bundles
    remain readable, but cannot acquire an unbound calibration file by accident.
    """
    root = Path(bundle)
    if model_name not in manifest.get("models", []):
        raise ValueError(f"model {model_name!r} is not recorded in this bundle")
    if int(manifest.get("schema_version", 1)) < 2:
        if (root / "calibration.json").exists() or (root / "feature_processor.json").exists():
            raise ValueError("new preprocessing/calibration artifacts require a version 2 manifest")
        return
    if model_name == "lgbm":
        raise ValueError("LGBM is a development-only probe and is ineligible for locked evaluation")
    if model_name not in manifest.get("deployable_models", []):
        raise ValueError(f"model {model_name!r} is not an eligible linear model in this bundle")
    hashes = manifest.get("artifact_sha256", {})
    required = {f"{model_name}.pkl", "feature_processor.json", "feature_processor.npz",
                "scaler.npz", "severity_norm.json"}
    options = manifest.get("calibration", {})
    if (root / "calibration.json").exists() or options.get("platt") or options.get("conformal"):
        required.add("calibration.json")
    if model_name == "logreg" and options.get("platt"):
        required.add("logreg_calibrated.pkl")
    missing = required - set(hashes)
    if missing:
        raise ValueError(f"manifest is missing artifact hashes: {sorted(missing)}")
    for filename, expected in hashes.items():
        if Path(filename).name != filename:
            raise ValueError("manifest artifact names must be local filenames")
        path = root / filename
        if not path.is_file():
            raise ValueError(f"required bundle artifact is missing: {filename}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"bundle artifact hash mismatch: {filename}")


def load_calibration(bundle, manifest, model_name):
    """Load and validate calibration before any evaluation data is opened."""
    path = Path(bundle) / "calibration.json"
    if not path.exists():
        return None, {}
    calibration = json.loads(path.read_text())
    if calibration.get("schema_version") != 1:
        raise ValueError("unsupported calibration artifact schema")
    model_calibration = calibration.get("models", {}).get(model_name, {})
    options = manifest.get("calibration", {})
    platt, point = model_calibration.get("platt"), model_calibration.get("operating_point")
    if model_name == "logreg" and options.get("platt") and (not platt or not point):
        raise ValueError("requested logreg Platt fit or operating threshold is missing")
    if platt:
        C.apply_platt(np.array([0.0]), platt)
    if point:
        threshold = point.get("threshold")
        if threshold is None or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("saved operating threshold must be a finite probability")
    conformal = model_calibration.get("conformal")
    if model_name in {"ridge", "ridge_2stage", "huber"} and options.get("conformal") and not conformal:
        raise ValueError(f"requested {model_name} conformal fit is missing")
    if conformal:
        if conformal.get("unit") != "speaker_max" or conformal.get("residual_units") != "normalized_severity":
            raise ValueError("conformal artifact must use speaker maxima in normalized severity units")
        if conformal.get("status") not in {"ok", "unbounded", "unavailable"}:
            raise ValueError("unsupported conformal calibration status")
        if conformal["status"] != "unavailable":
            C.conformal_interval(np.array([0.0]), conformal)
    return calibration, model_calibration


def conformal_report(pred_normalized, native_severity, speakers, params, corpus, norm):
    """Apply a saved interval and report native-unit clip and speaker coverage."""
    report = dict(params)
    report["evaluation_corpus"] = corpus
    report["native_units"] = "corpus_native_severity"
    if params.get("status") == "unavailable":
        report.update({"q_native": None, "empirical_clip_coverage": None,
                       "empirical_speaker_coverage": None, "intervals": []})
        return report
    lower, upper = C.conformal_interval(pred_normalized, params)
    native_lower = H.invert_severity_norm(lower, corpus, norm)
    native_upper = H.invert_severity_norm(upper, corpus, norm)
    native_pred = H.invert_severity_norm(pred_normalized, corpus, norm)
    native_severity = np.asarray(native_severity, dtype=float)
    speakers = np.asarray(speakers, dtype=str)
    if native_severity.shape != native_pred.shape or speakers.shape != native_pred.shape:
        raise ValueError("evaluation labels, predictions and speaker IDs must match")
    covered = (native_severity >= native_lower) & (native_severity <= native_upper)
    block_covered = [bool(covered[speakers == s].all()) for s in np.unique(speakers)]
    sd = float(norm[corpus][1])
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("native severity norm must have positive finite scale")
    report.update({
        "q_native": None if params.get("q_is_infinite") else float(params["q"] * sd),
        "empirical_clip_coverage": float(covered.mean()),
        "empirical_speaker_coverage": float(np.mean(block_covered)),
        "n_evaluation_clips": int(len(covered)),
        "n_evaluation_speakers": int(len(block_covered)),
        "interval_endpoint_encoding": "null lower/upper means negative/positive infinity when q_is_infinite",
        "intervals": [{"speaker_id": str(speaker), "prediction": float(pred),
                       "lower": float(lo) if np.isfinite(lo) else None,
                       "upper": float(hi) if np.isfinite(hi) else None,
                       "covered": bool(hit)}
                      for speaker, pred, lo, hi, hit in
                      zip(speakers, native_pred, native_lower, native_upper, covered)]})
    return report


def bootstrap_ci(x, y, stat_fn, n_boot=2000, seed=11):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(x)
    if n < 2:
        return None, None
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        value = stat_fn(x[idx], y[idx])
        if np.isfinite(value):
            vals.append(value)
    if not vals:
        return None, None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def load_v1_predictions(path):
    """Read v1 per-speaker CAPE-V severity predictions in native mm (0-100).

    The values must come from v1's OWN frozen CAPE-V severity head
    (Pitchee voice_severity_model.hpp: StandardScaler -> Nystroem RBF(384,
    gamma=0.001) -> Ridge(alpha=0.1) -> clip[0,100]), produced by
    scripts/v1_severity_from_frozen_head.py. That head already emits the same
    unit as v2's severity_pred_mm, so no mapping is applied here and none may
    be: every constant is frozen in the shipped v1 binary and was fit on
    split=="train" speakers only.

    The legacy `100 - health_score` proxy is REFUSED. health_score is v1's
    health/classification score; its probability mapping is anchored on the
    binary decision threshold (raw 0.4136 -> health 30), not calibrated to
    CAPE-V mm. Measured against the frozen severity head it is off by mean
    27.97 mm / max 61.03 mm and has nearly double the spread, which would
    inflate v1's paired residuals and manufacture a v2 win out of a unit
    mismatch. See reports/V1_SEVERITY_MAPPING_CONFIRMATION.md.
    """
    # Ground-truth columns that must never be mistaken for v1 predictions:
    # in Pitchee's voice_health_test_predictions.csv, `cape_v_severity` and
    # `grbas_grade` are PVQD clinical labels, not model outputs, despite
    # sharing their names with Core CLI prediction fields.
    label_columns = {"cape_v_severity", "grbas_grade", "true_label", "correct"}

    with open(path) as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        leaked = sorted(fields & label_columns)
        if leaked:
            raise SystemExit(
                "v1 predictions file %s carries ground-truth column(s) %s. "
                "These are PVQD clinical labels, not v1 predictions. Point "
                "--v1-predictions at data/v1_test_severity_predictions.csv "
                "instead. Refusing." % (path, ", ".join(leaked)))
        if "v1_severity_pred" not in fields:
            extra = ""
            if "health_score" in fields:
                extra = (" This file only has `health_score`; the "
                         "`100 - health_score` proxy is wrong by ~28 mm on "
                         "average and is no longer accepted. Regenerate with "
                         "scripts/v1_severity_from_frozen_head.py.")
            raise SystemExit(
                "v1 predictions file %s has no `v1_severity_pred` column.%s "
                "Refusing." % (path, extra))

        out = {}
        for r in reader:
            speaker = r["speaker_id"].strip()
            value = float(r["v1_severity_pred"])
            if not np.isfinite(value) or not 0.0 <= value <= 100.0:
                raise SystemExit("v1_severity_pred for %s is %r, outside the "
                                 "CAPE-V range [0, 100]. Refusing."
                                 % (speaker, value))
            if speaker in out:
                raise SystemExit("duplicate speaker_id %r in %s. Refusing."
                                 % (speaker, path))
            out[speaker] = value
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--model", required=True,
                    choices=["logreg", "ridge", "ridge_2stage", "huber", "lgbm"])
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--lock-test", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--native-corpus", default="pvqd")
    ap.add_argument("--v1-predictions", default=None)
    args = ap.parse_args()

    lock_sha = hashlib.sha256(Path(args.lock_test).read_bytes()).hexdigest()
    manifest = json.loads((Path(args.bundle) / "manifest.json").read_text())
    if manifest.get("lock_test_sha256") != lock_sha:
        raise SystemExit("manifest lock hash mismatch. Refusing.")
    verify_bundle_artifacts(args.bundle, manifest, args.model)
    require_gated_canonical_training_input(manifest)
    calibration, model_calibration = load_calibration(args.bundle, manifest, args.model)
    locked = D.read_locked_speakers(args.lock_test)

    processor_path = os.path.join(args.bundle, "feature_processor.json")
    processor = F.FeatureProcessor.load(args.bundle) if os.path.exists(processor_path) else None
    if processor is not None:
        if processor.feature_set != manifest.get("feature_set"):
            raise ValueError("manifest and feature processor disagree on the feature set")
        saved_processor = json.loads(Path(processor_path).read_text())
        if "feature_processing" in manifest and manifest["feature_processing"] != saved_processor:
            raise ValueError("manifest and saved feature processor metadata disagree")
        if "n_features_used" in manifest and manifest["n_features_used"] != processor.n_features_out:
            raise ValueError("manifest and feature processor output dimensions disagree")
    allow_nan = bool(processor is not None and
                     (processor.task_aware or processor.missing_indicators or processor.imputer is not None))
    test = D.load_npz(args.test_npz, allow_acoustic_nan=allow_nan)
    # THE GATE. First thing done with the loaded arrays, before a single column
    # is selected: the file must declare the canonical layout for every row and
    # prove it structurally. Nothing downstream re-checks column order -- every
    # later gate verifies only the 398-column width and then slices by
    # position -- so a legacy file has to die here or not at all.
    V.assert_canonical_input(test, label=str(args.test_npz))
    test_spk = set(map(str, test["speaker_id"]))
    if test_spk != locked:
        raise SystemExit("test npz speakers != locked set.")
    test_corpora = set(map(str, test["corpus"]))
    if test_corpora != {args.native_corpus}:
        raise SystemExit("test corpus must exactly match --native-corpus for native-unit reporting.")

    if processor is not None:
        feature_set = processor.feature_set
        Xt = processor.transform(test)
    else:
        with np.load(os.path.join(args.bundle, "scaler.npz"), allow_pickle=False) as sc:
            feature_set = str(sc["feature_set"]) if "feature_set" in sc else "F1"
            X = F.select_features(test["X"], feature_set)
            if (sc["mean_"].shape != (X.shape[1],) or sc["scale_"].shape != (X.shape[1],)
                    or not np.isfinite(sc["mean_"]).all() or not np.isfinite(sc["scale_"]).all()
                    or (sc["scale_"] <= 0).any()):
                raise ValueError("invalid legacy scaler artifact")
            Xt = (X - sc["mean_"]) / sc["scale_"]
    if feature_set != manifest.get("feature_set"):
        raise ValueError("manifest and feature processor disagree on the feature set")

    with open(os.path.join(args.bundle, f"{args.model}.pkl"), "rb") as f:
        model = pickle.load(f)
    if getattr(model, "n_features_in_", Xt.shape[1]) != Xt.shape[1]:
        raise ValueError("model input dimension does not match the saved feature processor")

    if args.model == "logreg":
        raw_logits = model.decision_function(Xt)
        raw = expit(raw_logits)
        score = C.apply_platt(raw_logits, model_calibration["platt"]) if model_calibration.get("platt") else raw
        sign = +1.0
        v2_sev = None
    else:
        norm = {k: tuple(v) for k, v in
                json.loads((Path(args.bundle) / "severity_norm.json").read_text()).items()}
        if (args.native_corpus not in norm or len(norm[args.native_corpus]) != 2
                or not np.isfinite(norm[args.native_corpus]).all()
                or norm[args.native_corpus][1] <= 0):
            raise ValueError("invalid or missing native-corpus severity normalization")
        y_reg = model.predict(Xt)
        native = H.invert_severity_norm(y_reg, args.native_corpus, norm)
        score, sign = native, +1.0
        v2_sev = native

    sev = test["y_severity"].astype(float)
    spk = np.asarray(test["speaker_id"], dtype=str)
    rs = stats.spearmanr(sign * score, sev) if len(sev) > 1 else None
    pr = stats.pearsonr(sign * score, sev) if len(sev) > 1 else None
    rs_lo, rs_hi = bootstrap_ci(sign * score, sev,
                                lambda a, b: stats.spearmanr(a, b).statistic)
    pr_lo, pr_hi = bootstrap_ci(sign * score, sev,
                                lambda a, b: stats.pearsonr(a, b).statistic)
    if len(sev) > 1 and np.ptp(sev) > 0:
        slope, intercept, rval, _, _ = stats.linregress(sev, sign * score)
        ols = {"intercept": float(intercept), "slope": float(slope), "r2": float(rval ** 2)}
    else:
        ols = {"intercept": None, "slope": None, "r2": None}

    def correlation_value(result, key):
        value = getattr(result, key, None)
        return float(value) if value is not None and np.isfinite(value) else None

    report = {
        "model": args.model,
        "feature_set": feature_set,
        "n_test_speakers": len(locked),
        "n_test_clips": int(len(sev)),
        "spearman_r": correlation_value(rs, "statistic"),
        "spearman_p": correlation_value(rs, "pvalue"),
        "spearman_ci95": [rs_lo, rs_hi],
        "pearson_r": correlation_value(pr, "statistic"),
        "pearson_p": correlation_value(pr, "pvalue"),
        "pearson_ci95": [pr_lo, pr_hi],
        "ols_native": ols,
        "lock_test_sha256": lock_sha,
        "native_corpus": args.native_corpus,
    }

    # Absolute error in the corpus's own units. Reported in the same pass as
    # the correlations on purpose: the locked test set is evaluated once, so a
    # metric left out here cannot be added later without spending it again.
    if v2_sev is not None:
        errors = np.abs(np.asarray(v2_sev, dtype=float) - sev)
        mae_lo, mae_hi = bootstrap_ci(
            np.asarray(v2_sev, dtype=float), sev,
            lambda a, b: float(np.mean(np.abs(a - b))))
        report["mae_native"] = float(errors.mean())
        report["mae_native_ci95"] = [mae_lo, mae_hi]
        report["median_abs_error_native"] = float(np.median(errors))
        report["max_abs_error_native"] = float(errors.max())
        report["native_units"] = H.native_units_label(test)
    else:
        report["mae_native"] = None
        report["mae_note"] = (
            f"--model {args.model} predicts a probability, not a severity; "
            "absolute error against the native scale is not defined for it")
    if args.model == "logreg":
        report["classification"] = {
            "platt_applied": bool(model_calibration.get("platt")),
            "before": C.reliability_report(test["y_binary"], raw),
            "after": C.reliability_report(test["y_binary"], score),
            "fit_source": "saved development calibration only; no evaluation fitting",
        }
        if model_calibration.get("operating_point"):
            point = model_calibration["operating_point"]
            report["classification"]["operating_point"] = C.operating_point_report(
                test["y_binary"], score, point["threshold"])
            report["classification"]["operating_point"]["selection_source"] = "saved development calibration"
    elif model_calibration.get("conformal"):
        report["conformal"] = conformal_report(y_reg, sev, spk,
                                               model_calibration["conformal"],
                                               args.native_corpus, norm)
    if calibration:
        report["calibration_split"] = {key: value for key, value in calibration.items()
                                       if key != "models"}

    if args.v1_predictions and v2_sev is not None:
        v1 = load_v1_predictions(args.v1_predictions)
        per_spk = {}
        for s, sv, pv2 in zip(spk, sev, v2_sev):
            per_spk.setdefault(s, []).append((sv, pv2))
        pairs = []
        for s, rows in per_spk.items():
            if s in v1:
                sv_m = float(np.mean([r[0] for r in rows]))
                pv2_m = float(np.mean([r[1] for r in rows]))
                pairs.append((abs(sv_m - v1[s]), abs(sv_m - pv2_m)))
        if len(pairs) != len(locked):
            raise SystemExit("v1 predictions cover %d/%d speakers; refusing."
                             % (len(pairs), len(locked)))
        r1 = np.array([p[0] for p in pairs])
        r2 = np.array([p[1] for p in pairs])
        w = stats.wilcoxon(r1, r2, alternative="greater", zero_method="wilcox")
        report["paired_wilcoxon_v1v2"] = {
            "n_pairs": int(len(pairs)),
            "v1_comparator": "v1 frozen CAPE-V severity head "
                             "(voice_severity_model.hpp), native mm, no mapping",
            "v1_predictions_path": str(args.v1_predictions),
            "v1_predictions_sha256": hashlib.sha256(
                Path(args.v1_predictions).read_bytes()).hexdigest(),
            "mean_abs_resid_v1": float(r1.mean()),
            "mean_abs_resid_v2": float(r2.mean()),
            "median_abs_resid_v1": float(np.median(r1)),
            "median_abs_resid_v2": float(np.median(r2)),
            "n_v2_better": int((r2 < r1).sum()),
            "n_v1_better": int((r1 < r2).sum()),
            "n_tied": int((r1 == r2).sum()),
            "statistic": float(w.statistic),
            "p_one_sided": float(w.pvalue),
        }

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
