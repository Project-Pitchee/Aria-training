"""Synthetic data generator for pipeline self-test.

Produces a PVQD-like corpus where the true severity signal is a nonlinear
function of a few features plus noise. Uses the v1 398-dim layout so the
--feature-set gate (F1/F2/F3) can be smoke-tested:
    [0:14]    14 acoustic-like features (signal lives here)
    [14:206]  192 emb-mean-like features (weak signal)
    [206:398] 192 emb-std-like features (pure noise, like v1's finding)

The blocks are emitted on their real physical scales, not as a plain Gaussian
matrix, because `verify_layout.assert_canonical_input` proves column order from
structural invariants of those scales. See `to_canonical_layout`.

Writes:
    smoke_train.npz   240 speakers (PVQD-like)
    smoke_test.npz    58 locked speakers (PVQD-like, 1 clip each)
    smoke_lock.txt    the 58 locked speaker IDs
    smoke_v1pred.csv  fake v1 per-speaker predictions (for Wilcoxon smoke test)

Run: python make_smoke_data.py (inside training/)
"""

from __future__ import annotations

import numpy as np

try:
    from .verify_layout import ACOUSTIC_SPEC
except ImportError:
    from verify_layout import ACOUSTIC_SPEC

N_ACOUSTIC, N_EMB = 14, 192
N_FEAT = N_ACOUSTIC + 2 * N_EMB   # 398
SEED = 123


def _squash(values):
    return 1.0 / (1.0 + np.exp(-values))


def to_canonical_layout(latent):
    """Map a latent (n, 398) matrix onto the canonical layout's physical scales.

    A fixture drawn straight from `rng.normal` cannot stand in for a feature
    file: `verify_layout.assert_layout` proves column order from invariants of
    the data itself, and a Gaussian matrix satisfies none of them. Its std block
    is half negative, and its acoustic block never leaves the unit box the two
    embedding blocks live in, so both candidate layouts score 3/5 and the gate
    refuses the file for having no provable column order at all.

    So each block is emitted the way its producer emits it, monotonically in the
    latent value so the fixture's severity signal survives the mapping:

      acoustic        inside the physical range declared in ACOUSTIC_SPEC, which
                      is dB and Hz and therefore far outside [-1, 1]
      embedding mean   small and centred on zero, both signs represented
      embedding std    strictly positive, as a standard deviation must be
    """
    latent = np.asarray(latent, dtype=np.float64)
    if latent.ndim != 2 or latent.shape[1] != N_FEAT:
        raise ValueError(f"latent matrix must have shape (n, {N_FEAT}), got {latent.shape}")
    X = np.empty_like(latent)
    for index, (_, low, high) in enumerate(ACOUSTIC_SPEC):
        X[:, index] = low + (high - low) * _squash(latent[:, index])
    mean_block = slice(N_ACOUSTIC, N_ACOUSTIC + N_EMB)
    std_block = slice(N_ACOUSTIC + N_EMB, N_FEAT)
    X[:, mean_block] = 0.05 * np.tanh(latent[:, mean_block])
    X[:, std_block] = 0.02 + 0.05 * _squash(latent[:, std_block])
    return X


def make_corpus(prefix, n_speakers, kind, seed, sev_scale, clips_per_speaker=None):
    rng = np.random.default_rng(seed)
    X, sev, bin_, spk, corp, sk = [], [], [], [], [], []
    for i in range(n_speakers):
        s = f"{prefix}{i:04d}"
        n_clips = clips_per_speaker or rng.integers(1, 4)
        z = rng.normal(size=N_FEAT)
        sig = (z[:4] @ np.array([1.5, -1.0, 0.8, 0.6])
               + 0.3 * (z[4] ** 2)
               + 0.2 * z[N_ACOUSTIC:N_ACOUSTIC + 8].sum())
        true_sev = sev_scale * float(1 / (1 + np.exp(-sig)))
        for _ in range(n_clips):
            # Per-clip noise lives in the latent space: the mapping has to be
            # the last step, or the std block could be pushed negative.
            xn = to_canonical_layout((z + rng.normal(scale=0.3, size=N_FEAT))[None, :])[0]
            sn = float(np.clip(true_sev + rng.normal(scale=sev_scale * 0.12),
                               0, sev_scale))
            X.append(xn)
            sev.append(sn)
            bin_.append(1 if sn > sev_scale * 0.35 else 0)
            spk.append(s)
            corp.append("pvqd")
            sk.append(kind)
    return {
        "X": np.array(X, dtype=np.float64),
        "y_severity": np.array(sev),
        "severity_kind": np.array(sk),
        "y_binary": np.array(bin_),
        "speaker_id": np.array(spk),
        "corpus": np.array(corp),
        # Explicit semantic order: width alone cannot distinguish [ac|emb]
        # from the older preparation script's incompatible [emb|ac] layout.
        "feature_layout": np.full(len(X), "acoustic14_embedding_mean192_embedding_std192"),
    }


def main() -> None:
    pvqd_tr = make_corpus("PVQD", 240, "cape_v", SEED, 100.0)
    np.savez("smoke_train.npz", **pvqd_tr)
    pvqd_te = make_corpus("PVQDT", 58, "cape_v", SEED + 2, 100.0,
                          clips_per_speaker=1)
    np.savez("smoke_test.npz", **pvqd_te)
    with open("smoke_lock.txt", "w") as f:
        f.write("\n".join(sorted(set(pvqd_te["speaker_id"].tolist()))) + "\n")
    import csv
    rng = np.random.default_rng(99)
    # Mirrors data/v1_test_severity_predictions.csv: v1's frozen CAPE-V
    # severity head emits native mm, so the fixture carries severity directly.
    # evaluate.py refuses a `health_score` column outright.
    with open("smoke_v1pred.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker_id", "v1_severity_pred"])
        for s in sorted(set(pvqd_te["speaker_id"].tolist())):
            m = pvqd_te["speaker_id"] == s
            sev_true = pvqd_te["y_severity"][m].mean()
            w.writerow([s, float(np.clip(
                sev_true + rng.normal(scale=35), 0, 100))])
    print(f"train clips={len(pvqd_tr['X'])} feats={N_FEAT}")
    print(f"test clips={len(pvqd_te['X'])}")


if __name__ == "__main__":
    main()
