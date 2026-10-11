Aria-training commit bundle SHA-256 `4689ea1ffa5130568100baf352e9b64a5eb4ddf1e55f19c46a49a959b09af2de` corresponds to the Aria v0.1 weights.

# Aria-training

研究 repo，服务于 Aria v0.1-pre 的可复现训练和一次性评估。§2.5 尚未运行，test pending；本 repo 不携带测试标签或任何 `.npz` 数据。训练输入、锁名单和 bundle 通过命令行从外部提供。

## Layout

- `training/pipeline.py`: speaker-grouped train/dev fitting and bundle emission.
- `training/verify_layout.py`: canonical `[14 acoustic | 192 embedding mean | 192 embedding std]` gate.
- `training/acoustic_missingness.py`, `training/verify_sentinels.py`: `core_v1` sentinel and train-only median handling.
- `training/harmonize.py`: train-only per-corpus severity normalization/inversion.
- `evaluate.py`: locked-set evaluation only; it never trains or tunes.
- `training/test_*.py`: unit and synthetic acceptance tests.

## Reproduce

From this directory, provide a clean canonical training NPZ and a lock list from outside the repo:

```bash
python training/verify_layout.py --data /path/to/train.npz
python training/verify_sentinels.py --data /path/to/train.npz \
  --lock-test /path/to/test_speakers.txt --report /tmp/aria-sentinels.json
python training/pipeline.py --data /path/to/train.npz \
  --lock-test /path/to/test_speakers.txt --out /tmp/aria-bundle \
  --models ridge --acoustic-sentinels core_v1 --impute median --missing-indicators
```

The evaluation script requires a separately supplied locked test NPZ and lock list. Do not put either in this repository.
