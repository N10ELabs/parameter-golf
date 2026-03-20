# Batch 5: Artifact Engineering and Cap-Edge Validation

## Why This Batch Exists

Batch 4 left `run18_depth12_924` as the best legal run, but only by `3,977` total bytes. At that point, broad hyperparameter search was no longer the highest-value move. The search needed to shift from model design toward artifact engineering.

The goal of batch 5 was:

- shrink code bytes without changing training behavior
- keep or improve compressed model size
- use the recovered headroom to retest the best pure-depth stop points

## Artifact Engineering Work

### Code-size reduction

`train_gpt.py` was trimmed from `47,642` bytes to `40,354` bytes by removing nonessential comments, the large top docstring, and startup logging that did not affect training or evaluation.

### Serializer benchmark

Two serialization changes were tested against the saved `run18` full-precision checkpoint:

1. A compact positional format:
   - reduced metadata structure aggressively
   - made the packed model worse
   - projected total bytes for `run18`: `16,005,422`
   - verdict: rejected

2. A hybrid keyed format:
   - kept the old keyed `quantized/scales/passthrough` layout
   - dropped redundant dtype metadata and recovered dtypes from the template state at load time
   - projected total bytes for `run18`: `15,988,688`
   - recovered `11,312` bytes of total headroom
   - verdict: adopted

## Runs

### run19_depth12_925_codecut

- Change: `NUM_LAYERS=12`, `ITERATIONS=925`, trimmed `train_gpt.py`, hybrid serializer
- Goal: retest the old `925` pure-depth stop point under the smaller code budget
- Outcome:
  - roundtrip `val_bpb`: `1.35971082`
  - int8+zlib model bytes: `15,952,925`
  - total submission size int8+zlib: `15,993,279`
- Result: new best valid run overall

### run20_depth12_926_codecut

- Change: `NUM_LAYERS=12`, `ITERATIONS=926`, trimmed `train_gpt.py`, hybrid serializer
- Goal: test whether the newly recovered artifact headroom was enough for one more pure-depth step
- Outcome:
  - roundtrip `val_bpb`: `1.36003188`
  - int8+zlib model bytes: `15,958,008`
  - total submission size int8+zlib: `15,998,362`
- Result: legal, but worse than `run19`

## Batch 5 Decision

Batch 5 showed that artifact engineering was the right move. The code trim plus hybrid serializer legalized `925` and made it the new best valid stop point.

Current best valid base:

- `NUM_LAYERS=12`
- `ITERATIONS=925`
- trimmed `train_gpt.py`
- hybrid int8 serialization

Current best valid score:

- `final_int8_zlib_roundtrip_exact val_bpb: 1.35971082`
- total submission size int8+zlib: `15,993,279`

## Next Direction

The next gains are unlikely to come from another naive step-count increase. `926` fit, but it regressed slightly and left only `1,638` bytes of headroom.

The most sensible next search space is:

- more code-byte reduction
- compression-friendly changes that do not harm score
- small architecture changes that reduce artifact size per unit of quality
