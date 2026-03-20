# Track Run Spec

This file turns the competition timing rule into concrete guidance for this
fork.

## Competition Interpretation

For leaderboard submissions, the competition gives us two separate budgets on
`8xH100`:

1. up to `10` minutes for training
2. up to `10` minutes for evaluation

That means:

- training should spend its budget on actual learning
- evaluation can be optimized separately
- periodic validation during the timed training run is not free in the spirit
  of the rule, even if the current trainer does not count it cleanly

## Implications For This Fork

### 1. Research runs and track-faithful runs are different

Research runs can keep `VAL_LOSS_EVERY=200` or similar if the extra signal is
worth the cost.

Track-faithful runs should not spend much of the training window on periodic
validation.

### 2. `MAX_WALLCLOCK_SECONDS=600` is not enough by itself

In the current `train_gpt.py`, the stop condition is driven by accumulated
training-time segments, not a clean end-to-end elapsed-wallclock measurement.

So a run can respect the current internal cap while still taking more than
`10` real minutes once warmup, periodic validation, and final export/eval are
included.

### 3. Separate training and evaluation on purpose

The clean workflow is:

1. run timed training with minimal or no periodic validation
2. save the checkpoint
3. launch a separate export/eval run under the evaluation budget

## Current Track-Focused Direction

The safest live family is now the dense `11`-layer line closest to
`run95` / `run97`, not the `10`-layer `run101` branch.

Current policy for that family:

- dense exporter: `INT4_NAME_PATTERNS=`
- packaging: `MODEL_COMPRESSOR=lzma`, `MODEL_COMPRESS_PRESET=6`,
  `QUANT_PICKLE_PROTOCOL=4`, `QUANT_LOAD_WEIGHTS_ONLY=0`
- evaluation: `EVAL_SEQ_LEN=1024`, `EVAL_STRIDE=64`
- training focus: use the full `10`-minute training budget on `8xH100`

## Next Runs

### Track-faithful timed training run

Use this when the goal is a real leaderboard-style training attempt:

- dense `11`-layer family
- `MAX_WALLCLOCK_SECONDS=600`
- `VAL_LOSS_EVERY=0` or very sparse
- no mixed-`int4` exporter
- save the checkpoint and stop

The goal is to spend the training budget on model improvement, not monitoring.

### Separate evaluation/promotion run

After a timed training run finishes:

- load the saved checkpoint with `INIT_MODEL_PATH=...`
- use export-only or near-export-only settings
- run `1024/64` sliding-window eval
- measure final roundtrip score and artifact bytes under the separate eval
  budget

### Research-only timed runs

These are still useful, but should be labeled clearly as research runs:

- periodic validation enabled
- not a strict leaderboard-faithful timing rehearsal
- useful for curve shape, stop-point intuition, and regression checks

## Short Rule

If the question is “what should the next serious runs optimize for,” the answer
is:

- **full `10` minutes for training**
- **separate `10` minutes for evaluation**
- **dense `11`-layer `run95/run97` family first**
