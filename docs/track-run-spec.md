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
- periodic validation during the timed training run counts against the training
  wallclock in practice, even if older local scripts did not measure it
  correctly

## Implemented Strict Workflow

The morning `8xH100` session added the missing controls to
[train_gpt.py](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/train_gpt.py):

### `STRICT_WALLCLOCK=1`

- training stops against true elapsed wallclock
- the timer starts before warmup
- this is the right mode for strict training-budget runs

### `SKIP_POST_TRAIN_EVAL=1`

- save `final_model.pt`
- exit before quantization and final eval
- use this for the timed training phase only

### No forced last-step validation in training-only mode

This bug mattered in practice. The first strict attempt still overran because
training-only mode was validating at the final step. The clean fix was to skip
that validation path when `SKIP_POST_TRAIN_EVAL=1`.

## Research Runs Versus Track Runs

### Research runs

Use these when you want curve visibility:

- `VAL_LOSS_EVERY=200` or similar
- periodic validation on
- not a strict leaderboard-faithful rehearsal

### Track-faithful runs

Use these when the goal is a real competition-valid attempt:

- `VAL_LOSS_EVERY=0`
- timed training only
- separate export/eval afterward

## Measured `8xH100` Timing Reality

### `run103` showed why `600` internal seconds was unsafe

First strict attempt:

- `STRICT_WALLCLOCK=1`
- `MAX_WALLCLOCK_SECONDS=600`
- `VAL_LOSS_EVERY=0`
- `SKIP_POST_TRAIN_EVAL=1`

Observed timing:

- internal `training_phase_wallclock_elapsed: 644757ms`
- external wrapper wallclock: `656s`

Why it failed:

- training-only mode still triggered a final validation
- `torchrun` startup also adds real wallclock before `main()` begins timing

### `run106` established the practical safe cap

Clean strict run:

- `STRICT_WALLCLOCK=1`
- `MAX_WALLCLOCK_SECONDS=588`
- `VAL_LOSS_EVERY=0`
- `SKIP_POST_TRAIN_EVAL=1`

Observed timing:

- internal `training_only_exit wallclock_elapsed: 588125ms`
- external wrapper wallclock: `598s`

Practical rule:

- with the current `torchrun --standalone --nproc_per_node=8` launch pattern,
  use `MAX_WALLCLOCK_SECONDS=588` for strict training attempts

### Separate evaluation easily fit the eval budget

From the `run106` checkpoint:

- `run107` dense eval: `108s`
- `run108` legal mixed-`int4` eval: `107s`
- `run109` legal mixed-`int4` plus float-passthrough eval: `106s`

So the split workflow is real:

- under `10` minutes to train
- under `10` minutes to evaluate

## Concrete Run Policy

The clean workflow is:

1. run timed training with minimal or no periodic validation
2. save the checkpoint
3. launch a separate export/eval run under the evaluation budget

## Current Track-Focused Direction

The best banked baseline is still:

- `run97_min_upstream_slide64_from_run28`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.32149156`

The most important new infrastructure result is:

- `run106_track_dense11_train_strict588_clean_8xh100_20260320d`

That run proved the timing workflow, but not the modeling branch. The next pod
attempt should reuse the strict workflow while choosing a branch for likely
legal export quality, not just training-time loss.

Current default policy for strict runs:

- dense exporter: `INT4_NAME_PATTERNS=`
- packaging: `MODEL_COMPRESSOR=lzma`, `MODEL_COMPRESS_PRESET=6`,
  `QUANT_PICKLE_PROTOCOL=4`, `QUANT_LOAD_WEIGHTS_ONLY=0`
- evaluation: `EVAL_SEQ_LEN=1024`, `EVAL_STRIDE=64`
- training focus: use almost the full `10`-minute training budget on `8xH100`
  while leaving wrapper headroom

## Next Runs

### Track-faithful timed training run

Use this when the goal is a real leaderboard-style training attempt:

- dense `11`-layer family
- `STRICT_WALLCLOCK=1`
- `MAX_WALLCLOCK_SECONDS=588`
- `VAL_LOSS_EVERY=0`
- `SKIP_POST_TRAIN_EVAL=1`
- no mixed-`int4` exporter
- save the checkpoint and stop

The goal is to spend the training budget on model improvement, not monitoring.

### Separate evaluation/promotion run

After a timed training run finishes:

- load the saved checkpoint with `INIT_MODEL_PATH=...`
- use `ITERATIONS=0`
- use `WARMUP_STEPS=0`
- use `SKIP_PREQUANT_EVAL_ZERO_ITERS=1`
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

- **a real `598s`-or-less training process, which means about `588s` internal
  wallclock**
- **separate `10` minutes for evaluation**
- **reuse the strict workflow from `run106`, but only on a branch with a
  plausible legal roundtrip path**
