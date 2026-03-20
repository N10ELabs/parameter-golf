# Batch 23: Strict `8xH100` Promotion

## Summary

By batch 23, the goal is no longer open-ended research.
The goal is to convert the best planned branch into a clean
competition-faithful run:

- under `10` real minutes for training
- under `10` real minutes for evaluation
- legal total submission size

The strict workflow is already proven operationally from the morning
`run106` session.
Batch 23 is about using that workflow on a genuinely export-plausible recipe.

## Required Inputs

Only start batch 23 once the following are known:

1. best exporter from batch 20
2. best compression-aware schedule from batch 21
3. best context policy from batch 22, or an explicit decision to stay with the
   safer `1024/64` path

## Fixed Strict Workflow

Training:

- `STRICT_WALLCLOCK=1`
- `MAX_WALLCLOCK_SECONDS=588`
- `VAL_LOSS_EVERY=0`
- `SKIP_POST_TRAIN_EVAL=1`

Evaluation:

- separate pass from `INIT_MODEL_PATH=.../final_model.pt`
- `ITERATIONS=0`
- `WARMUP_STEPS=0`
- `SKIP_PREQUANT_EVAL_ZERO_ITERS=1`

Shared:

- `MODEL_COMPRESSOR=lzma`
- `MODEL_COMPRESS_PRESET=6`
- `QUANT_PICKLE_PROTOCOL=4`
- `QUANT_LOAD_WEIGHTS_ONLY=0`

## Planned Runs

### run129_track_bestrecipe_train_strict588

- recipe: best branch from batches 20 to 22
- hardware: `8xH100`
- goal: first strict timed training attempt for the new mixed-precision or
  compression-aware recipe

### run130_track_bestrecipe_eval_primary

- source checkpoint: `run129`
- recipe: exact primary export/eval configuration chosen from batches 20 to 22
- goal: main competition-faithful evaluation pass

### run131_track_bestrecipe_eval_safe_fallback

- source checkpoint: `run129`
- change: slightly safer export or eval fallback if the primary pass is too
  byte-tight or too slow
- goal: preserve a legal fallback from the same timed checkpoint

## Promotion Rules

- primary metric: lower final exact roundtrip `val_bpb`
- hard gates:
  - training process under `10` real minutes
  - evaluation process under `10` real minutes
  - total submission size `<= 16,000,000`
- tie-break 1: prefer the recipe with simpler export machinery if within
  `0.002 val_bpb`
- tie-break 2: prefer the recipe with more measured headroom on time and bytes

## Success Criteria

Batch 23 is successful if it does at least one of these:

- produces a new legal run that beats `run97`
- produces a new legal run close enough to `run97` to justify immediate follow-up
- proves that the new exporter and training recipe survive the strict `8xH100`
  workflow cleanly even if the first score is not yet a winner

## Failure Rule

If batch 23 again produces a checkpoint that trains well but exports badly,
stop and treat that as a strategy failure, not a reason for another blind timed
rerun.

The point of batches 19 to 22 is to arrive at the pod with one branch that is
already export-plausible.
