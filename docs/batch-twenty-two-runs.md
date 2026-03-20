# Batch 22: Context Reinvestment

## Summary

If batch 21 succeeds, the next high-value move is to spend any recovered
compression or quantization margin on context rather than on more blind depth
or stop-count probing.

The upstream scan suggests two promising context levers:

- moderate eval-context extension
- longer training context

But this fork already learned one cautionary lesson:

- `2048/256` eval lost to `1024/64` on the old dense `11`-layer family

So batch 22 should be a controlled reinvestment batch, not a leap.

## Why This Is The Right Next Batch

The point of better quantization is not just legality.
It is to buy model quality elsewhere.

For this fork, the most believable reinvestment target is context:

- it costs time, not artifact bytes, at evaluation
- it lines up with recent upstream results
- it can be tested incrementally before the next `8xH100` window

## Fixed Setup

Use the best compression-aware schedule from batch 21 and the best exporter
from batch 20.

Keep these fixed unless a run explicitly changes them:

- dense `11`-layer family
- `MODEL_COMPRESSOR=lzma`
- `MODEL_COMPRESS_PRESET=6`
- `QUANT_PICKLE_PROTOCOL=4`
- `QUANT_LOAD_WEIGHTS_ONLY=0`

## Planned Runs

### run125_export_bestrecipe_eval1408s64

- source checkpoint: best batch-21 checkpoint
- change:
  - `EVAL_SEQ_LEN=1408`
  - `EVAL_STRIDE=64`
- goal: test the smallest plausible eval-context extension that could beat the
  current `1024/64` policy without repeating the failed `2048/256` jump

### run126_export_bestrecipe_eval1536s64

- source checkpoint: same as `run125`
- change:
  - `EVAL_SEQ_LEN=1536`
  - `EVAL_STRIDE=64`
- goal: test whether a slightly larger eval-context increase is better than
  `1408` once the exporter and schedule are improved

### run127_depth11_compaware_seq2048

- training change:
  - `TRAIN_SEQ_LEN=2048`
  - otherwise stay as close as possible to the winning batch-21 schedule
- evaluation:
  - first on the standard `1024/64` path
- goal: test whether longer training context improves exporter behavior before
  adding a new eval-context variable

### run128_depth11_compaware_seq2048_eval1408

- source checkpoint: `run127`
- evaluation change:
  - `EVAL_SEQ_LEN=1408`
  - `EVAL_STRIDE=64`
- goal: test the combined “longer train context plus moderate eval extension”
  branch only after the individual pieces are screened separately

## Promotion Rules

- primary metric: lower final post-export `val_bpb`
- hard gate: legal artifact size
- tie-break 1: prefer the cheaper evaluation policy if within `0.002 val_bpb`
- tie-break 2: prefer the recipe that keeps training closer to the existing
  dense `11`-layer family if still tied

## Success Criteria

Batch 22 is successful if it does at least one of these:

- finds a moderate eval-context extension that beats `1024/64` under the new
  export recipe
- shows that longer training context helps the chosen compression-aware branch
- identifies a single best “reinvested” recipe for the next strict `8xH100`
  attempt

## Decision Rule For Batch 23

If batch 22 produces a clear winner, batch 23 should take exactly that recipe
into the strict `8xH100` timed workflow.

If batch 22 is ambiguous, batch 23 should stay with the safer batch-21 winner
and keep context conservative.
