# Batch 21: Compression-Aware Schedule Sweep

## Summary

By the end of batch 20, we should know the best export recipe worth training
for.

That shifts the bottleneck again.

The next likely win is not “more exporter cleverness.”
It is training the model so the chosen exporter hurts less.

The upstream scan points to the most relevant schedule themes:

- lower learning rates
- longer warmdown
- smoother weight distributions
- optimize for post-export score, not just dense pre-quant loss

So batch 21 should be the first schedule sweep judged primarily by the batch-20
export recipe, not by plain dense `int8`.

## Why This Is The Right Next Batch

This fork already learned the core exporter lesson:

- better post-quant results do not reliably come from better pre-quant loss

That means the next screening batch should not ask:

- which run has the best dense validation curve

It should ask:

- which schedule family survives the final exporter best

This is the first batch where the export recipe becomes part of the objective
function rather than just the final packaging step.

## Fixed Setup

Unless a run explicitly changes them:

- dense `11`-layer family near the `run92/run95/run97` line
- `EVAL_SEQ_LEN=1024`
- `EVAL_STRIDE=64`
- `MODEL_COMPRESSOR=lzma`
- `MODEL_COMPRESS_PRESET=6`
- `QUANT_PICKLE_PROTOCOL=4`
- `QUANT_LOAD_WEIGHTS_ONLY=0`
- evaluate every candidate with the winning batch-20 export recipe

Training should stay research-style in this batch:

- not strict timed training yet
- periodic validation allowed if useful
- small enough schedules to compare multiple runs cheaply on `1xH100`

## Planned Runs

### run121_depth11_compaware_lr002_wd3600

- base family: dense `11` layers
- intended schedule shift:
  - lower matrix/scalar/head LRs toward `0.02`
  - longer `WARMDOWN_ITERS=3600`
- goal: test the simplest “train for smoother quantization” schedule against
  the chosen export recipe

### run122_depth11_compaware_lr0015_wd3600

- same as `run121`, but even gentler matrix/scalar/head LRs around `0.015`
- goal: test whether the exporter prefers a clearly smoother late-phase regime

### run123_depth11_compaware_lr002_wd4800

- same LR family as `run121`
- `WARMDOWN_ITERS=4800`
- goal: test whether an even longer low-LR tail improves post-export score more
  than it hurts raw optimization progress

### run124_depth11_compaware_bestscreen_extend

- continue the best of `run121/run122/run123`
- lower-LR continuation or slightly longer schedule
- goal: confirm that the best compression-aware branch still has runway once it
  is identified

## Evaluation Rules

For this batch, record both:

- dense pre-quant `val_bpb`
- final post-export `val_bpb` under the chosen batch-20 recipe

But the promotion rule changes:

- primary metric: lower post-export `val_bpb`
- dense pre-quant metrics are only a safety check

## Success Criteria

Batch 21 is successful if it does at least one of these:

- finds a schedule that materially improves post-export score even if dense
  pre-quant loss barely changes
- confirms that lower LR plus longer warmdown is the correct direction for this
  export recipe
- identifies a single best compression-aware schedule family to carry forward
  into context and timed training

## Decision Rule For Batch 22

If batch 21 finds a clear schedule winner, batch 22 should reinvest the gained
headroom or reduced quantization gap into context.

If batch 21 fails to separate the schedules meaningfully, batch 22 should stay
closer to pure export/eval changes and avoid a large context jump.
