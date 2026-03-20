# Batch 18: 8xH100 Morning Runs And Strict Timing

## Summary

This morning's `8xH100` pod session had two goals:

1. test whether the upstream-style `10`-layer `run101` branch justified a
   serious multi-GPU push
2. convert this fork into a competition-faithful `10` minute training plus
   `10` minute evaluation workflow

The outcome split cleanly:

- infrastructure succeeded
- the model branches we tested did not

`run106` proved that this fork can now do a real split run on `8xH100`:

- training under `10` real minutes
- a separate evaluation pass under `10` real minutes
- legal artifact size when using the mixed-`int4` exporter

But the quality results were poor enough that the morning session did not
replace the existing banked frontier. The best legal run in the repo is still
`run97_min_upstream_slide64_from_run28` at
`final_int8_zlib_roundtrip_exact val_bpb: 1.32149156`.

## Chronology

### 1. `run101_10l_slide64_fp16emb_latek_muwd_8xh100_20260320`

First real `8xH100` training run of the new upstream-style `10`-layer branch.

Config:

- `NUM_LAYERS=10`
- `ITERATIONS=1400`
- `EVAL_SEQ_LEN=1024`
- `EVAL_STRIDE=64`
- `TIED_EMBED_LR=0.1`
- `MUON_WEIGHT_DECAY=0.02`
- `INT8_KEEP_FLOAT_LARGE_NAME_PATTERNS=tok_emb.weight,blocks.8.attn.c_k.weight,blocks.9.attn.c_k.weight`

Result:

- completed `1400/1400`
- `train_time: 69069ms`
- `step_avg: 49.33ms`
- peak memory: `11262 MiB allocated`, `11706 MiB reserved`
- final pre-quant eval: `val_loss: 2.2493`, `val_bpb: 1.3321`
- final exact roundtrip: `val_loss: 2.40777710`, `val_bpb: 1.42602122`
- quantized artifact: `10,882,226` bytes
- total submission size: `10,936,504` bytes

Takeaway:

- the branch was operationally healthy on `8xH100`
- the quantization gap was too large to make it competitive

### 2. `run101_dense_export_control_8xh100_20260320`

Dense export-only control from the saved `run101` checkpoint.

Config:

- `INIT_MODEL_PATH=.../run101_8xh100_20260320/final_model.pt`
- `ITERATIONS=0`
- `WARMUP_STEPS=0`
- `SKIP_PREQUANT_EVAL_ZERO_ITERS=1`
- `DISABLE_MODEL_COMPILE=1`
- `INT4_NAME_PATTERNS=`
- same `EVAL_SEQ_LEN=1024`, `EVAL_STRIDE=64`

Result:

- final exact roundtrip: `val_loss: 3.65026163`, `val_bpb: 2.16189055`
- quantized artifact: `16,094,636` bytes
- total submission size: `16,148,914` bytes

Takeaway:

- the poor `run101` outcome was not just the mixed-`int4` default path
- the checkpoint itself quantized very badly on the dense export path
- this effectively killed the `run101` branch for current leaderboard work

### 3. `run102_11l_dense_slide64_lzma_p4_8xh100_20260320`

Research-style dense `11`-layer run closer to the `run95` / `run97` family.

Config:

- `NUM_LAYERS=11`
- `ITERATIONS=20000`
- `MAX_WALLCLOCK_SECONDS=600`
- `VAL_LOSS_EVERY=200`
- `EVAL_SEQ_LEN=1024`
- `EVAL_STRIDE=64`
- `INT4_NAME_PATTERNS=`
- `MODEL_COMPRESSOR=lzma`
- `MODEL_COMPRESS_PRESET=6`
- `QUANT_PICKLE_PROTOCOL=4`
- `QUANT_LOAD_WEIGHTS_ONLY=0`

Key observed validation points before the run was stopped:

- `step 600`: `val_bpb 1.4070`
- `step 1800`: `val_bpb 1.2834`
- `step 2200`: `val_bpb 1.2681`
- `step 5400`: `val_bpb 1.2198`

Takeaway:

- the dense `11`-layer family remained the safer modeling branch
- but the run exposed an important tooling problem:
  `MAX_WALLCLOCK_SECONDS=600` did not mean `10` real minutes once periodic
  validation overhead was included

### 4. Competition timing clarification

The repo's rule interpretation was tightened during the session:

- training has its own `10` minute budget on `8xH100`
- evaluation has its own separate `10` minute budget on `8xH100`

This means research runs and leaderboard-faithful runs must be treated
differently:

- research runs can afford periodic validation
- strict track runs should spend the training budget on learning, then run
  evaluation separately

### 5. Trainer patch for strict timing

The morning session added the missing wiring in [train_gpt.py](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/train_gpt.py):

- `STRICT_WALLCLOCK=1`
  - stop against true elapsed wallclock, not accumulated training segments
- timer starts before warmup
- `SKIP_POST_TRAIN_EVAL=1`
  - save `final_model.pt` and exit before quantization and final eval
- training-only mode no longer forces a last-step validation

That last point mattered. The first strict attempt still overran because
training-only mode was accidentally validating at the final step.

### 6. `run103_track_dense11_train_strictwall_8xh100_20260320`

First strict training-only attempt after the code patch.

Config:

- dense `11`-layer family
- `STRICT_WALLCLOCK=1`
- `MAX_WALLCLOCK_SECONDS=600`
- `VAL_LOSS_EVERY=0`
- `SKIP_POST_TRAIN_EVAL=1`

Result:

- internal `training_phase_wallclock_elapsed: 644757ms`
- external wrapper wallclock: `656s`
- `step 9322`

Takeaway:

- the forced last-step validation bug had to be fixed
- external wallclock was still about `10-11` seconds above the trainer's own
  timer because `torchrun` startup happens before `main()` starts timing

### 7. `run104` and `run105`

These were transitional cleanup attempts while fixing the strict-timing
workflow.

- both were intentionally terminated
- both ended with launcher-side `SIGTERM`
- neither should be treated as a real modeling result

### 8. `run106_track_dense11_train_strict588_clean_8xh100_20260320d`

Clean strict training-only run after fixing the last-step-validation bug and
adding wrapper headroom.

Config:

- dense `11`-layer family
- `STRICT_WALLCLOCK=1`
- `MAX_WALLCLOCK_SECONDS=588`
- `VAL_LOSS_EVERY=0`
- `SKIP_POST_TRAIN_EVAL=1`
- `INT4_NAME_PATTERNS=`
- `MODEL_COMPRESSOR=lzma`
- `MODEL_COMPRESS_PRESET=6`
- `QUANT_PICKLE_PROTOCOL=4`
- `QUANT_LOAD_WEIGHTS_ONLY=0`

Result:

- internal stop: `wallclock_elapsed: 587772ms`
- `training_phase_wallclock_elapsed: 587794ms`
- `training_only_exit wallclock_elapsed: 588125ms`
- external wrapper wallclock: `598s`
- `step 9978`
- checkpoint saved successfully

Takeaway:

- this was the first genuinely competition-faithful training run of the
  morning
- the safe internal cap with the current `torchrun` wrapper is about `588`
  seconds, not `600`

### 9. Separate evaluation runs from the `run106` checkpoint

All of the following runs loaded the `run106` checkpoint in a separate pass
with:

- `ITERATIONS=0`
- `WARMUP_STEPS=0`
- `DISABLE_MODEL_COMPILE=1`
- `SKIP_PREQUANT_EVAL_ZERO_ITERS=1`
- `EVAL_SEQ_LEN=1024`
- `EVAL_STRIDE=64`
- `MODEL_COMPRESSOR=lzma`
- `MODEL_COMPRESS_PRESET=6`
- `QUANT_PICKLE_PROTOCOL=4`
- `QUANT_LOAD_WEIGHTS_ONLY=0`

#### `run107_eval_dense_slide64_from_run106_8xh100_20260320d`

- exporter: dense (`INT4_NAME_PATTERNS=`)
- external wallclock: `108s`
- total submission size: `19,316,058`
- final exact roundtrip: `val_loss: 3.71032873`, `val_bpb: 2.19746567`

Takeaway:

- evaluation time was fine
- artifact bytes were not legal
- the dense export path was still poor on quality

#### `run108_eval_mixedint4_from_run106_8xh100_20260320d`

- exporter: default mixed-`int4`
- external wallclock: `107s`
- total submission size: `13,243,654`
- final exact roundtrip: `val_loss: 4.08271236`, `val_bpb: 2.41801223`

Takeaway:

- fully legal on time and bytes
- quality collapsed badly

#### `run109_eval_mixedint4_tokemb_ck_from_run106_8xh100_20260320d`

- exporter: default mixed-`int4` plus
  `INT8_KEEP_FLOAT_LARGE_NAME_PATTERNS=tok_emb.weight,blocks.9.attn.c_k.weight,blocks.10.attn.c_k.weight`
- external wallclock: `106s`
- total submission size: `13,755,786`
- final exact roundtrip: `val_loss: 4.05366229`, `val_bpb: 2.40080714`

Takeaway:

- this was the best legal artifact from the new timed checkpoint
- it was still nowhere close to the current frontier

## Major Findings

### 1. The strict `10 + 10` workflow is now real

This was the most important operational result of the morning.

- strict training now works under real wallclock
- separate evaluation also fits comfortably under the evaluation budget
- future pod attempts can use the split workflow with confidence

### 2. The safe internal training cap is `588s`, not `600s`

With the current `torchrun` launcher pattern, process startup costs real time
before `main()` starts the timer.

Measured effect:

- `run103`: internal `644.8s`, external `656s`
- `run106`: internal `588.1s`, external `598s`

So the practical strict-launch rule is:

- use `MAX_WALLCLOCK_SECONDS=588`
- keep `STRICT_WALLCLOCK=1`
- keep `VAL_LOSS_EVERY=0`
- keep `SKIP_POST_TRAIN_EVAL=1`

### 3. `run101` is not the path forward

The new `10`-layer branch had one plausible selling point:

- smaller model
- potentially better byte efficiency

But the actual results were bad on both export paths:

- mixed-export final exact roundtrip: `1.42602122`
- dense-control final exact roundtrip: `2.16189055`

This branch is no longer the lead bet.

### 4. A full-budget dense `11`-layer checkpoint can fit the training rule, but not yet the promotion goal

`run106` mattered because it answered the operational question:

- yes, the dense `11`-layer family can be trained under a real `10` minute
  wallclock on `8xH100`

But it failed the more important leaderboard question:

- dense export was byte-illegal and poor
- legal mixed-`int4` export destroyed score

So `run106` was an infrastructure success, not a frontier success.

### 5. Pre-quant quality alone is not enough

The morning runs reinforced a recurring lesson:

- good pre-quant loss does not guarantee a good legal submission
- export behavior is a first-class model-selection criterion

Future pod runs should only be given serious budget if they have a plausible
legal roundtrip path.

### 6. The pod environment itself was not the problem

Operational notes from the session:

- the `8xH100` pod was healthy
- volume space was not constrained
- there were no disk-pressure errors
- the pod was stopped cleanly after the session

## Recommended Use Of These Results

1. Keep `run97` as the banked legal baseline.
2. Reuse the strict split workflow from `run106` for future `8xH100` attempts.
3. Do not spend another pod window on `run101`.
4. Do not assume that more time on the same dense `11`-layer export path will
   fix the problem.
5. Choose the next pod branch based on likely legal roundtrip behavior, not
   just training-time validation curves.
