# Current Frontier

## Best Position Right Now

### Best legal run

- `run97_min_upstream_slide64_from_run28`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.32149156`
- total submission size: `15,908,380`

This is the strongest banked run in the repo right now.

### Best screened score

- `run95_screen_slide1024s64_from_run92`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.32008711`

This is the best score we have seen on the current fork, but it is only a
screening result on the `run92` checkpoint, not yet a finalized
submission-script promotion.

### Strongest active training branch

- `run101_10l_slide64_fp16emb_latek_muwd`
- `NUM_LAYERS=10`
- `EVAL_SEQ_LEN=1024`
- `EVAL_STRIDE=64`
- `TIED_EMBED_LR=0.1`
- `MUON_WEIGHT_DECAY=0.02`
- `INT8_KEEP_FLOAT_LARGE_NAME_PATTERNS=tok_emb.weight,blocks.8.attn.c_k.weight,blocks.9.attn.c_k.weight`

Current signal:

- model params: `18,897,488`
- reached `1400/1400` training steps
- step average stayed around `445-501 ms`
- final export/eval is still pending

This is the highest-value model-side branch still alive.

## What Is Working

### 1. Dense int8 is still the main line

The best branch is still dense, not low-bit.

- Low-bit work taught us useful things.
- It did not beat the dense frontier.
- Dense plus better evaluation/compression policy is the current winner.

### 2. `lzma + pickle_protocol=4` was a real breakthrough

That packaging shift is what reopened artifact budget and let us move past the
old `run28` and `run92` barriers.

### 3. Sliding-window eval is the highest-ROI upstream port so far

`EVAL_SEQ_LEN=1024`, `EVAL_STRIDE=64` is now the best validated eval policy on
our current model family.

It moved us from the old `1.35x` dense range down into the low `1.32x` range
without retraining.

### 4. Upstream-style sensitive-tensor handling is directionally right

The infrastructure is now in place for:

- NTK-aware RoPE extrapolation at longer eval lengths
- FP16 passthrough for large sensitive tensors
- decoupled Muon weight decay

Those are now usable levers, not just ideas.

### 5. The `10`-layer branch is viable

The first corrected `10`-layer upstream-style run is not obviously too slow,
too large, or unstable on `1xH100`.

That matters because a smaller architecture is what makes FP16 tied embedding
and late-layer FP16 passthrough realistically affordable.

## What Is Not Working

### 1. Low-bit is not the shortest path to SOTA

It was a good learning branch, but not the best current leaderboard branch.

### 2. `2048/256` sliding eval did not beat `1024/64` on the current `11`-layer family

That may still matter later with a different training recipe, but for the
current checkpoints it lost.

### 3. Two late `c_k` FP16 tensors are too expensive on `run28`

`run99` made this clear:

- serialized model int8+lzma: `16,142,248`
- total submission size: `16,174,242`

So the current `run28` branch cannot afford that export recipe.

### 4. Pure export tweaking is now lower ROI than the new `10`-layer training branch

We already got the big eval-policy win. The next meaningful score jump is more
likely to come from a new training recipe than from one more export-only sweep.

## Strongest Approach To Keep Pushing

If we want one sentence:

- **Current best practical path is dense + `lzma+p4` + `1024/64` sliding eval, while we test whether the upstream-style `10`-layer branch can beat it.**

If we want the execution order:

1. Keep `run97` as the banked legal baseline.
2. Finish `run101` and inspect final score, total bytes, and quantization gap.
3. If `run101` is competitive, stay on the `10`-layer branch.
4. If `run101` disappoints, fall back to the `run97` / `run95` family and do a
   submission-specific promotion of the best screened slide64 result.

## Next Experiments

### If `run101` is promising

Run one of these next:

1. Lower-LR / longer-warmdown variant on the same `10`-layer recipe.
2. Slightly cheaper sensitive-tensor export:
   - keep `tok_emb.weight` in FP16
   - keep only one late `c_k` in FP16 if total bytes are tight
3. Longer-context follow-up only if the `10`-layer branch looks robust enough
   to justify it.

### If `run101` is not promising

Do this instead:

1. Treat `run97` as the dense legal control.
2. Revisit the `run95` screened frontier.
3. Use submission-specific code golf to see whether the best `run92 + slide64`
   promotion can be made legal.

## `8xH100` Reality

It is fine if `8xH100` is unavailable tonight.

What is blocked:

- true leaderboard-style validation under the official wallclock regime
- confirming how much extra optimization the best recipe gets from distributed
  throughput

What is not blocked:

- pruning bad recipes on `1xH100`
- validating export/compression policy
- deciding which branch deserves the first `8xH100` attempt

So the current goal should be:

- arrive at the first `8xH100` window with one clearly preferred branch,
  not three half-believed ones

## Bottom Line

- Best legal run right now: `run97` at `1.32149156`
- Best screened score right now: `run95` at `1.32008711`
- Strongest active branch right now: `run101` `10`-layer upstream-style recipe
- Best next move right now: finish `run101`, then either stay on that branch or
  fall back to a tighter `run95/run97` promotion path
