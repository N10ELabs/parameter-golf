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

- `run102_11l_dense_slide64_lzma_p4_8xh100_20260320`
- dense `11`-layer family
- `EVAL_SEQ_LEN=1024`
- `EVAL_STRIDE=64`
- `INT4_NAME_PATTERNS=`
- `MODEL_COMPRESSOR=lzma`
- `MODEL_COMPRESS_PRESET=6`
- `QUANT_PICKLE_PROTOCOL=4`
- `QUANT_LOAD_WEIGHTS_ONLY=0`

Current signal:

- model params: `20,734,552`
- active `8xH100` timed run
- dense exporter and `lzma+p4` packaging are enabled
- periodic validation is still on, so it is a good research run but not a
  strict leaderboard-faithful training rehearsal

This is now the highest-value active branch.

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

### 5. The competition timing rule is now clear

For leaderboard attempts, we should think in two separate budgets:

- `10` minutes for training
- `10` minutes for evaluation

That means timed training runs should not waste much of the training budget on
periodic validation. Evaluation can be optimized separately.

### 6. The dense `11`-layer family is the safest live branch

The strongest near-frontier results are still closest to the `run92` / `run95`
/ `run97` line:

- dense export
- `1024/64` sliding eval
- `lzma+p4` packaging

## What Is Not Working

### 1. Low-bit is not the shortest path to SOTA

It was a good learning branch, but not the best current leaderboard branch.

### 2. `2048/256` sliding eval did not beat `1024/64` on the current `11`-layer family

That may still matter later with a different training recipe, but for the
current checkpoints it lost.

### 3. The `10`-layer `run101` branch is no longer the lead bet

It did not hold up under export controls:

- mixed-export result was poor
- dense-control result was worse
- it is not the branch we should spend the next serious timed runs on

### 4. Periodic validation during timed training is not leaderboard-faithful

Those runs are still useful for research, but they are not the cleanest
interpretation of the competition rule once we separate training and evaluation
budgets.

### 5. Pure export tweaking is not the main `8xH100` training objective

Export policy still matters, but the next timed pod runs should primarily spend
their budget on training, not on monitoring or promotion-only work.

## Strongest Approach To Keep Pushing

If we want one sentence:

- **Current best practical path is dense `11`-layer training close to the `run95/run97` family, using the full `10`-minute training budget, then a separate `1024/64` slide64 dense `lzma+p4` evaluation pass.**

If we want the execution order:

1. Keep `run97` as the banked legal baseline.
2. Use the dense `11`-layer family for the next serious `8xH100` timed runs.
3. Treat timed training and evaluation as separate budgets.
4. Only return to the `10`-layer branch if a new model-side idea specifically
   addresses its export gap.

## Next Experiments

### Track-faithful timed training run

1. Dense `11`-layer family
2. `MAX_WALLCLOCK_SECONDS=600`
3. `VAL_LOSS_EVERY=0` or very sparse
4. `INT4_NAME_PATTERNS=`
5. `MODEL_COMPRESSOR=lzma`
6. `MODEL_COMPRESS_PRESET=6`
7. `QUANT_PICKLE_PROTOCOL=4`
8. `QUANT_LOAD_WEIGHTS_ONLY=0`

### Separate evaluation/promotion run

1. Load the saved checkpoint in a separate pass
2. Use `1024/64` sliding-window eval
3. Keep the dense exporter
4. Measure final roundtrip score and artifact bytes under the separate eval
   budget

For the explicit timing interpretation and run policy, see
[track-run-spec.md](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/docs/track-run-spec.md).

## `8xH100` Reality

It is fine if `8xH100` is unavailable tonight.

What is blocked:

- strict leaderboard-faithful timing still needs a cleaner separation between
  training and evaluation than the current research launch style gives us

What is not blocked:

- deciding which family deserves the next serious `8xH100` attempts
- defining the next track-faithful training run spec
- separating training and evaluation policy

So the current goal should be:

- arrive at the first `8xH100` window with one clearly preferred branch,
  not three half-believed ones

## Bottom Line

- Best legal run right now: `run97` at `1.32149156`
- Best screened score right now: `run95` at `1.32008711`
- Strongest active branch right now: dense `11`-layer `run95/run97` family
- Best next move right now: use the full `10`-minute training budget on that
  dense family, then evaluate separately
