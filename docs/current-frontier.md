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

### Best proven `8xH100` timing workflow

- strict training run:
  `run106_track_dense11_train_strict588_clean_8xh100_20260320d`
- follow-up eval runs:
  `run107_eval_dense_slide64_from_run106_8xh100_20260320d`,
  `run108_eval_mixedint4_from_run106_8xh100_20260320d`,
  `run109_eval_mixedint4_tokemb_ck_from_run106_8xh100_20260320d`

What it proved:

- real split-budget runs now work on `8xH100`
- training can finish under `10` real minutes
- separate evaluation can finish under `10` real minutes

What it did not prove:

- the resulting checkpoint is not competitive
- the dense export was byte-illegal
- the legal mixed-`int4` exports collapsed quality badly

There is no active `8xH100` training run right now.

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

### 4. The split `10 + 10` competition workflow is now implemented

The morning `8xH100` session turned this from theory into working practice:

- `STRICT_WALLCLOCK=1`
- `SKIP_POST_TRAIN_EVAL=1`
- separate eval/export pass from `INIT_MODEL_PATH=...`

The clean proof run was
`run106_track_dense11_train_strict588_clean_8xh100_20260320d`.

### 5. The competition timing rule is now both clear and measured

For leaderboard attempts, we should think in two separate budgets:

- `10` minutes for training
- `10` minutes for evaluation

That means:

- timed training runs should not waste much of the training budget on periodic
  validation
- evaluation can be optimized separately
- with the current `torchrun` wrapper, a safe internal training cap is about
  `588` seconds rather than `600`

### 6. The pod environment was healthy

- `8` H100s came up cleanly
- volume space was not the bottleneck
- the morning failures were modeling/export problems, not infrastructure ones

### 7. The dense `11`-layer family is still the safest timed-training base

The strongest near-frontier banked results are still closest to the
`run92` / `run95` / `run97` line:

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

It failed on both export paths:

- mixed-export final exact roundtrip: `1.42602122`
- dense-control final exact roundtrip: `2.16189055`
- it is not the branch we should spend the next serious timed runs on

### 4. The `run106` checkpoint family is not a promotion candidate as-is

The strict timed run was operationally valid, but the checkpoint exported
poorly:

- dense eval `run107` was byte-illegal at `19,316,058` total bytes
- legal mixed-`int4` eval `run108` fell to `2.41801223`
- best legal variant `run109` only recovered to `2.40080714`

### 5. Periodic validation during timed training is still research-only

Those runs are still useful for research, but they are not the cleanest
interpretation of the competition rule once we separate training and evaluation
budgets.

### 6. Pure pre-quant progress is not enough

The morning session reinforced that a branch must look plausible after legal
roundtrip export, not just before quantization.

## Strongest Approach To Keep Pushing

If we want one sentence:

- **Keep `run97` as the banked legal baseline, reuse the strict `10 + 10` workflow from `run106`, and only spend the next serious pod window on a branch with a plausible legal export path.**

If we want the execution order:

1. Keep `run97` as the banked legal baseline.
2. Reuse the strict split workflow from `run106` for future `8xH100` attempts.
3. Treat timed training and evaluation as separate budgets.
4. Do not return to `run101` without a specific idea for fixing its export gap.
5. Do not assume another full-budget dense `11`-layer rerun will solve the
   current export problem by itself.

## Next Experiments

### Planned Sequence While No Pods Are Active

1. Batch 19:
   [batch-nineteen-runs.md](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/docs/batch-nineteen-runs.md)
   map the true byte and score trade of FP16 `tok_emb.weight` and identify the
   gentlest payer tensors on `run97` and `run92`.
2. Batch 20:
   [batch-twenty-runs.md](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/docs/batch-twenty-runs.md)
   add and test an intermediate-precision exporter, likely grouped `int6`, if
   the current `int8` plus selective `int4` toolbox is too coarse.
3. Batch 21:
   [batch-twenty-one-runs.md](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/docs/batch-twenty-one-runs.md)
   run a compression-aware schedule sweep judged by final exporter score rather
   than dense pre-quant loss.
4. Batch 22:
   [batch-twenty-two-runs.md](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/docs/batch-twenty-two-runs.md)
   reinvest any recovered compression headroom into moderate context gains.
5. Batch 23:
   [batch-twenty-three-runs.md](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/docs/batch-twenty-three-runs.md)
   take the single best branch into the strict `8xH100` timed workflow.

For the explicit timing interpretation and run policy, see
[track-run-spec.md](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/docs/track-run-spec.md).

## `8xH100` Reality

The operational questions from this morning are mostly answered.

What is blocked:

- finding a branch that both trains well and survives legal export

What is not blocked:

- strict leaderboard-faithful timing
- separating training and evaluation policy
- launching the next pod run with a clean split workflow

So the current goal should be:

- arrive at the next `8xH100` window with one export-plausible branch, not
  just one branch that looks good before quantization

## Bottom Line

- Best legal run right now: `run97` at `1.32149156`
- Best screened score right now: `run95` at `1.32008711`
- Best proven timed workflow right now: `run106` training plus `run107` to
  `run109` eval follow-ups
- Best next move right now: keep the strict split workflow, but pick the next
  branch for legal roundtrip potential, not just training-time loss
