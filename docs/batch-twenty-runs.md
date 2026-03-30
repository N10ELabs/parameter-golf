# Batch 20: Intermediate Precision Export

## Summary

Batch 19 should tell us whether FP16 tied embeddings are worth spending real
budget on.

If the answer is yes, the current exporter has a structural gap:

- `int8` is often too expensive to fund FP16-sensitive tensors
- `int4` is often too destructive when applied to the large tensors that would
  pay for them

That makes an intermediate precision mode the highest-value exporter change.

The upstream scan already points at the most relevant precedent:

- mixed `int8` plus `int6`

So batch 20 should add a grouped `int6` path and test whether it can fund FP16
`tok_emb.weight` more gently than the current grouped `int4` payer.

## Why This Is The Right Next Batch

The existing repo history already resolved several alternatives:

- pure packaging work already paid off
- broad mixed-`int4` MLP export is too destructive
- even good selective `int4` branches are still not close enough to the current
  `run95/run97` frontier

What remains unresolved is exactly the space between `int8` and `int4`.

That is the missing tool if we want to:

- keep `tok_emb.weight` in FP16
- maybe keep one late `c_k` in FP16 too
- still stay legal without destroying the rest of the checkpoint

## Code Prerequisites

The needed exporter hooks now exist in
[train_gpt.py](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/train_gpt.py):

1. `INT6_NAME_PATTERNS`
2. `INT6_GROUP_SIZE`
3. `INT6_CLIP_PERCENTILE`
4. packed `int6` save/load support parallel to the current `int4` path
5. `QAT_INT6` for future low-bit adaptation work if it becomes necessary
6. the artifact format remains self-contained and backwards-compatible for dense
   `int8` users

This batch should stay export-only first.
No training changes yet.

## Fixed Setup

Unless a run explicitly changes them:

- export-only evaluation from saved checkpoints
- `EVAL_SEQ_LEN=1024`
- `EVAL_STRIDE=64`
- `MODEL_COMPRESSOR=lzma`
- `MODEL_COMPRESS_PRESET=6`
- `QUANT_PICKLE_PROTOCOL=4`
- `QUANT_LOAD_WEIGHTS_ONLY=0`
- `SKIP_PREQUANT_EVAL_ZERO_ITERS=1`

Primary source checkpoint:

- `run97` if batch 19 says the idea is viable there
- `run92` as the stronger screened fallback if `run97` is too byte-tight

## Planned Runs

### run116_export_int6g64_block6_from_run97

- checkpoint: `run97`
- change:
  - `INT6_NAME_PATTERNS=blocks.6.mlp.fc.weight`
  - `INT6_GROUP_SIZE=64`
  - initial `INT6_CLIP_PERCENTILE=99.9`
- goal: establish the first gentler payer baseline against the current one-layer
  `int4` reference

### run117_export_int6g32_block6_from_run97

- checkpoint: `run97`
- change: same as `run116`, but `INT6_GROUP_SIZE=32`
- goal: test whether finer grouped `int6` buys better fidelity at an acceptable
  metadata cost

### run118_export_fp16tokemb_int6g64_block6_from_run97

- checkpoint: `run97`
- change: combine FP16 `tok_emb.weight` with the `run116` payer
- goal: test the first full “spend on embedding, pay with int6” recipe

### run119_export_fp16tokemb_int6g32_block6_from_run97

- checkpoint: `run97`
- change: combine FP16 `tok_emb.weight` with the `run117` payer
- goal: test the higher-fidelity grouped `int6` variant under the full mixed
  recipe

### run120_export_best_int6_recipe_from_run92

- checkpoint: `run92`
- change: run the better `run118/run119` recipe on the stronger screened dense
  checkpoint
- goal: see whether the new exporter can legalize or nearly legalize the
  strongest screened branch without the catastrophic transfer failures seen in
  older low-bit batches

## Optional Contingent Runs

Only execute these if batch 19 shows one payer is not enough:

- add `blocks.5.mlp.fc.weight` as a second `int6` payer
- or add a tiny FP16 late `c_k` carveout if the main recipe is already safely
  under the cap

## Promotion Rules

- primary metric: lower `final_int8_zlib_roundtrip_exact val_bpb`
- hard gate: total submission size `<= 16,000,000`
- tie-break 1: smaller total submission size if within `0.003 val_bpb`
- tie-break 2: prefer the recipe that keeps fewer tensors off the dense `int8`
  path if still tied

## Success Criteria

Batch 20 is successful if it does at least one of these:

- finds a legal or near-legal FP16-embed mixed exporter materially better than
  the old selective `int4` path
- proves that grouped `int6` is the right intermediate precision for this fork
- shows that FP16 `tok_emb.weight` plus a gentler payer is competitive enough
  to justify training for that exporter in batch 21
- rules out intermediate precision cleanly so we stop spending time on exporter
  innovation and shift entirely to schedule/model changes

## Decision Rule For Batch 21

If batch 20 finds a mixed exporter that clearly beats the batch-19 recipes,
batch 21 should optimize training directly for that exporter.

If batch 20 still fails to produce a convincing mixed exporter, batch 21 should
still do compression-aware training, but it should target the strongest dense
`int8` path instead of the new mixed recipe.
