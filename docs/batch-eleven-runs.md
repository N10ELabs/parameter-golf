# Batch 11: Finer Scales

## Summary

Batch 10 found the best low-bit branch so far:

- `run55_qat_group128_mid1_block6_extend`
- grouped `int4`
- `INT4_GROUP_SIZE=128`
- `INT4_CLIP_PERCENTILE=99.99`
- quantize only `blocks.6.mlp.fc.weight`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.45908502`

That branch matters because it changed the tradeoff.

We are no longer quantizing two large layers.
We are quantizing only one.

That means we still have meaningful artifact headroom left, and we can spend
some of it on better fidelity without risking the cap.

## Why This Is The Right Next Batch

With only one `int4` matrix left, the most direct remaining lever is scale
granularity.

The current one-layer branch uses:

- grouped `int4` with `INT4_GROUP_SIZE=128`

That is already better than the older rowwise recipe, but it is still a coarse
approximation for a single `1024 x 512` matrix.

Because we only quantize one layer now, we can afford smaller groups such as:

- `32`
- `16`

and still stay well under the `16 MB` cap.

So batch 11 should answer:

1. does a smaller `INT4_GROUP_SIZE` improve the one-layer recipe meaningfully
2. which grouped setting is the best balance of score and bytes
3. once a better group size is found, does a short adaptation tail improve it
4. does the finer-scale recipe want a slightly different clip than `99.99`

## Planned Runs

### run56_export_group32_block6_from_run55

- export-only from `run55`
- `INT4_GROUP_SIZE=32`
- same one-layer `blocks.6.mlp.fc.weight` target
- same `INT4_CLIP_PERCENTILE=99.99`

Goal:
- quick read on whether finer grouped scales help without retraining

### run57_export_group16_block6_from_run55

- export-only from `run55`
- `INT4_GROUP_SIZE=16`
- same one-layer target and clip

Goal:
- test an even finer grouped recipe while still staying comfortably inside the
  size budget

### run58_qat_groupbest_block6_from_run55

- resume `run55`
- use the better group size from `run56/run57`
- `200`-step low-LR adaptation tail

Goal:
- adapt the one-layer branch to the better grouped exporter

### run59_qat_groupbest_block6_clip_tune

- resume the best grouped batch-11 checkpoint
- keep the winning group size
- test a slightly tighter clip such as `INT4_CLIP_PERCENTILE=99.9`

Goal:
- check whether finer scales change the best clip setting

### run60_promote_best_batch11

- continue only the strongest batch-11 branch
- lower-LR `200`-step continuation

Goal:
- confirm whether the best finer-scale recipe has real continuation headroom

## Success Criteria

Batch 11 is successful if it does at least one of these:

- beats `run55` by a meaningful margin
- shows that finer grouped scales are worth the extra bytes
- finds a better clip setting for the one-layer branch
- identifies the practical low-bit fidelity limit before the next bigger recipe
  change

## Results

| Run | Executed change | Final roundtrip val_bpb | Total submission size int8+zlib | Outcome |
| --- | --- | --- | --- | --- |
| run56_export_group32_block6_from_run55 | export-only from `run55`, keep one-layer target, switch to `INT4_GROUP_SIZE=32` | `1.37526421` | `15,741,550` | Huge fidelity win immediately; much better than the old `group128` exporter |
| run57_export_group16_block6_from_run55 | export-only from `run55`, keep one-layer target, switch to `INT4_GROUP_SIZE=16` | `1.37525084` | `15,741,607` | Slightly better than `run56`; `group16` won the exporter probe |
| run58_qat_group16_block6_from_run55 | `200`-step low-LR QAT tail from `run55` with `group16` | `1.37346168` | `15,741,831` | Real adaptation gain on top of the better exporter |
| run59_qat_group16_block6_clip999 | resume `run58`, keep `group16`, tighten clip to `99.9` | `1.37299245` | `15,741,649` | Better score and slightly smaller artifact; tighter clip helped |
| run60_qat_group16_block6_clip999_extend | lower-LR continuation of the `group16 + clip99.9` branch | `1.37275832` | `15,741,523` | Best batch-11 result and new best low-bit run overall |

## What This Revealed

- The batch-10 one-layer checkpoint was much stronger than it looked. `run55`
  was being evaluated with fake quantization active during QAT, but when the
  saved dense checkpoint was reloaded and only quantized at export time, the
  exporter fidelity improved dramatically.
- Finer grouped scales were the missing lever. Moving from `INT4_GROUP_SIZE=128`
  to `32` or `16` on the same checkpoint cut the low-bit score from
  `1.45908502` to about `1.3753` with essentially no size penalty.
- `group16` is the better setting for the current one-layer recipe. It beat
  `group32` by a small but consistent margin and stayed well below the
  `16,000,000` byte cap.
- The finer grouped recipe still had QAT headroom. `run58`, `run59`, and
  `run60` each improved the branch further, and the tighter `99.9` clip was
  slightly better than `99.99` once the group size was small enough.

## Batch 11 Decision

Batch 11 was a major success.

It did not require a new compression family. It showed that the one-layer
grouped `int4` branch from batch 10 was already close to competitive, and the
main remaining problem was coarse scale granularity.

The new best low-bit run is:

- `run60_qat_group16_block6_clip999_extend`
- `INT4_GROUP_SIZE=16`
- `INT4_CLIP_PERCENTILE=99.9`
- quantize only `blocks.6.mlp.fc.weight`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.37275832`
- total submission size int8+zlib: `15,741,523`

That cuts the gap to the dense/int8 control `run30_depth11_1218_codecut`
(`1.35492895`) to about `0.01783 val_bpb`.

## Next Direction

The low-bit branch is now close enough that broad recipe changes are no longer
the best next step.

The highest-value batch-12 directions are:

- keep the `group16 + clip99.9 + block6` recipe and test whether even smaller
  groups such as `8` are still worth the metadata on a single layer
- or keep `group16` fixed and search whether a slightly different one-layer
  target, or a tiny two-layer hybrid, can close the last gap to the dense/int8
  control
