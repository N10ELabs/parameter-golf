# Batch 9: Int4 Fidelity

## Summary

Batch 8 proved that selective QAT makes low-bit export viable, but it also showed a clear plateau shape.
The best low-bit line is now the middle-`2` MLP branch, and the last continuation only improved the exported score by a small margin.

That changes the question again.

Batch 9 should not spend more time on the same rowwise `int4` recipe.
It should spend some of the remaining artifact headroom on better `int4` fidelity.

The simplest high-value change is:

- keep the winning middle-`2` target set
- keep the QAT continuation approach
- improve the `int4` recipe itself with grouped scales and an `int4`-specific clipping knob

## Why This Is The Right Next Batch

The current `int4` path uses one scale for an entire row.
That is cheap, but it is also a blunt approximation for large MLP matrices.

We have about `0.5 MB` of artifact headroom relative to the dense control, so we can afford to spend some bytes on better low-bit fidelity if it buys real post-export score.

The highest-value questions are:

1. does grouped `int4` scaling improve the current best low-bit checkpoint even without retraining
2. if yes, which group size is the better trade: `128` or `64`
3. once the exporter improves, does another short QAT adaptation tail recover more quality than plain continuation on the old rowwise recipe
4. does a slightly tighter `int4` clip percentile help the grouped recipe further

## Planned Runs

### run46_export_group128_from_run45

- export-only from `run45_qat_tail_mid2_fc_extend4`
- `INT4_GROUP_SIZE=128`
- goal: quick read on whether grouped scales help at a modest metadata cost

### run47_export_group64_from_run45

- export-only from `run45_qat_tail_mid2_fc_extend4`
- `INT4_GROUP_SIZE=64`
- goal: test a finer grouped recipe with higher fidelity and slightly larger scale metadata

### run48_qat_groupbest_mid2_resume_run45

- resume `run45` with the better grouped setting from `run46/run47`
- `200`-step low-LR QAT continuation
- goal: adapt the best low-bit branch to the new grouped exporter

### run49_qat_groupbest_mid2_clip_tune

- same grouped setting as `run48`
- alter `INT4_CLIP_PERCENTILE` to a tighter value such as `99.99`
- goal: test whether the grouped recipe still benefits from less outlier-driven scale inflation

### run50_promote_best_batch9

- continue only the strongest batch-9 recipe if it clearly beats the old rowwise `run45` line
- otherwise stop and record that grouped scales were not enough

## Promotion Rules

- primary metric: lower `final_int8_zlib_roundtrip_exact val_bpb`
- hard gate: total submission size `<= 16,000,000`
- tie-break 1: smaller total submission size if within `0.003 val_bpb`
- tie-break 2: lower code complexity if still tied

## Success Criteria

Batch 9 is successful if it does at least one of these:

- improves the low-bit branch without changing the target set
- shows that grouped scales are the right way to spend low-bit artifact headroom
- identifies a better clip setting for the grouped recipe
- proves that the next step must be a deeper recipe change rather than more continuation

## Results

| Run | Executed change | Final roundtrip val_bpb | Total submission size int8+zlib | Outcome |
| --- | --- | --- | --- | --- |
| run46_export_group128_from_run45 | export-only from `run45`, grouped `int4` scales with `INT4_GROUP_SIZE=128` | `1.46727119` | `15,484,582` | Effectively flat versus `run45`; grouped export alone is not a free win |
| run47_export_group64_from_run45 | export-only from `run45`, grouped `int4` scales with `INT4_GROUP_SIZE=64` | `1.46727684` | `15,484,663` | Also flat and slightly worse than `128` on both score and size |
| run48_qat_group128_mid2_resume_run45 | `200`-step grouped-QAT continuation from `run45` with `INT4_GROUP_SIZE=128` | `1.46518465` | `15,485,052` | Real improvement; grouped scales need adaptation, not just export |
| run49_qat_group128_mid2_clip9999 | `200`-step grouped-QAT continuation from `run48` with `INT4_CLIP_PERCENTILE=99.99` | `1.46335568` | `15,484,940` | Another real improvement; tighter clipping helped |
| run50_qat_group128_mid2_clip9999_extend | `200`-step lower-LR continuation of the grouped+clipped recipe | `1.46225881` | `15,484,940` | Best batch-9 result and new best low-bit run overall |

## What This Revealed

- `run46` and `run47` ruled out the cheap path. Grouped scales do not help meaningfully without re-adapting the model to the new exporter.
- `run48` showed that grouped scales are still the right direction once QAT is allowed to react. Moving from the old rowwise `run45` recipe to grouped `128` improved the exported score from `1.46727031` to `1.46518465`.
- `run49` showed that the grouped recipe benefits from its own clip setting. Tightening `INT4_CLIP_PERCENTILE` to `99.99` improved the exported score again to `1.46335568`.
- `run50` confirmed that the grouped+clipped recipe still has some continuation headroom. It improved the exported score further to `1.46225881`.

## Batch 9 Decision

Batch 9 was successful.

It improved the best low-bit line from the rowwise `run45` result:

- from `1.46727031`
- to `1.46225881`

That is not enough to threaten the dense/int8 control, but it is a real recipe win and it validates the batch-9 hypothesis:

- grouped `int4` scales are useful
- but only after QAT adapts to them
- and the grouped recipe wants a tighter clip than the inherited rowwise default

## Next Direction

The low-bit branch is still behind the dense/int8 control, but it is now improving through recipe changes rather than only through more continuation.

The next sensible directions are:

- one more grouped+clipped continuation only if we want to chase a tiny incremental gain
- or a new recipe branch on top of the grouped `128` + `99.99` clip baseline, such as a different target set or a different scale/quantization parameterization
