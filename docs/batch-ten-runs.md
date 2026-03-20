# Batch 10: Frontier Transfer

## Summary

Batch 9 proved that the low-bit branch still improves when we change the
recipe, not just when we keep extending the same rowwise `int4` setup.

The new best low-bit recipe is:

- middle `2` `mlp.fc.weight` layers
- grouped `int4` scales with `INT4_GROUP_SIZE=128`
- `INT4_CLIP_PERCENTILE=99.99`

That branch reached:

- `run50_qat_group128_mid2_clip9999_extend`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.46225881`

But the main leaderboard path is still the dense/int8 control:

- `run30_depth11_1218_codecut`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.35492895`

So batch 10 should ask a higher-value question than “can we shave another
`0.002` from `run50`?”

The best unused lever is the starting checkpoint.

The low-bit artifact size is mostly determined by model shape and serializer,
not by how many training steps produced the weights.
That means stronger but over-cap dense checkpoints can still be legal once
stored with the low-bit recipe.

## Why This Is The Right Next Batch

We already have stronger dense frontiers saved on the pod:

- `run23_depth11_1400_codecut` with roundtrip `val_bpb: 1.34726355`
- `run25_depth11_1700_codecut` with roundtrip `val_bpb: 1.33858647`

Those checkpoints were invalid only because the conservative int8 serializer
put them over the total artifact cap.

The grouped low-bit branch sits around `15.48 MB`, leaving enough room to make
those same weights legal if the quantized score survives adaptation.

So batch 10 should answer:

1. can the best grouped low-bit recipe transfer to a better dense frontier
2. is `run25` too aggressive as a starting point, or is it actually the best
3. once the best frontier transfer is identified, does it still improve under a
   lower-LR continuation
4. if frontier transfer still underperforms, is spending more headroom on a
   one-layer target set the next step

## Planned Runs

### run51_qat_group128_mid2_clip9999_from_run23

- start from `artifacts/run23_depth11_1400_codecut.final_model.pt`
- same middle-`2` target set
- grouped `int4` with `INT4_GROUP_SIZE=128`
- `INT4_CLIP_PERCENTILE=99.99`
- fresh `200`-step QAT adaptation tail

Goal:
- moderate frontier transfer from a stronger dense checkpoint

### run52_qat_group128_mid2_clip9999_from_run25

- start from `artifacts/run25_depth11_1700_codecut.final_model.pt`
- same grouped+clipped recipe as `run51`
- fresh `200`-step QAT adaptation tail

Goal:
- aggressive frontier transfer from the best saved dense checkpoint

### run53_promote_best_frontier_transfer

- compare `run51` and `run52`
- primary metric: lower `final_int8_zlib_roundtrip_exact val_bpb`
- hard gate: total submission size `<= 16,000,000`
- tie-break 1: smaller total submission size if within `0.003 val_bpb`

If `run52` wins:
- continue `run52` with a lower-LR `200`-step tail

If `run51` wins:
- continue `run51` with a lower-LR `200`-step tail

Goal:
- test whether the winning frontier-transfer branch still has easy continuation
  gains

### run54_optional_one_layer_budget_reallocation

- only execute if the winning `run53` branch is still clearly behind the dense
  control and we need a new lever
- start from the best frontier-transfer checkpoint
- quantize only one of the two middle layers instead of both

Goal:
- spend more of the low-bit artifact headroom on fidelity rather than on
  compression

## Success Criteria

Batch 10 is successful if it does at least one of these:

- beats `run50` by a meaningful margin using a better dense starting point
- shows that a stronger dense frontier transfers cleanly into the grouped low-bit
  recipe
- identifies whether `run25` is too aggressive or actually the best launch point
- narrows the final gap enough that low-bit becomes a serious submission path

## Results

| Run | Executed change | Final roundtrip val_bpb | Total submission size int8+zlib | Outcome |
| --- | --- | --- | --- | --- |
| run51_qat_group128_mid2_clip9999_from_run23 | fresh grouped+clipped QAT adaptation from `run23_depth11_1400_codecut` | `1.52719524` | `16,194,832` | Failed badly; invalid and much worse than the current low-bit line |
| run52_export_group128_mid2_clip9999_from_run25 | export-only grouped+clipped evaluation from `run25_depth11_1700_codecut` | `1.82858883` | `17,115,813` | Catastrophic and invalid; aggressive frontier transfer is not viable in this form |
| run53_qat_group128_mid1_block5_from_run50 | spend headroom on fidelity by quantizing only `blocks.5.mlp.fc.weight` | `1.46015294` | `15,741,904` | Real improvement over the batch-9 winner and still legal |
| run54_qat_group128_mid1_block6_from_run50 | same one-layer idea, but quantize only `blocks.6.mlp.fc.weight` | `1.46007628` | `15,741,104` | Slightly better than `run53` on both score and size |
| run55_qat_group128_mid1_block6_extend | lower-LR continuation of the winning one-layer branch | `1.45908502` | `15,741,394` | Best batch-10 result and new best low-bit run overall |

## What This Revealed

- `run51` and `run52` were the key negative result. The best grouped low-bit recipe does not transfer cleanly to stronger dense frontiers in this form. Better dense checkpoints can become much less compressible under the low-bit exporter, and the result can be both invalid and worse on score.
- `run53` showed that the better way to use the low-bit artifact budget is not “stronger dense frontier,” but “more fidelity.” Quantizing only one of the two middle layers improved the exported score immediately.
- `run54` showed that the single-layer choice matters slightly. `blocks.6.mlp.fc.weight` beat `blocks.5.mlp.fc.weight` by a small but real margin.
- `run55` confirmed that the one-layer branch still has continuation headroom. The exported score improved again to `1.45908502` while staying well under the cap.

## Batch 10 Decision

Batch 10 was successful, but not for the original frontier-transfer hypothesis.

The original bet failed:

- stronger dense frontiers did not survive the grouped low-bit recipe

The replacement bet worked:

- use the grouped `128` + `99.99` clip recipe
- spend more of the remaining budget on fidelity
- quantize only `blocks.6.mlp.fc.weight`

That produced the new best low-bit run:

- `run55_qat_group128_mid1_block6_extend`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.45908502`
- total submission size int8+zlib: `15,741,394`

## Next Direction

The low-bit branch is now better than batch 9, but it is still behind the dense/int8 control.

The most sensible next directions are:

- one more continuation on the winning one-layer branch only if we want a small incremental gain
- or a recipe search on top of the new one-layer baseline, because that is now the strongest low-bit starting point
