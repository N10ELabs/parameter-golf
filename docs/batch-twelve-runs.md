# Batch 12: Gentle Frontier Transfer

## Summary

Batch 11 made the one-layer low-bit recipe much stronger:

- quantize only one tensor
- grouped `int4`
- `INT4_GROUP_SIZE=16`
- `INT4_CLIP_PERCENTILE=99.9`

That changed the next question.

Instead of asking whether low-bit can beat the dense/int8 control on the old
QAT branch, batch 12 asks whether the new gentler recipe can make stronger
dense checkpoints legal without destroying their score.

The best saved dense frontier checkpoints near the cap were:

- `run30_depth11_1218_codecut`: current best valid dense/int8 control,
  `1.35492895`
- `run29_depth11_1220_codecut`: slightly stronger dense run, but invalid by
  `5,351` bytes
- `run28_depth11_1225_codecut`: strongest of the near-cap dense runs,
  `1.35468228`, but invalid by `24,770` bytes

## Why This Was The Right Next Batch

The new one-layer low-bit recipe saves far more than `24,770` bytes.

So if transfer fidelity is good enough, `run28` and `run29` can become legal
with room to spare.

That makes this the highest-value post-batch-11 branch:

1. test whether the gentle one-layer recipe transfers cleanly to near-cap dense
   checkpoints
2. identify which dense frontier is the best launch point
3. only then decide whether QAT adaptation is worth the extra compute

## Results

| Run | Executed change | Final roundtrip val_bpb | Total submission size int8+zlib | Outcome |
| --- | --- | --- | --- | --- |
| run61_export_group16_block6_from_run30 | export-only low-bit transfer from `run30` using one-layer `blocks.6.mlp.fc.weight` | `1.36089601` | `15,742,440` | Legal, but too much transfer loss |
| run62_export_group16_block6_from_run29 | same transfer from `run29` | `1.36109898` | `15,754,967` | Legal, but worse than `run61` |
| run63_export_group16_block6_from_run28 | same transfer from `run28` | `1.36067705` | `15,773,187` | Best export-only dense transfer of the three |
| run64_qat_group16_block6_from_run28 | full-model `200`-step QAT adaptation from `run28` | `1.36001684` | `15,773,258` | Real improvement, but still not close enough to beat `run30` |

## What This Revealed

- The batch-11 recipe does transfer to strong dense checkpoints, but not for
  free. The transfer penalty was about `0.006-0.007 val_bpb`, which is small
  enough to be interesting but too large to beat the current dense/int8
  control on its own.
- `run28_depth11_1225_codecut` is still the best source checkpoint for this
  line. It remained the strongest of the dense frontier transfers.
- QAT adaptation helped, but only modestly. `run64` recovered about
  `0.00066 val_bpb`, which was not enough to justify staying with the large
  one-layer MLP target.

## Batch 12 Decision

Batch 12 was useful, but it changed the optimization target.

The lesson is not “transfer failed.”
The lesson is:

- the dense frontier is good enough
- the one-layer MLP target is too invasive for cap-edge transfer

So the next branch should use the same `run28` dense checkpoint, but look for a
smaller, less harmful tensor that still saves at least `24,770` bytes.

## Next Direction

Batch 13 should search “minimal-invasive” targets:

- smaller attention matrices instead of a whole `mlp.fc.weight`
- especially `c_k` or `c_v` matrices in later decoder blocks
- exporter-only sweeps first, because they are cheap and isolate target
  sensitivity cleanly
