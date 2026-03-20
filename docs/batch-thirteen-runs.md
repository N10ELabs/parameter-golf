# Batch 13: Minimal-Invasive Targets

## Summary

Batch 12 showed that the low-bit recipe can legalize `run28`, but the
`blocks.6.mlp.fc.weight` target was too harmful. The right next move was not
more QAT on the same target. It was a search for the smallest tensor that buys
the required byte savings with the least score loss.

The key observation:

- `run28` only needed about `24.8 KB` of savings to become legal

That means a much smaller tensor than a full `mlp.fc.weight` can solve the
artifact problem.

## Why This Was The Right Next Batch

If the goal is to make `run28` legal while preserving its score, the best
target is not the biggest matrix.

It is the least important matrix that still saves enough bytes.

That makes later decoder attention projections attractive:

- smaller than `mlp.fc.weight`
- likely less quality-sensitive than a large MLP expansion
- still big enough to buy the `24.8 KB` we need

## Results

| Run | Executed change | Final roundtrip val_bpb | Total submission size int8+zlib | Outcome |
| --- | --- | --- | --- | --- |
| run65_export_group16_ck_block6_from_run28 | export-only, `blocks.6.attn.c_k.weight`, `group16`, clip `99.9` | `1.35528972` | `15,967,287` | First near-control legal result |
| run66_export_group16_cv_block6_from_run28 | export-only, `blocks.6.attn.c_v.weight`, `group16`, clip `99.9` | `1.35566902` | `15,966,550` | Worse than `c_k` |
| run67_export_group16_ck_block5_from_run28 | export-only, `blocks.5.attn.c_k.weight`, `group16`, clip `99.9` | `1.35533697` | `15,967,425` | Slightly worse than block 6 |
| run68_export_group16_ck_block7_from_run28 | export-only, `blocks.7.attn.c_k.weight`, `group16`, clip `99.9` | `1.35522728` | `15,967,446` | Better than blocks 5 and 6 |
| run69_export_group8_ck_block7_from_run28 | same target as `run68`, but `group8` | `1.35522869` | `15,967,587` | No meaningful gain over `group16` |
| run70_export_group8_ck_block6_from_run28 | same target as `run65`, but `group8` | `1.35527953` | `15,967,425` | No meaningful gain over `group16` |
| run71_qat_only_ck_block7_from_run28 | target-only QAT on `blocks.7.attn.c_k.weight` | `1.35522815` | `15,967,453` | Target-only QAT did not help |
| run72_export_group16_ck_block8_from_run28 | export-only, `blocks.8.attn.c_k.weight`, `group16`, clip `99.9` | `1.35514585` | `15,968,000` | New leader at the time |
| run73_export_group16_ck_block9_from_run28 | export-only, `blocks.9.attn.c_k.weight`, `group16`, clip `99.9` | `1.35512322` | `15,967,854` | Best target-family result |
| run74_export_group16_ck_block10_from_run28 | export-only, `blocks.10.attn.c_k.weight`, `group16`, clip `99.9` | `1.35601409` | `15,968,121` | Final block was clearly worse |
| run75_qat_ck_block9_from_run28 | full-model QAT adaptation on the `run73` target | `1.35538168` | `15,967,941` | Full-model QAT also hurt this branch |
| run76_export_group16_ck_block9_clip9999_from_run28 | same target as `run73`, clip `99.99` | `1.35511732` | `15,967,871` | Slight improvement over `99.9` |
| run77_export_group16_ck_block9_clip995_from_run28 | same target as `run73`, clip `99.5` | `1.35511435` | `15,967,778` | Best batch-13 result |
| run78_export_group16_ck_block9_clip990_from_run28 | same target as `run73`, clip `99.0` | `1.35513788` | `15,967,688` | Slight regression from `99.5` |

## What This Revealed

- The “minimal-invasive” idea was correct. Quantizing a single `c_k` matrix is
  far less harmful than quantizing a whole `mlp.fc.weight`, while still buying
  enough bytes to legalize `run28`.
- The best target family is not MLP at all. It is the key projection in the
  late decoder.
- The best target location in this sweep was `blocks.9.attn.c_k.weight`.
- `group16` was already fine-grained enough for this smaller target. `group8`
  did not help meaningfully.
- QAT did not help this branch. Both target-only and full-model QAT tails were
  slightly worse than export-only on the best attention target.
- Clip still mattered a little. `99.5` was the best value tested for the
  winning `blocks.9.attn.c_k.weight` target.

## Batch 13 Decision

Batch 13 found the strongest legal low-bit variant of the `run28` frontier:

- `run77_export_group16_ck_block9_clip995_from_run28`
- quantize only `blocks.9.attn.c_k.weight`
- `INT4_GROUP_SIZE=16`
- `INT4_CLIP_PERCENTILE=99.5`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.35511435`
- total submission size int8+zlib: `15,967,778`

This is the closest low-bit run so far to the dense/int8 control
`run30_depth11_1218_codecut` at `1.35492895`.

The remaining gap is only about `0.000185 val_bpb`.

## Next Direction

The remaining frontier is now very clear.

The strongest next options are:

- keep the `run28 + blocks.9.attn.c_k.weight + group16 + clip99.5` recipe and
  search nearby attention targets or mixed tiny-target combinations
- or shift back to code-size golf, because `run28` itself is only `24,770`
  bytes over the cap and already beats `run30`
