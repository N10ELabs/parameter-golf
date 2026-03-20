# Batch 16: LZMA Cap-Edge Retraining

## Goal

Spend the headroom recovered in batch 15 on more model quality.

Batch 15 made the old `run28` dense frontier legal with about `97.8 KB` of
room left under the cap. That meant the old `1225`-step dense edge was no
longer the real limit, so batch 16 reopened the step-count search above `1225`.

## Runs

| Run | Purpose | Change | Final roundtrip val_loss | Final roundtrip val_bpb | Artifact bytes | Total bytes | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| run90_depth11_1250_lzma_p4 | First probe above the old dense edge | `11` layers, `1250` steps, dense `lzma+p4` packaging | 2.28538953 | 1.35353641 | 15988208 | 16013592 | Best raw score in the batch, but over by `13,592` bytes |
| run91_depth11_1246_lzma_p4 | Safer fallback near the new edge | `11` layers, `1246` steps, dense `lzma+p4` packaging | 2.28586673 | 1.35381904 | 15972748 | 15998132 | Legal and better than `run89` |
| run92_depth11_1247_lzma_p4 | Tight probe for the new legal edge | `11` layers, `1247` steps, dense `lzma+p4` packaging | 2.28490481 | 1.35324933 | 15974060 | 15999444 | New best valid run overall |
| run93_depth11_1248_lzma_p4 | Check whether the edge is one step higher | `11` layers, `1248` steps, dense `lzma+p4` packaging | 2.28531651 | 1.35349316 | 15984108 | 16009492 | Over by `9,492` bytes and slightly worse than `run92` |

## What We Learned

- The recovered `lzma` headroom was real enough to buy additional dense steps.
- `1250` was too aggressive, but it proved the new packaging path moved the
  frontier materially.
- `1246` was safely legal and already improved on `run89`.
- `1247` is the new best legal edge in this regime.
- `1248` was both over the cap and slightly worse than `1247`, so the immediate
  neighborhood is now mapped tightly enough.

## Outcome

Batch 16 produced the new best valid run overall:

- `run92_depth11_1247_lzma_p4`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.35324933`
- total submission size int8+lzma: `15,999,444`
- remaining headroom: `556` bytes

## Recommendation

Use `run92_depth11_1247_lzma_p4` as the new banked best-valid baseline.

Do not keep blindly stepping upward in this exact range:

- `1248` was already over and slightly worse
- `1250` was over by too much to be the first recovery target

The next batch should change a model-side lever on top of the `run92` regime,
not keep sweeping the same stop-point axis.
