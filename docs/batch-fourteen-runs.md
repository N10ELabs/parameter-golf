# Batch 14: Dense-Only Code Golf

## Goal

Use code-size reduction, not model changes, to legalize the strongest saved
dense checkpoints.

The key targets were:

- `run29_depth11_1220_codecut`: slightly better than `run30`, only modestly
  over the cap
- `run28_depth11_1225_codecut`: strongest dense checkpoint overall, but still
  over the cap

## What Changed

I generated a minified trainer candidate as
`/Users/anthonymarti/Desktop/N10E LABS Code/parameter-golf/train_gpt_min.py`
and patched the small syntax issues introduced by automated minification.

The decisive diagnostic finding was that the current repo had drifted into
default mixed `int4` export: even with `QAT_INT4=0`, the exporter still applied
`int4` to `mlp.fc.weight` and `mlp.proj.weight` unless `INT4_NAME_PATTERNS=`
was explicitly cleared. That was shrinking artifacts to about `10.25 MB` and
destroying the dense roundtrip score.

Once `INT4_NAME_PATTERNS=` was forced empty, the historical dense exporter was
restored exactly.

## Runs

| Run | Purpose | Change | Final val_loss | Final val_bpb | Final roundtrip val_loss | Final roundtrip val_bpb | Int8+zlib bytes | Total bytes | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run81b_current_from_run29 | Diagnose the current exporter on `run29` | current `train_gpt.py`, default export path | 2.2858 | 1.3538 | 3.04070474 | 1.80087662 | 10252545 | 10298830 | Revealed that default export was still using mixed `int4` and no longer matched the saved dense frontier |
| run81c_current_dense_from_run29 | Confirm the historical dense exporter | current `train_gpt.py`, `INT4_NAME_PATTERNS=` | 2.2858 | 1.3538 | 2.28772143 | 1.35491749 | 15964994 | 16011279 | Restored the exact historical `run29` dense result |
| run82_minified_dense_from_run29 | Legalize `run29` by code size alone | `train_gpt_min.py`, `INT4_NAME_PATTERNS=` | 2.2858 | 1.3538 | 2.28772143 | 1.35491749 | 15964994 | 15990340 | New best valid run overall |
| run83_minified_dense_from_run28 | Test whether the same code-golf path is enough for `run28` | `train_gpt_min.py`, `INT4_NAME_PATTERNS=` | 2.2851 | 1.3534 | 2.28732429 | 1.35468228 | 15984432 | 16009778 | Still over the cap by `9,778` bytes |

## What We Learned

- The dense frontier was still intact; the main issue was exporter drift, not a
  loss of the underlying checkpoint quality.
- `run29` is now legal and becomes the best valid dense run:
  `1.35491749` roundtrip `val_bpb` at `15,990,340` total bytes.
- `run28` remains the strongest dense checkpoint, and it is now only `9,778`
  bytes away from legality under the minified dense exporter.
- The next meaningful step is not another ML batch. It is a purpose-built
  dense-only submission script that removes the low-bit/QAT machinery cleanly.

## Recommendation

Use `run82_minified_dense_from_run29` as the new banked best valid baseline.

Then run one more dense-only code-golf pass aimed specifically at `run28`. The
target is to cut the effective code size from `25,346` bytes down to about
`15,568` bytes while preserving the restored dense exporter behavior.
