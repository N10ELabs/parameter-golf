# Batch 15: Serializer and Compressor Search

## Goal

Use packaging changes, not model changes, to turn the strongest saved dense
checkpoint into the best valid run overall.

After batch 14, the situation was:

- `run82_minified_dense_from_run29`: legal at `1.35491749`
- `run83_minified_dense_from_run28`: better score at `1.35468228`, but still
  over by `9,778` bytes

So batch 15 focused on the remaining artifact bytes.

## Approach

The dense quantizer stayed fixed. Every run used:

- `INT4_NAME_PATTERNS=` to preserve the dense exporter
- the same saved `run28` checkpoint

The only changes were:

1. Torch serialization format
2. pickle protocol
3. outer lossless compressor

## Runs

| Run | Purpose | Change | Final roundtrip val_loss | Final roundtrip val_bpb | Artifact bytes | Total bytes | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| run84_minified_dense_p5_from_run28 | Test newer pickle protocol | `pickle_protocol=5`, `zlib` | N/A | N/A | 15979730 | 16005094 | Better artifact size, but still `5,094` bytes over and blocked by `weights_only=True` |
| run85_minified_dense_oldser_from_run28 | Test old Torch serialization format | legacy Torch save, `zlib` | 2.28732429 | 1.35468228 | 15996414 | 16021797 | Worse than the default minified dense exporter |
| run86_minified_dense_oldser_p5_from_run28 | Combine old serialization with protocol 5 | legacy Torch save, `pickle_protocol=5`, `zlib` | N/A | N/A | 15987492 | 16012893 | Better than `run85`, still worse than plain `p5`, blocked by `weights_only=True` |
| run87_minified_dense_p4_from_run28 | Test protocol 4 instead of 5 | `pickle_protocol=4`, `zlib` | N/A | N/A | 15979727 | 16005091 | Best `zlib` artifact, still `5,091` bytes over and blocked by `weights_only=True` |
| run88_minified_dense_oldser_p4_from_run28 | Combine old serialization with protocol 4 | legacy Torch save, `pickle_protocol=4`, `zlib` | N/A | N/A | 15995519 | 16020920 | Worse than the best zip-based protocol variants |
| run89_minified_dense_lzma_p4_from_run28 | Swap the outer compressor on top of the best serializer | `pickle_protocol=4`, `lzma`, `weights_only=False` on reload | 2.28732429 | 1.35468228 | 15876848 | 15902232 | New best valid run overall |

## Offline Sweep

Before the full validation run, I measured several format and compression
combinations on the same dense quantized blob from `run28`.

Most important result:

- `pickle_protocol=4` + `lzma` preset `6`: `15,876,848` bytes

That was about `102.9 KB` smaller than the best `zlib` path on the same raw
payload, with decompression still under a second in the quick measurement.

## What We Learned

- Serializer choice mattered, but not enough by itself.
- `pickle_protocol=4` was the best `zlib`-based serializer variant.
- The remaining problem was solved by changing the outer compressor, not by
  more script golf.
- `lzma` preserved the exact dense score while creating comfortable artifact
  headroom.

## Outcome

Batch 15 produced the new best valid run overall:

- `run89_minified_dense_lzma_p4_from_run28`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.35468228`
- total submission size int8+lzma: `15,902,232`
- remaining headroom: `97,768` bytes

That changes the strategy:

- `run28` is no longer the best illegal dense checkpoint
- it is now the best valid dense checkpoint
- the next step is to spend the recovered headroom on a slightly stronger dense
  run above `1225` steps
