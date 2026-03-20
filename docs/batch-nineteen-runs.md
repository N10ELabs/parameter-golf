# Batch 19: FP16 Embed Budget Map

## Summary

The next serious jump is unlikely to come from more serializer tweaking alone.
That win already happened with `lzma+p4`.

The most credible remaining export-side idea is upstream's strongest repeated
signal:

- keep `tok_emb.weight` in FP16

The problem is budget.

The current banked legal run, `run97_min_upstream_slide64_from_run28`, only has
about `91.6 KB` of remaining headroom, while FP16 tied embeddings likely cost
roughly `500 KB` more than the current dense `int8` path.

So batch 19 should not jump straight into a new exporter.
It should first map the byte budget precisely:

1. how much score does FP16 `tok_emb.weight` actually buy on our banked runs
2. how many bytes does it actually cost after compression
3. which currently supported low-bit or float carveouts are the best “payers”
   for that cost
4. whether the existing `int8` plus selective `int4` toolbox is already enough,
   or whether batch 20 must add a new intermediate precision mode

## Why This Is The Right Next Batch

The existing evidence already narrows the search:

- `1024/64` sliding eval is the best proven eval policy
- dense `int8` plus `lzma+p4` is the strongest banked legal path
- default mixed `int4` on big MLP targets is too destructive
- one-layer and tiny-target low-bit recipes can save meaningful bytes with much
  less damage than the old broad mixed-`int4` path

That means the next question is not “does low-bit work in general?”
It is:

- can we spend bytes on a high-value FP16 tensor and recover those bytes with a
  much gentler payer than the old MLP-heavy recipe

## Fixed Setup

Unless a run explicitly changes them:

- use saved checkpoints only
- do export-only evaluation
- use `EVAL_SEQ_LEN=1024`
- use `EVAL_STRIDE=64`
- use `MODEL_COMPRESSOR=lzma`
- use `MODEL_COMPRESS_PRESET=6`
- use `QUANT_PICKLE_PROTOCOL=4`
- use `QUANT_LOAD_WEIGHTS_ONLY=0`
- use `SKIP_PREQUANT_EVAL_ZERO_ITERS=1`
- keep `DISABLE_MODEL_COMPILE=1`

Primary source checkpoints:

- `run97_min_upstream_slide64_from_run28`
- `run92_depth11_1247_lzma_p4`

## Planned Runs

### run110_export_fp16_tokemb_from_run97

- checkpoint: `run97` source checkpoint
- change: `INT8_KEEP_FLOAT_LARGE_NAME_PATTERNS=tok_emb.weight`
- goal: measure the true compressed byte delta and score gain of FP16 tied
  embeddings on the strongest legal banked line

### run111_export_fp16_tokemb_from_run92

- checkpoint: `run92`
- change: same FP16 tied-embedding carveout as `run110`
- goal: compare the effect on the stronger screened dense checkpoint

### run112_export_ck9float_from_run97

- checkpoint: `run97` source checkpoint
- change: `INT8_KEEP_FLOAT_LARGE_NAME_PATTERNS=blocks.9.attn.c_k.weight`
- goal: re-measure the strongest tiny late-attention float carveout on the
  current `lzma+p4` path and treat it as a small-byte sensitivity reference

### run113_export_group16_block6_from_run97

- checkpoint: `run97` source checkpoint
- change:
  - `INT4_NAME_PATTERNS=blocks.6.mlp.fc.weight`
  - `INT4_GROUP_SIZE=16`
  - `INT4_CLIP_PERCENTILE=99.9`
- goal: measure how much byte budget the best historical one-layer low-bit
  payer buys when transferred onto the `run97` family

### run114_export_fp16tokemb_group16_block6_from_run97

- checkpoint: `run97` source checkpoint
- change: combine `run110` and `run113`
- goal: test whether the current code path can already fund FP16 tied
  embeddings with a one-layer low-bit payer

### run115_export_fp16tokemb_group16_block6_ck9float_from_run97

- checkpoint: `run97` source checkpoint
- change: combine FP16 `tok_emb.weight`, one-layer `blocks.6.mlp.fc.weight`
  `int4`, and the tiny late `c_k` float carveout
- goal: probe whether a mixed “spend on sensitive tensors, pay with one large
  low-bit tensor” recipe is already close enough to legal and competitive to
  avoid a new exporter family

## Promotion Rules

- primary metric: lower `final_int8_zlib_roundtrip_exact val_bpb`
- hard gate: total submission size `<= 16,000,000`
- tie-break 1: smaller total submission size if within `0.003 val_bpb`
- tie-break 2: prefer the simpler exporter if still tied

## Success Criteria

Batch 19 is successful if it does at least one of these:

- proves that FP16 `tok_emb.weight` gives a meaningful score gain on our banked
  runs
- finds a currently supported legal recipe that combines FP16 `tok_emb.weight`
  with a gentle payer
- shows quantitatively that the current `int8` plus `int4` toolbox cannot fund
  the FP16 embed idea cleanly
- produces exact byte accounting that justifies the batch-20 exporter work

## Decision Rule For Batch 20

If `run114` or `run115` is already legal and close enough to `run97`, batch 20
should stay small and refine that recipe.

If the combined recipe is still too damaging or still over budget by more than
about `100 KB`, batch 20 should add a new intermediate-precision exporter
rather than keep forcing the current `int4` path.
