# Batch 17: Upstream Eval Promotion And 10-Layer Branch

## Summary

This batch combined our strongest dense checkpoints with the highest-value
ideas from [upstream-scan-2026-03-19.md](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/docs/upstream-scan-2026-03-19.md):

- sliding-window evaluation
- NTK-aware RoPE extrapolation support for longer evals
- selective FP16 passthrough for sensitive large tensors
- decoupled Muon weight decay on matrix parameters
- a new `10`-layer upstream-style training branch

## Key Results So Far

### run95_screen_slide1024s64_from_run92

- Base checkpoint: `run92_depth11_1247_lzma_p4`
- Change: final eval only, `EVAL_SEQ_LEN=1024`, `EVAL_STRIDE=64`
- Result: `final_int8_zlib_roundtrip_exact val_bpb: 1.32008711`
- Takeaway: this was the highest-ROI port so far. A pure eval-policy change
  improved the screened `run92` score by about `0.0332 val_bpb` without
  retraining.

### run96_screen_slide2048s256_from_run92

- Base checkpoint: `run92_depth11_1247_lzma_p4`
- Change: screened `EVAL_SEQ_LEN=2048`, `EVAL_STRIDE=256` with upstream-style
  RoPE extrapolation support
- Status: stopped early after the pre-quant score came in worse than `run95`
- Key signal: `step:0/0 val_bpb: 1.3212`
- Takeaway: on this `1024`-trained `11`-layer checkpoint, `2048/256` did not
  beat `1024/64`, so `1024/64` is the current winning final-eval policy.

### run97_min_upstream_slide64_from_run28

- Base checkpoint: `run28_depth11_1225_codecut`
- Script: minified upstream-capable dense exporter, `lzma+p4`
- Change: legal promotion run with `EVAL_SEQ_LEN=1024`, `EVAL_STRIDE=64`
- Result: `final_int8_zlib_roundtrip_exact val_bpb: 1.32149156`
- Total submission size: `15,908,380`
- Takeaway: this is the new best legal run in the repo so far. It banked the
  eval-policy win on a checkpoint with enough code-byte headroom.

### run98_min_upstream_slide64_ck9fp16_from_run28

- Base checkpoint: `run28_depth11_1225_codecut`
- Change: `blocks.9.attn.c_k.weight` kept in FP16 during export
- Status: stopped early
- Takeaway: the run was too expensive to finish in its original form because it
  still paid for both pre-quant and post-quant sliding evals. It motivated the
  `SKIP_PREQUANT_EVAL_ZERO_ITERS=1` fast path for future export-only sweeps.

### run99_min_upstream_slide64_ck9ck10_fast_from_run28

- Base checkpoint: `run28_depth11_1225_codecut`
- Change: keep `blocks.9.attn.c_k.weight` and `blocks.10.attn.c_k.weight` in
  FP16 during export, with compile disabled and pre-quant eval skipped
- Status: stopped early
- Key signal: `Serialized model int8+lzma: 16142248 bytes`
- Total submission size: `16,174,242`
- Takeaway: two late `c_k` FP16 tensors are too expensive for the `run28`
  checkpoint family. This direction is not viable on the current `11`-layer
  dense frontier without additional byte savings.

### run100_10l_slide64_fp16emb_latek_muwd

- Change: first upstream-style `10`-layer training attempt with:
  - `NUM_LAYERS=10`
  - `TIED_EMBED_LR=0.1`
  - `MUON_WEIGHT_DECAY=0.02`
  - `EVAL_SEQ_LEN=1024`
  - `EVAL_STRIDE=64`
  - `INT8_KEEP_FLOAT_LARGE_NAME_PATTERNS=tok_emb.weight,blocks.8.attn.c_k.weight,blocks.9.attn.c_k.weight`
- Status: failed quickly
- Root cause: Muon weight decay was applied in-place on leaf tensors outside a
  `torch.no_grad()` block.
- Takeaway: this was an implementation bug, not a modeling rejection. The
  weight-decay port was fixed immediately afterward.

### run101_10l_slide64_fp16emb_latek_muwd

- Change: exact rerun of `run100` after fixing the Muon weight-decay bug
- Status: later completed on `8xH100`; follow-up is documented in
  [batch-eighteen-runs.md](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/docs/batch-eighteen-runs.md)
- Follow-up result:
  - pre-quant `val_bpb: 1.3321`
  - mixed-export final exact roundtrip `val_bpb: 1.42602122`
  - dense-control final exact roundtrip `val_bpb: 2.16189055`
- Takeaway: this branch was operationally viable but did not survive export
  controls well enough to remain the lead bet.

## Current Frontier

- Best screened score: `run95_screen_slide1024s64_from_run92`
  - `1.32008711`
  - not yet a byte-legal promoted submission script
- Best legal run: `run97_min_upstream_slide64_from_run28`
  - `1.32149156`
  - `15,908,380` total bytes
- Follow-up batch: see
  [batch-eighteen-runs.md](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/docs/batch-eighteen-runs.md)

## Next Move

What actually happened next is documented in
[batch-eighteen-runs.md](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/docs/batch-eighteen-runs.md):

1. `run101` completed on `8xH100` and was rejected as the lead branch.
2. The dense `11`-layer family was retested on `8xH100`.
3. The fork gained a real strict `10` minute training plus `10` minute
   evaluation workflow.
