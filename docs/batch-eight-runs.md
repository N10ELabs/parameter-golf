# Batch 8: Compression Frontier

## Summary

Batch 7 gave us a clean new control point:

- `run30_depth11_1218_codecut`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.35492895`
- total submission size int8+zlib: `15,992,456`

That result changes the search again.

We are no longer blocked by model shape or stopping point.
We are now blocked by how efficiently we serialize the same model.

The next batch should therefore focus on compression engineering that stays
inside the competition rules:

- fully self-contained artifact
- no external downloads or calibration data during evaluation
- decode fast enough to preserve evaluation practicality
- keep code growth smaller than the model-byte savings it buys

## Why This Is The Right Next Batch

The current export path in [train_gpt.py](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/train_gpt.py) is strong but still simple:

- mostly symmetric per-row `int8`
- `fp16` per-row scales
- selected small tensors kept in `fp16` or `fp32`
- `torch.save(...)` followed by `zlib`

That means there are still several realistic gains available without changing
the model family:

1. scale storage is probably too expensive
2. some tensors are likely over-quantized or under-quantized uniformly
3. the current path does not yet exploit 4-bit packing
4. the model is optimized for post-training quantization, but not explicitly for lower-bit export

So batch 8 should answer:

- can we shrink the export meaningfully without retraining
- can mixed-precision export beat uniform `int8`
- do a few sensitive exceptions buy enough quality to make low-bit export viable
- if plain post-training quantization is too weak, does a short quantization-aware tail recover the loss

## Rules

Keep these fixed unless a run explicitly changes them:

- `NUM_LAYERS=11`
- `MODEL_DIM=512`
- `NUM_KV_HEADS=4`
- `ITERATIONS=1218` as the control schedule
- `sp1024` tokenizer
- no external data use during evaluation
- no external libraries added just for compression
- export must still be implemented in [train_gpt.py](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/train_gpt.py)

Use this base command on the pod:

```bash
source .venv/bin/activate

DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 \
WARMUP_STEPS=5 \
TRAIN_LOG_EVERY=20 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=0 \
ITERATIONS=1218 \
python train_gpt.py
```

## Experimental Strategy

This batch is ordered by risk.

Start with export-only changes that cost little code and no extra training
signal.
Then move to mixed-precision export.
Only after that try a short quantization-aware tail if the lower-bit path still
hurts score too much.

That sequencing matters because the challenge counts both code bytes and model
bytes. A fancy exporter that saves `20 KB` of model bytes but adds `25 KB` of
code is a net loss.

## Runs

### run31_scale_quant_only

- Goal: test whether scale coding alone buys meaningful bytes with negligible score loss
- Change:
  - keep weight values in the current `int8` format
  - replace raw `fp16` per-row scales with a smaller representation, ideally quantized scales with one tensor-level second-order scale
- Expected value:
  - low implementation risk
  - should preserve most of the current score
  - tells us whether metadata, not weights, is the easiest remaining win

### run32_mixed_int4_int8_ptq

- Goal: test whether the largest 2D matrices can move to packed `int4` while the sensitive tensors stay at `int8` or float
- Change:
  - keep embeddings and control tensors on the existing safe path
  - apply packed `int4` only to selected large matrices
  - preserve per-row or per-group scaling
- Expected value:
  - highest likely byte savings in the batch
  - moderate quality risk
  - most important direct test of whether the current winner is over-serialized

### run33_outlier_aware_mixed_precision

- Goal: test whether a tiny high-precision carveout makes low-bit export viable
- Change:
  - start from `run32`
  - keep a small structured subset at `int8` or `fp16`
  - choose the carveout by a cheap heuristic such as largest row norm, max abs weight, or known-sensitive tensor names
- Expected value:
  - inspired by outlier-aware quantization
  - should recover quality more cheaply than reverting full tensors to high precision

### run34_qat_tail_lowbit

- Goal: test whether a short quantization-aware fine-tuning tail can recover the post-compression loss of the lower-bit exporter
- Change:
  - use the best low-bit exporter from `run32` or `run33`
  - add only a short final training tail under fake quantization, for example the last `100-300` steps
- Expected value:
  - higher implementation risk
  - strongest path if plain post-training low-bit export loses too much score
  - keeps the rest of the training recipe unchanged

### run35_promote_best_compressor

- Promotion pool: `run31`, `run32`, `run33`, `run34`
- Primary metric: lower `final_int8_zlib_roundtrip_exact val_bpb` or equivalent final post-export score
- Hard gate: total submission size must stay `<= 16,000,000`
- Tie-break 1: smaller total submission size if within `0.003 val_bpb`
- Tie-break 2: lower added code bytes if still tied
- Goal: pick the best compression path as the new control for the next batch

## What To Measure

For every run, record:

- final `train_loss`
- final `val_loss`
- final `val_bpb`
- final post-export `val_loss`
- final post-export `val_bpb`
- model bytes before and after compression
- total submission bytes
- code size delta in `train_gpt.py`
- step time impact, if any
- decode/eval overhead if the new exporter materially changes load time

## Success Criteria

Batch 8 is successful if it does at least one of these:

- finds a smaller export with no meaningful score regression
- recovers enough bytes to reopen the stopping-point search above `1218`
- proves that mixed low-bit export is viable for this model family
- identifies that the next real move must be quantization-aware training rather than serializer engineering alone

## Educational Read

This batch is about a different kind of optimization than the earlier ones.

The first seven batches mostly answered:

- what model family should we train
- how long should we train it

Batch 8 asks:

- how should we store the model we already know we want

That is a core parameter-golf lesson.
The best model is not just the one that trains well.
It is the one that survives serialization most efficiently.

## Briefing

The current best valid run is excellent, but it is still using a fairly conservative export path.

Batch 8 should tell us whether:

- better serialization is enough on its own
- mixed low-bit export is the next frontier
- or the model must be trained to expect the final compressor

## Results So Far

These first three runs were evaluated as export-only experiments on the saved
`run30_depth11_1218_codecut` checkpoint. That was intentional: for serializer
changes, reusing the same weights isolates the storage path more cleanly than
rerunning training.

| Run | Executed change | Final roundtrip val_bpb | Total submission size int8+zlib | Outcome |
| --- | --- | --- | --- | --- |
| run31_scale_quant_only | log-quantized per-row scales, same `int8` weights | `1.81027706` | `15,995,663` | Failed badly: worse score and no meaningful size win |
| run32_mixed_int4_int8_ptq | packed `int4` for both `mlp.fc.weight` and `mlp.proj.weight` | `2.24513555` | `10,282,161` | Huge size win, but catastrophic quality loss |
| run33_outlier_aware_mixed_precision | packed `int4` only for `mlp.fc.weight`, keep `mlp.proj.weight` on safer path | `2.01473719` | `13,134,281` | Quality recovered somewhat, but still far too weak |
| run34_qat_tail_lowbit | `200`-step low-LR QAT tail from `run30`, fake `int4` on `mlp.fc.weight` during training, export with same mixed path | `1.55880293` | `13,138,419` | Strong recovery relative to PTQ-only, but still well behind the `int8` control |
| run35_qat_tail_mid5_fc | `200`-step low-LR QAT tail from `run30`, fake `int4` only on middle `5` `mlp.fc.weight` layers, export with same mixed path | `1.52494300` | `14,706,667` | Best low-bit result so far at that point; gentler targeting helped a lot |
| run36_qat_tail_mid5_fc_frozen | same middle-`5` target set as `run35`, but freeze all non-target params during the QAT tail | `1.77476178` | `14,706,446` | Clearly worse; freezing non-target params hurt adaptation badly |
| run37_qat_tail_mid3_fc | `200`-step low-LR QAT tail from `run30`, fake `int4` only on middle `3` `mlp.fc.weight` layers | `1.51891396` | `15,224,119` | Narrower targeting helped again and slightly beat `run35` |
| run38_qat_tail_mid3_fc_extend | continue `run37` for another `200` low-LR QAT steps with the same middle-`3` target set | `1.48744380` | `15,225,168` | First continuation win; low-bit branch kept improving materially |
| run39_qat_tail_mid3_fc_extend2 | continue `run38` for another `200` lower-LR QAT steps with the same middle-`3` target set | `1.47664042` | `15,225,480` | Best low-bit result so far; continuation remains the strongest recipe |
| run40_export_mid2_from_run39 | export-only probe from the `run39` checkpoint, quantizing only `2` of the previously adapted middle layers | `1.47879984` | `15,481,611` | Slightly worse than exporting the original `3`-layer target set; export-mask-only changes are not enough |
| run41_qat_tail_mid2_fc | fresh `200`-step low-LR QAT tail from `run30`, fake `int4` on only `2` middle `mlp.fc.weight` layers | `1.51571360` | `15,482,627` | Stronger than the earlier first-step selective tails, but still behind the fully continued `3`-layer branch |
| run42_qat_tail_mid2_fc_extend | continue `run41` for another `200` lower-LR QAT steps with the same middle-`2` target set | `1.48479290` | `15,482,963` | First continuation win for the `2`-layer branch; now close to `run39` |
| run43_qat_tail_mid2_fc_extend2 | continue `run42` for another `200` lower-LR QAT steps with the same middle-`2` target set | `1.47407062` | `15,483,545` | New best low-bit result overall; the `2`-layer branch overtook the `3`-layer line |
| run44_qat_tail_mid2_fc_extend3 | continue `run43` for another `200` ultra-low-LR QAT steps with the same middle-`2` target set | `1.46943807` | `15,483,457` | Another real gain; the `2`-layer continuation line is still improving |
| run45_qat_tail_mid2_fc_extend4 | continue `run44` for another `200` even-lower-LR QAT steps with the same middle-`2` target set | `1.46727031` | `15,483,260` | New best low-bit result overall, but the gain is now clearly in the plateau regime |

## What This Revealed

- `run31` showed that serializer-only metadata tricks are not enough in this form. Quantized scale storage did not just fail on quality; it also failed to beat the existing blob size after code growth was counted.
- `run32` proved that this model family has enormous theoretical compressibility. The artifact dropped to about `10.24 MB`, so the size frontier is not the issue.
- `run33` showed that the failure is not simply “too many tensors went low-bit.” Even a conservative tensor-level carveout still left the post-export score far worse than the `int8` control.
- `run34` showed that QAT is directionally correct. A short low-LR tail improved the mixed low-bit exporter from `2.01473719` to `1.55880293`, which is a large recovery, but still not enough to make the current low-bit recipe competitive.
- `run35` showed that target selection matters almost as much as QAT itself. Moving from `11` low-bit `mlp.fc.weight` layers to only the middle `5` improved the exported score from `1.55880293` to `1.52494300` while still preserving more than `1.28 MB` of total artifact savings versus the `int8` control.
- `run36` ruled out the “freeze everything but the target weights” shortcut. The branch regressed to `1.77476178`, so the rest of the model needs to co-adapt during the QAT tail.
- `run37`, `run38`, and `run39` showed that a narrower middle-`3` target set plus continued low-LR QAT is the best low-bit recipe so far. That line improved from `1.51891396` to `1.48744380` to `1.47664042` while staying at about `15.23 MB` total artifact size.
- `run40` showed that we cannot cheaply “spend” the saved headroom just by changing the export mask on an already-adapted checkpoint. Exporting only `2` layers from the `run39` model was slightly worse than exporting the original adapted `3`-layer target set.
- `run41` reopened the smaller-target branch properly. A fresh middle-`2` QAT tail from the dense `run30` checkpoint reached `1.51571360`, which is a better first-step result than the earlier middle-`3` and middle-`5` starts, but it still trails the continued `run39` line.
- `run42`, `run43`, and `run44` showed that the middle-`2` branch has the better continuation frontier. It improved from `1.51571360` to `1.48479290`, then to `1.47407062`, then to `1.46943807`, and it now leads the low-bit search while staying at about `15.48 MB` total submission size.
- `run45` confirmed the shape of the curve. The branch still improved to `1.46727031`, but only by about `0.0022`, so further gains are now likely to be small continuation wins rather than major leaps.

## Batch 8 Decision So Far

The evidence now points in one direction:

- plain post-training low-bit export is not sufficient for the current model
- QAT does help substantially
- gentler low-bit targeting helps a lot
- continued low-LR QAT keeps paying off on the best selective target set
- but the present recipe is still not good enough to beat the existing `int8` exporter

In other words, if we want `int4`-class storage to be competitive here, the
model has to be trained to expect it, and the next refinement should probably
be either:

- one more continuation only if we are willing to chase tiny incremental gains
- or a more structural recipe change, because the low-bit branch is now improving mostly by small continuation gains rather than large target-set wins

The best low-bit result so far is now `run45_qat_tail_mid2_fc_extend4` at
`1.46727031` roundtrip `val_bpb` and `15,483,260` total submission bytes.
