# Batch 7: 11-Layer Cap-Edge Search

## Summary

Batch 6 established a new best valid run:

- `run21_depth11_1100_codecut`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.35771622`
- total submission size int8+zlib: `15,473,960`

It also established the new failure point:

- `run23_depth11_1400_codecut`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.34726355`
- total submission size int8+zlib: `16,705,633`

That means the next batch is straightforward:

- keep the new `11-layer` model regime fixed
- stop exploring width changes for now
- map the best legal stopping point between `1100` and `1400` steps

## Why This Is The Right Next Batch

The evidence from batch 6 is unusually clean:

- `11` layers is better than `12` layers under the submission cap
- `MODEL_DIM=480` is worse than the `11-layer` branch even when given more steps
- longer training on the `11-layer` branch keeps improving raw score, but eventually breaks the cap

So the highest-value work now is a cap-edge search, not another architecture sweep.

Using the batch-6 results, the rough linear estimate puts the legal edge around:

- `1228` steps
- estimated legal roundtrip `val_bpb` around `1.3533`

That estimate is only a guide. The point of batch 7 is to measure the real boundary.

## Rules

Keep these fixed for the whole batch:

- trimmed [train_gpt.py](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/train_gpt.py)
- hybrid int8 serializer
- `sp1024` tokenizer
- `NUM_LAYERS=11`
- `NUM_KV_HEADS=4`
- `MODEL_DIM=512`
- `TRAIN_BATCH_TOKENS=524288`
- `VAL_LOSS_EVERY=0`
- `VAL_BATCH_SIZE=524288`
- `MAX_WALLCLOCK_SECONDS=0`

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
python train_gpt.py
```

## Runs

### run26_depth11_1200_codecut

- Override: `RUN_ID=run26_depth11_1200_codecut ITERATIONS=1200`
- Goal: establish a safer improvement point above `1100` and recalibrate the size slope inside the winning regime

### run27_depth11_1250_codecut

- Override: `RUN_ID=run27_depth11_1250_codecut ITERATIONS=1250`
- Goal: test a likely-over-the-line checkpoint to bracket the legal edge from above

### run28_depth11_1225_codecut

- Override: `RUN_ID=run28_depth11_1225_codecut ITERATIONS=1225`
- Goal: test the estimated legal edge directly

### run29_depth11_1230_codecut

- Override: `RUN_ID=run29_depth11_1230_codecut ITERATIONS=1230`
- Goal: probe just above the interpolation estimate to see whether the boundary is slightly higher than expected

### run30_promote_best_capedge

- Promotion pool: `run26`, `run27`, `run28`, `run29`
- Primary metric: lower `final_int8_zlib_roundtrip_exact val_bpb`
- Hard gate: total submission size int8+zlib must stay `<= 16,000,000`
- Tie-break 1: smaller total submission size if within `0.003 val_bpb`
- Tie-break 2: lower `step_avg` if still tied
- Decision rule:
  - if `run29` is legal with more than `20,000` bytes of headroom, run `ITERATIONS=1235`
  - if `run29` is illegal, choose the best legal run below it and run the smallest upward refinement, usually `1227` or `1228`
- Goal: convert the bracketed cap edge into the best legal `11-layer` checkpoint

## Expected Learning

Batch 7 should answer:

- where the real legal edge of the `11-layer` branch sits
- how much of the `run23` raw-score gain can be kept legally
- whether the new best valid line is closer to `1100` or closer to the estimated `1228` edge

## Success Criteria

Batch 7 is successful if it does at least one of these:

- produces a new best valid run better than `1.35771622`
- maps the `11-layer` cap edge tightly enough that the next batch can move to compression or systems work
- proves that the current `1100` stop is already close to optimal under the legal byte budget

## Briefing

This batch is the natural follow-up to batch 6.

Batch 6 answered the model-shape question.
Batch 7 answers the stopping-point question.

If batch 7 works, the repo should end with:

- a stronger legal run than `run21`, or
- a much sharper understanding of the exact legal frontier for the new best model family

## Results

| Run | Executed change | Final roundtrip val_bpb | Total submission size int8+zlib | Outcome |
| --- | --- | --- | --- | --- |
| run26_depth11_1200_codecut | `ITERATIONS=1200` | `1.35605768` | `15,918,409` | Legal and better than `run21` |
| run27_depth11_1250_codecut | `ITERATIONS=1250` | `1.35364565` | `16,125,200` | Over cap by `125,200` bytes |
| run28_depth11_1225_codecut | `ITERATIONS=1225` | `1.35468228` | `16,024,770` | Over cap by `24,770` bytes |
| run29_depth11_1220_codecut | `ITERATIONS=1220` | `1.35491749` | `16,005,351` | Over cap by `5,351` bytes |
| run30_depth11_1218_codecut | `ITERATIONS=1218` | `1.35492895` | `15,992,456` | New best valid run overall |

## Execution Notes

- The original queue silently moved on to an obsolete `1230`-step branch after `run28`. Once `run28` showed the edge was already below `1225`, that stale process was terminated to avoid wasting more GPU time.
- After the bracket tightened, the useful refinement path was `1200 -> 1225 -> 1220 -> 1218`. That sequence was enough to locate the legal edge without spending another run on a clearly-invalid `1230`.
- `1219` was skipped on purpose. After `1220` missed by only `5,351` bytes, the measured byte slope made a one-step drop too risky to justify another near-duplicate probe.

## Outcome

- Batch 7 succeeded. It replaced `run21_depth11_1100_codecut` with a stronger legal checkpoint in the same `11-layer` regime.
- The new best valid run is `run30_depth11_1218_codecut` at `1.35492895` roundtrip val_bpb and `15,992,456` total submission bytes.
- The practical legal edge for this regime is now mapped tightly: `1220` is too large, `1218` is legal, and the score gain beyond `1200` is real but small enough that future work should probably move away from pure step-count search.
