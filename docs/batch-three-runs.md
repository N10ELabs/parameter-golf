# Batch Three Runs

This batch turns the batch-2 findings into a cleaner search for the best valid
score under the `16 MB` submission cap.

The read going in is:

- `NUM_LAYERS=12` is still the best architecture base
- `TRAIN_BATCH_TOKENS=262144` is the most important new training lever
- the main risk is now artifact size, not raw optimization

## Goal

Learn two things without opening too many new variables:

1. What is the best pure-depth stopping point right below the size cap?
2. Can the smaller-batch depth regime beat the current valid winner while still
   staying under the cap?

## Ground Rules

- Keep the tokenizer fixed at `sp1024`
- Keep the 1-shard dataset fixed
- Keep validation fixed
- Do not revisit width, KV heads, or lower `MATRIX_LR` in this batch
- Judge runs first by `final_int8_zlib_roundtrip_exact val_bpb` under
  `16,000,000` int8+zlib bytes

## Fixed Base Command

Run all experiments from `/workspace/parameter-golf` on the H100 pod:

```bash
source .venv/bin/activate

DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=12 \
WARMUP_STEPS=5 \
TRAIN_LOG_EVERY=20 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=0 \
python train_gpt.py
```

## Why These Runs

`run10_depth12_1000` missed the cap by only `316,309` bytes, so the safest way
to harvest more valid score is to calibrate the stopping point instead of
changing architecture again.

At the same time, `run07_depth12_batch262k_eqtokens` was the strongest new
signal from batch 2, so it now deserves full promotion from diagnostic to
first-class search dimension.

## Run 11: Pure Depth Near-Cap Probe

Hypothesis:

- `950` steps should land very close to the cap while retaining most of the
  score gain from the `1000`-step run.

Override:

- `RUN_ID=run11_depth12_950`
- `ITERATIONS=950`

Expected role:

- Aggressive cap-edge candidate

## Run 12: Pure Depth Safety Bracket

Hypothesis:

- `925` steps should give a safer under-cap version of the same pure-depth run
  and help measure the score-size slope near the boundary.

Override:

- `RUN_ID=run12_depth12_925`
- `ITERATIONS=925`

Expected role:

- Safer cap calibration point

## Run 13: Smaller Batch Wallclock-Matched Promotion

Hypothesis:

- `TRAIN_BATCH_TOKENS=262144` at `600` steps is a fairer challenge-style test
  against the current `500`-step incumbent because the wallclock is similar.

Override:

- `RUN_ID=run13_depth12_batch262k_600`
- `TRAIN_BATCH_TOKENS=262144`
- `ITERATIONS=600`

Expected role:

- First head-to-head promotion test for the smaller-batch regime

## Run 14: Smaller Batch Long Push

Hypothesis:

- The smaller-batch regime may keep improving cleanly at `800` steps without
  hitting the cap.

Override:

- `RUN_ID=run14_depth12_batch262k_800`
- `TRAIN_BATCH_TOKENS=262144`
- `ITERATIONS=800`

Expected role:

- Main smaller-batch candidate

## Run 15: Smaller Batch Near-Cap Probe

Hypothesis:

- `900` steps should tell us whether the smaller-batch regime can reach the
  same score zone as the long pure-depth run while still remaining valid.

Override:

- `RUN_ID=run15_depth12_batch262k_900`
- `TRAIN_BATCH_TOKENS=262144`
- `ITERATIONS=900`

Expected role:

- Cap-edge small-batch candidate

## Comparison Rules

Use these promotion rules after all five runs:

1. Keep only runs with `Serialized model int8+zlib` bytes at or below
   `16,000,000`.
2. Among valid runs, the winner is the one with the lowest
   `final_int8_zlib_roundtrip_exact val_bpb`.
3. If two runs are within `0.01 val_bpb`, prefer the smaller int8+zlib model.
4. If still tied, prefer the lower `step_avg`.

Important comparisons:

- Compare `run11` and `run12` to `run05_extend_best` and `run10_depth12_1000`
- Compare `run13`, `run14`, and `run15` to `run05_extend_best`
- Do not compare over-cap runs as winners, but still record them because they
  map the score-size frontier

## Metrics To Record

For every run, record:

- final `train_loss`
- final `val_loss`
- final `val_bpb`
- `final_int8_zlib_roundtrip_exact val_loss`
- `final_int8_zlib_roundtrip_exact val_bpb`
- `Serialized model int8+zlib` bytes
- `peak memory allocated`
- `step_avg` milliseconds
- one sentence on what changed and what you learned

## Expected Batch-4 Base

- If `run11` or `run12` wins, keep pure depth as the base and refine stopping
  point near the cap
- If `run13`, `run14`, or `run15` wins, switch the base to
  `NUM_LAYERS=12` plus `TRAIN_BATCH_TOKENS=262144`
