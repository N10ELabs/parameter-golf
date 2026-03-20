# Batch Four Runs

Batch 4 is not a new wide search. It is a narrow cap-edge refinement pass
around the best current valid run:

- `NUM_LAYERS=12`
- `ITERATIONS=925`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.36010808`
- `Serialized model int8+zlib: 15,953,418` bytes

## Goal

Find the best valid stopping point just above `925` steps without drifting back
over the `16 MB` cap.

The reason this is worth doing is simple:

- `run11_depth12_950` was only `72,708` bytes over the cap
- that means the next real improvement is probably not architectural
- it is probably a fine-grained stopping-point win

## Ground Rules

- Keep the tokenizer fixed at `sp1024`
- Keep the 1-shard dataset fixed
- Keep validation fixed
- Keep `NUM_LAYERS=12`
- Do not open new optimizer or architecture branches in this batch
- Prefer tiny, defensible step-count moves over bigger speculative jumps

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

## Run 16: First Cap-Edge Probe

Hypothesis:

- `935` steps is the highest-value first probe because it is close enough to
  beat `925` on score if the size stays valid, but still conservative compared
  with `950`.

Override:

- `RUN_ID=run16_depth12_935`
- `ITERATIONS=935`

## Adaptive Follow-Up Rule

After `run16`:

1. If `run16` is valid and still has more than `10,000` bytes of headroom, run
   `936`.
2. If `run16` is invalid or too close to the cap, run `934`.
3. Compare the best valid result against `run12_depth12_925` by
   `final_int8_zlib_roundtrip_exact val_bpb`.
4. If the score difference is within `0.01`, prefer the smaller int8+zlib
   artifact.

## Metrics To Record

- final `train_loss`
- final `val_loss`
- final `val_bpb`
- `final_int8_zlib_roundtrip_exact val_loss`
- `final_int8_zlib_roundtrip_exact val_bpb`
- `Serialized model int8+zlib` bytes
- `peak memory allocated`
- `step_avg` milliseconds
