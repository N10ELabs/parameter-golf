# Next Five Runs

This batch follows what the first five runs actually taught us.

The current read is:

- `NUM_LAYERS=12` is the strongest quality direction
- `NUM_KV_HEADS=2` is the strongest efficiency direction
- more width was not a good short-budget trade

The goal of batch 2 is to answer three questions:

1. Does depth still win when combined with fewer KV heads?
2. Does the deeper model want a slightly lower `MATRIX_LR`?
3. Does a smaller batch help when compared at equal token budget?

## Ground Rules

- Keep the tokenizer fixed at `sp1024`
- Keep the 1-shard dataset fixed
- Keep validation fixed
- Use `run07` only as a diagnostic, not as a promotion candidate
- Promote only between runs with the same effective comparison budget

## Fixed Base Command

Run all experiments from `/workspace/parameter-golf` on the H100 pod:

```bash
source .venv/bin/activate

DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
WARMUP_STEPS=5 \
TRAIN_LOG_EVERY=20 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=0 \
python train_gpt.py
```

## Run 6: Depth Plus KV Efficiency

Hypothesis:

- The depth winner may keep most of its quality while recovering some of the
  speed and memory benefits from fewer KV heads.

Override:

- `RUN_ID=run06_depth12_kv2`
- `NUM_LAYERS=12`
- `NUM_KV_HEADS=2`
- `ITERATIONS=200`

## Run 7: Smaller Batch At Equal Tokens

Hypothesis:

- A smaller batch may be slightly more sample-efficient for the deeper model,
  even if it is slower per unit of progress.

Override:

- `RUN_ID=run07_depth12_batch262k_eqtokens`
- `NUM_LAYERS=12`
- `TRAIN_BATCH_TOKENS=262144`
- `ITERATIONS=400`

Important:

- This run is diagnostic only because it uses a different step count to keep
  token budget comparable.

## Run 8: Lower Matrix Learning Rate

Hypothesis:

- The deeper model may prefer a slightly lower matrix learning rate and produce
  a better roundtrip score after quantization.

Override:

- `RUN_ID=run08_depth12_matrixlr003`
- `NUM_LAYERS=12`
- `MATRIX_LR=0.03`
- `ITERATIONS=200`

## Run 9: Promote The Best 200-Step Variant

Comparison set:

- Compare only `run06` and `run08`

Promotion rules:

1. Lower `final_int8_zlib_roundtrip_exact val_bpb`
2. If within `0.01`, smaller `Serialized model int8+zlib` bytes
3. If still tied, lower `step_avg`

If `run06` wins:

- `RUN_ID=run09_depth12_kv2_500`
- `NUM_LAYERS=12`
- `NUM_KV_HEADS=2`
- `ITERATIONS=500`

If `run08` wins:

- `RUN_ID=run09_depth12_matrixlr003_500`
- `NUM_LAYERS=12`
- `MATRIX_LR=0.03`
- `ITERATIONS=500`

## Run 10: Long Confirm

Comparison set:

- Compare `run09` against incumbent `run05_extend_best`

Promotion rules:

1. Lower `final_int8_zlib_roundtrip_exact val_bpb`
2. If within `0.01`, smaller `Serialized model int8+zlib` bytes
3. Do not promote any 500-step configuration above `14,500,000` int8+zlib bytes

If `run05_extend_best` wins:

- `RUN_ID=run10_depth12_1000`
- `NUM_LAYERS=12`
- `ITERATIONS=1000`

If `run09` wins with KV reduction:

- `RUN_ID=run10_depth12_kv2_1000`
- `NUM_LAYERS=12`
- `NUM_KV_HEADS=2`
- `ITERATIONS=1000`

If `run09` wins with lower matrix LR:

- `RUN_ID=run10_depth12_matrixlr003_1000`
- `NUM_LAYERS=12`
- `MATRIX_LR=0.03`
- `ITERATIONS=1000`

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
