# Batch 6: Budget Reallocation

## Summary

Batch 5 established a new best valid run:

- `run19_depth12_925_codecut`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.35971082`
- total submission size int8+zlib: `15,993,279`

That result changed the search again. We now know:

- the trimmed `train_gpt.py` plus hybrid serializer is the correct export path
- `925` is a better stop than `926` for the current `12 x 512` model
- another naive step-count bump is not the best next use of GPU time

So batch 6 should test a different idea:

- make the model slightly smaller
- spend the recovered artifact budget on more optimization steps
- see whether better-trained smaller models beat the current larger cap-edge model

## Rules

Keep these fixed for the whole batch:

- trimmed [train_gpt.py](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/train_gpt.py)
- hybrid int8 serializer
- `sp1024` tokenizer
- `NUM_KV_HEADS=4`
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
WARMUP_STEPS=5 \
TRAIN_LOG_EVERY=20 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=0 \
python train_gpt.py
```

## Runs

### run21_depth11_1100_codecut

- Override: `RUN_ID=run21_depth11_1100_codecut NUM_LAYERS=11 ITERATIONS=1100`
- Goal: test whether removing one layer and training longer beats the `12 x 512 @ 925` incumbent

### run22_dim480_1100_codecut

- Override: `RUN_ID=run22_dim480_1100_codecut MODEL_DIM=480 NUM_HEADS=8 ITERATIONS=1100`
- Goal: test whether a modest width cut buys a better size-to-quality trade than the depth cut

### run23_depth11_1400_codecut

- Override: `RUN_ID=run23_depth11_1400_codecut NUM_LAYERS=11 ITERATIONS=1400`
- Goal: push the depth-reduced branch farther if longer optimization is where the value is

### run24_dim480_1400_codecut

- Override: `RUN_ID=run24_dim480_1400_codecut MODEL_DIM=480 NUM_HEADS=8 ITERATIONS=1400`
- Goal: push the width-reduced branch farther on the same longer schedule

### run25_promote_best_reallocated

- Promotion pool: `run21`, `run22`, `run23`, `run24`
- Primary metric: lower `final_int8_zlib_roundtrip_exact val_bpb`
- Hard gate: total submission size int8+zlib must stay `<= 16,000,000`
- Tie-break 1: smaller total submission size if within `0.005 val_bpb`
- Tie-break 2: lower `step_avg` if still tied
- Override:
  - if the winner used `NUM_LAYERS=11`, run `ITERATIONS=1700`
  - if the winner used `MODEL_DIM=480`, run `ITERATIONS=1700`
- Goal: see whether the best smaller-model branch can turn saved artifact budget into a new best valid score

## Why These Runs

This batch is intentionally narrow. It tests two simple ways to free bytes:

1. One less layer
2. Slightly less width

And it tests each idea at two schedule lengths:

1. `1100` steps
2. `1400` steps

That keeps the batch educational:

- you learn whether depth reduction or width reduction is the cleaner budget-reallocation move
- you learn whether smaller models actually benefit more from longer training in this challenge
- you keep attribution clean because the changes are still easy to reason about

## Evaluation

Record for every run:

- final `train_loss`
- final `val_loss`
- final `val_bpb`
- final `final_int8_zlib_roundtrip_exact val_loss`
- final `final_int8_zlib_roundtrip_exact val_bpb`
- `Serialized model int8+zlib` bytes
- `Total submission size int8+zlib`
- `step_avg`
- peak memory

## Success Criteria

Batch 6 is successful if it answers at least one of these:

- a slightly smaller model trained longer beats `run19_depth12_925_codecut`
- one branch is clearly better at turning saved bytes into score
- we find a new model-size regime worth carrying to 8xH100 instead of staying locked on `12 x 512`

## Educational Read

This batch is about a core parameter-golf lesson:

- bigger is not always better
- smaller is not always better
- the best model is the one that spends the 16MB budget most efficiently after training and compression

## Results

### run21_depth11_1100_codecut

- roundtrip `val_bpb`: `1.35771622`
- int8+zlib model bytes: `15,433,606`
- total submission size int8+zlib: `15,473,960`
- result: new best valid run overall

### run22_dim480_1100_codecut

- roundtrip `val_bpb`: `1.37244609`
- int8+zlib model bytes: `11,274,049`
- total submission size int8+zlib: `11,314,403`
- result: much smaller, but clearly worse on score

### run23_depth11_1400_codecut

- roundtrip `val_bpb`: `1.34726355`
- int8+zlib model bytes: `16,665,279`
- total submission size int8+zlib: `16,705,633`
- result: best raw score in batch 6, but invalid by `705,633` bytes

### run24_dim480_1400_codecut

- roundtrip `val_bpb`: `1.36134672`
- int8+zlib model bytes: `12,179,721`
- total submission size int8+zlib: `12,220,075`
- result: valid, but still worse than `run21`

### run25_depth11_1700_codecut

- roundtrip `val_bpb`: `1.33858647`
- int8+zlib model bytes: `17,595,783`
- total submission size int8+zlib: `17,636,137`
- result: best raw score in the batch, but invalid by `1,636,137` bytes

## Batch 6 Decision

Batch 6 succeeded. It showed that the right new model regime is not “slightly narrower,” but “one layer shallower.”

The best valid run from the batch is:

- `run21_depth11_1100_codecut`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.35771622`
- total submission size int8+zlib: `15,473,960`

That means batch 6 produced a new best valid run overall, improving on `run19_depth12_925_codecut`.

The key lesson is:

- smaller width bought a lot of headroom but lost too much quality
- less depth was the better reallocation move
- pushing the winning 11-layer branch longer keeps improving score, but it eventually breaks the size cap just like the older 12-layer branch did
