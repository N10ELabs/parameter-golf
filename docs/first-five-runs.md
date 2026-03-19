# First Five Runs

This plan is meant to teach the repo, not chase leaderboard results yet.

The main idea is simple:

1. Fix most settings.
2. Change one thing at a time.
3. Record the outcome.
4. Learn which tradeoff moved: quality, speed, memory, or size.

## Ground Rules

For the first five runs:

- Use the same tokenizer: `sp1024`
- Use the same 1-shard training dataset you already downloaded
- Use the same validation setup
- Prefer one-variable changes over "clever" multi-change jumps
- Judge runs mostly by relative movement, not absolute score

Important note:

- Short runs are good for learning direction.
- Short runs are not good proxies for final compressed artifact size.
- Early in training, weights are still easy to compress, so `int8+zlib` size can
  look much better than it will after a serious run.

## Fixed Base Command

Run all experiments from `/workspace/parameter-golf` on the H100 pod:

```bash
source .venv/bin/activate

RUN_ID=<replace_me> \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=200 \
WARMUP_STEPS=5 \
TRAIN_LOG_EVERY=20 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=0 \
python train_gpt.py
```

## Metrics To Record

For every run, record:

- final `train_loss`
- final `val_loss`
- final `val_bpb`
- `final_int8_zlib_roundtrip_exact val_bpb`
- `Serialized model int8+zlib` bytes
- `peak memory allocated`
- `step_avg` milliseconds
- your own note on whether the result was expected

## Run 1: Baseline Calibration

Goal:

- Learn the default behavior
- Confirm you can read the logs
- Set your reference point for all later runs

Change:

- None

Command:

```bash
source .venv/bin/activate

RUN_ID=run01_baseline \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=200 \
WARMUP_STEPS=5 \
TRAIN_LOG_EVERY=20 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=0 \
python train_gpt.py
```

What to learn:

- What a healthy training curve looks like
- How fast the baseline is on 1xH100
- Rough memory usage at default settings

## Run 2: Width Sweep

Hypothesis:

- A wider model may improve early validation more than the baseline, but it will
  likely cost more time per step and more memory.

Change:

- `MODEL_DIM=640`

Command:

```bash
source .venv/bin/activate

RUN_ID=run02_width640 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MODEL_DIM=640 \
ITERATIONS=200 \
WARMUP_STEPS=5 \
TRAIN_LOG_EVERY=20 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=0 \
python train_gpt.py
```

What to compare against Run 1:

- Did `val_bpb` improve?
- How much slower did `step_avg` get?
- How much higher did memory go?

## Run 3: Depth Sweep

Hypothesis:

- More layers may help representation quality, but could be less efficient than
  adding width for the same short-run budget.

Change:

- `NUM_LAYERS=12`

Command:

```bash
source .venv/bin/activate

RUN_ID=run03_depth12 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=12 \
ITERATIONS=200 \
WARMUP_STEPS=5 \
TRAIN_LOG_EVERY=20 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=0 \
python train_gpt.py
```

What to compare:

- Width vs. depth on the same short budget
- Whether more layers help enough to justify extra step time

## Run 4: KV Head Sweep

Hypothesis:

- Reducing KV heads may save parameters and change the quality/speed tradeoff in
  a useful way, especially in a parameter-limited challenge.

Change:

- `NUM_KV_HEADS=2`

Command:

```bash
source .venv/bin/activate

RUN_ID=run04_kv2 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_KV_HEADS=2 \
ITERATIONS=200 \
WARMUP_STEPS=5 \
TRAIN_LOG_EVERY=20 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=0 \
python train_gpt.py
```

What to compare:

- Did fewer KV heads help or hurt early `val_bpb`?
- Did the speed or memory change enough to matter?

## Run 5: Extend The Most Promising Run

Goal:

- Learn whether the short-run winner still looks good when trained longer
- Get a slightly more meaningful view of compressed size and validation trend

Change:

- Pick the best run from 1-4 by `final_int8_zlib_roundtrip_exact val_bpb`
- Rerun that exact config for `ITERATIONS=500`

Command pattern:

```bash
source .venv/bin/activate

RUN_ID=run05_extend_best \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
<copy_the_best_extra_settings_here> \
ITERATIONS=500 \
WARMUP_STEPS=5 \
TRAIN_LOG_EVERY=25 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=0 \
python train_gpt.py
```

What to learn:

- Whether early ranking is stable
- Whether the run still improves smoothly
- Whether compressed size behavior changes as weights move away from init

## How To Read The First Five Runs

After all five runs, ask:

- Which change helped `val_bpb` the most?
- Which change gave the best quality per unit of training time?
- Which change looked promising but probably needs more training?
- Which change was clearly not worth the cost?

If two runs are close, prefer the simpler interpretation:

- "Width helped more than depth on short runs"
- "KV head reduction hurt quality"
- "Best short-run config stayed best at 500 steps"

That kind of conclusion is enough for your first round.
