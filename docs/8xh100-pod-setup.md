# 8xH100 Pod Prep

This runbook prepares the `8xH100` environment without starting a training run.

It is split into two phases:

1. pre-launch setup you can do now
2. on-pod bootstrap you run later after the pod exists

## Goal

End up with:

- the right Runpod settings written down
- SSH ready before you pay for compute
- one bootstrap command ready for first boot
- one known-good `8 GPU` baseline command ready, but not executed

## Pre-Launch Checklist

### 1. Runpod account and access

- Confirm your Runpod account is active.
- Add your SSH public key in Runpod settings before creating the pod.
- Keep SSH terminal access enabled for the pod you eventually launch.

If you need a key on your Mac and do not already have one:

```bash
ssh-keygen -t ed25519 -C "runpod-parameter-golf"
cat ~/.ssh/id_ed25519.pub
```

Paste the public key into Runpod.

### 2. Pod settings to queue up

Use the repo's existing Runpod guidance as the base:

- launch from the official Parameter Golf template in [README.md](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/README.md)
- choose an `8xH100` GPU pod when you are ready to run track-relevant experiments
- enable SSH terminal access
- keep everything else close to the template defaults unless you already know you need a different disk/network layout

Track-relevant multi-GPU runs in this fork have been launched with:

- `torchrun --standalone --nproc_per_node=8`
- `NCCL_IB_DISABLE=1`
- `OMP_NUM_THREADS=1`
- `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`

Those settings are captured in the verified record at [records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md).

### 3. Decide where the repo will live on the pod

Use one location consistently. The existing repo docs use `/workspace/parameter-golf`.

Keep these paths in mind for the later run:

```bash
REPO_ROOT=/workspace/parameter-golf
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
```

### 4. Optional local SSH config entry

This avoids reconstructing the SSH command later. Fill in the host and port once Runpod shows them.

```sshconfig
Host parameter-golf-8xh100
  HostName <runpod-hostname>
  User root
  Port <runpod-ssh-port>
  IdentityFile ~/.ssh/id_ed25519
  ServerAliveInterval 30
  ServerAliveCountMax 10
```

Then later you can connect with:

```bash
ssh parameter-golf-8xh100
```

## First-Boot Bootstrap

After the pod exists and you SSH in, clone your fork or this repo into `/workspace`, then run:

```bash
cd /workspace/parameter-golf
bash scripts/bootstrap_runpod_8xh100.sh
```

That script does not launch training. It only:

- checks for `python3`, `torchrun`, and `nvidia-smi`
- confirms the visible GPU count
- downloads the cached `sp1024` FineWeb export if missing
- prints the exact baseline launch command for later use

## Baseline 8-GPU Command

Do not run this until you actually want the pod working.

```bash
OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
NCCL_IB_DISABLE=1 \
RUN_ID=baseline_sp1024_8gpu \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 /workspace/parameter-golf/train_gpt.py
```

This is aligned with the verified `8xH100` record command, but uses the `/workspace/parameter-golf` path convention from the root repo docs.

## Track-Faithful Dense Training Command

Use this when the goal is a leaderboard-style timed training run rather than a
research run with periodic validation.

```bash
OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
NCCL_IB_DISABLE=1 \
RUN_ID=track_dense_11l_slide64_lzma_p4_8gpu \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 \
EVAL_SEQ_LEN=1024 \
EVAL_STRIDE=64 \
STRICT_WALLCLOCK=1 \
MAX_WALLCLOCK_SECONDS=588 \
VAL_LOSS_EVERY=0 \
SKIP_POST_TRAIN_EVAL=1 \
INT4_NAME_PATTERNS= \
MODEL_COMPRESSOR=lzma \
MODEL_COMPRESS_PRESET=6 \
QUANT_PICKLE_PROTOCOL=4 \
QUANT_LOAD_WEIGHTS_ONLY=0 \
torchrun --standalone --nproc_per_node=8 /workspace/parameter-golf/train_gpt.py
```

Notes:

- This is intended to spend the training budget on learning, not on mid-run
  validation.
- `588` rather than `600` is deliberate. On the measured morning `8xH100`
  runs, `torchrun` startup consumed about `10-11` seconds of real wallclock
  before the trainer's internal timer started.
- Follow it with a separate export/eval run from the saved checkpoint.
- If you want richer curve monitoring, that is a research run, not a strict
  leaderboard-faithful timing rehearsal.

## Track-Faithful Separate Evaluation Command

Use this after the strict training run finishes and the checkpoint is saved.

```bash
OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
NCCL_IB_DISABLE=1 \
RUN_ID=track_dense_11l_eval_8gpu \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
INIT_MODEL_PATH=/workspace/parameter-golf/runs/<train-run-id>/final_model.pt \
ITERATIONS=0 \
WARMUP_STEPS=0 \
DISABLE_MODEL_COMPILE=1 \
SKIP_PREQUANT_EVAL_ZERO_ITERS=1 \
NUM_LAYERS=11 \
EVAL_SEQ_LEN=1024 \
EVAL_STRIDE=64 \
INT4_NAME_PATTERNS= \
MODEL_COMPRESSOR=lzma \
MODEL_COMPRESS_PRESET=6 \
QUANT_PICKLE_PROTOCOL=4 \
QUANT_LOAD_WEIGHTS_ONLY=0 \
torchrun --standalone --nproc_per_node=8 /workspace/parameter-golf/train_gpt.py
```

This keeps the evaluation budget separate from training and matches the
competition interpretation used in
[track-run-spec.md](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/docs/track-run-spec.md).

## Readiness Check Once The Pod Is Live

These are the only checks you need before the first real run:

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python3 - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
print("device_count=", torch.cuda.device_count())
PY
ls /workspace/parameter-golf/data/datasets/fineweb10B_sp1024 | head
```

Expected outcome:

- `8` visible CUDA devices
- tokenizer file present
- FineWeb shard files present

## Notes

- The root [README.md](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/README.md) only documents the `1xH100` path directly, so this file fills in the missing `8xH100` operational details for this fork.
- Keep the actual training launch separate from bootstrap. That makes it easier to stop after setup and avoid accidental spend.
- For the current timing interpretation and what the next serious runs should
  optimize for, see [track-run-spec.md](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/docs/track-run-spec.md).
