# Parameter Golf Roadmap

Status: April upstream leaderboard scan complete. This is the single source of
truth for the project direction from here.

## Executive Decision

Pivot the main effort to the current official SP8192 leaderboard stack.

Do not spend primary compute on the old SP1024 dense-int8/LZMA/code-golf path.
Do not treat standalone Hadamard/rotation export as a mainline bet. Those ideas
can remain small diagnostics, but the leaderboard has moved to a different
regime.

The goal is now:

1. reproduce a known SP8192 record-family baseline;
2. port the required architecture, optimizer, compression, and eval pieces into
   this fork cleanly;
3. add one original contribution on top of that stack only after reproduction is
   credible.

## Current Gap

Official upstream has moved far beyond our local frontier.

| Source | Score | Notes |
| --- | ---: | --- |
| [SP8192 + 3-layer recurrence + parallel residuals + QK5.25 + legal TTT](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT) | `1.0810` | Current top leaderboard record from the April scan |
| [SP8192 + parallel residuals + score-first TTT](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-04-08_SP8192_ParallelResid_ScoreFirstTTT) | `1.0822` | Combines parallel residuals with legal TTT |
| [SP8192 + QK-gain 5 + legal score-first TTT](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-04-06_SP8192_QK5_LegalTTT_1.0828) | `1.0828` | Shows TTT is now table stakes near the top |
| [SP8192 + parallel residuals + Hessian-aware SDClip](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence) | `1.0835` | Best clue for an original compression-side angle |
| [SP8192 + GPTQ embeddings + SDClip + recurrence](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2) | `1.0856` | Clean base stack to reproduce before chasing the absolute top |
| Local best legal legacy run | `1.32149156` | `run97_min_upstream_slide64_from_run28`, SP1024 dense-int8/LZMA line |
| Local best screened legacy run | `1.32008711` | `run95_screen_slide1024s64_from_run92`, not finalized |
| Fresh 1xH100 legacy probe | `1.36447459` | dense int8 roundtrip; rotation probes were effectively flat |

The practical gap is about `0.24` BPB from our best local legal result to the
current official top score. That is too large for exporter tweaks, code golf, or
rotation-only compression to close.

## Upstream Evidence Consulted

Official repo snapshot at scan:

- `75700cb`: merge of PR `#1511`, April leaderboard README update.
- `bac888c`: merge of PR `#1493`, bigbag SP8192 TTT clean submission.
- `905ef58`: merge of PR `#1477`, aryanbhosale SP8192 parallel TTT
  submission.
- `6f92b13`: merge of PR `#1413`, dexhunter SP8192 QK5 legal TTT submission.
- `c714a4d`: merge of PR `#1412`, Robby955 parallel residuals plus
  Hessian-aware SDClip submission.
- `8d62bdd`: merge of PR `#1394`, Kevin Clark SP8192 GPTQ embeddings, SDClip,
  and Loop45x2 submission.

The commit sequence matters because the leaderboard is not a random collection
of independent tricks. The April submissions compound on one another:
SP8192/GPTQ/SDClip first, then QK gain and legal TTT, then parallel residuals,
then deeper recurrence and additional tuning.

## What Wins Now

The competitive recipe is no longer "train the original small model longer and
compress it harder." The current winning pattern is a coordinated stack.

Architecture:

- SP8192 tokenizer/data path.
- 11 layers, 512 model dimension, 8 attention heads, 4 KV heads.
- 4x MLP with `LeakyReLU(0.5)^2`.
- Partial RoPE.
- Tied embeddings.
- Depth recurrence, now including 3-layer recurrence in the top record.
- Parallel residuals starting in late layers.
- QK gain around `5.0` to `5.25`.

Training:

- MuonEq-R / row-normalized Muon for matrix parameters.
- AdamW for embeddings and scalar parameters.
- High weight decay around `0.095`.
- EMA around `0.9965`.
- Long warmdown, roughly `0.72` of the run in the top stack.
- Full leaderboard training still assumes `8xH100`; a single H100 is for
  smoke tests and ablations, not final claims.

Compression:

- GPTQ, not plain dense int8.
- SDClip for monotonic artifact/quality tuning.
- Int6 for attention and MLP matrices.
- Int8 GPTQ for token embeddings.
- Byte shuffle plus Brotli-11 style packaging.
- Artifact decisions judged by final legal roundtrip score and bytes, not by
  dense validation loss alone.

Evaluation:

- Legal score-first test-time training is now required to compete at the top.
- Score-before-update order matters for compliance.
- The recent records use chunked TTT with small SGD updates and stay inside the
  evaluation budget.

## What We Should Stop Doing

These are no longer mainline work:

- SP1024 cap-edge training sweeps.
- Dense int8 plus LZMA/code-golf optimization as a primary path.
- Mixed int4 rescue attempts on old checkpoints.
- Standalone Hadamard/rotation export as a primary path.
- More old-run documentation, batch planning, or promotion around `run92`,
  `run95`, `run97`, or `run106`.

The only useful legacy conclusions are:

- our old best legal score was `1.32149156`;
- the strict split training/eval workflow was operationally useful;
- final artifact roundtrip score matters more than dense pre-quant loss;
- rotation export was technically viable but not impactful on the tested
  checkpoint.

## Compute Policy

Use one H100 for:

- dependency setup and compilation checks;
- SP8192 dataset/tokenizer smoke tests;
- reduced-shard or reduced-step training sanity checks;
- exporter correctness checks;
- artifact-size and roundtrip-eval plumbing;
- small ablations that can reject broken ideas quickly.

Do not use one H100 for:

- claiming leaderboard-level results;
- judging whether the full SP8192 recipe is competitive;
- expensive sweeps of legacy SP1024 settings.

Use `8xH100` only after:

- the SP8192 code path runs end-to-end;
- exporter output is legal or predictably tunable under the cap;
- a known official stack has been approximately reproduced at small scale;
- the next run has a specific hypothesis and promotion criterion.

## Execution Plan

### Phase 1: Rebase The Base Stack

Start from the official OpenAI repo state rather than patching the old local
SP1024 code forward by hand.

Tasks:

- Fetch or refresh `https://github.com/openai/parameter-golf`.
- Compare this fork against upstream and identify local-only changes worth
  preserving.
- Port the current SP8192 dataset/tokenizer path.
- Make a clean branch whose first target is reproduction, not novelty.
- Keep the old dense-int8 branch available only through git history.

Success criterion:

- A clean SP8192 smoke run launches from this fork without shape, tokenizer,
  dataset, or dependency failures.

### Phase 2: Reproduce A Known Record-Family Baseline

Reproduce the April SP8192 lineage before adding new ideas.

Preferred reproduction target:

- [SP8192 + GPTQ embeddings + SDClip + Loop45x2](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2)

Reason:

- It is the clearest base stack before the later TTT and parallel-residual
  stacking.
- It already contains the compression regime we need: GPTQ embeddings, SDClip,
  int6 matrices, and SP8192.

Secondary reproduction target:

- [SP8192 + QK5 legal TTT](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-04-06_SP8192_QK5_LegalTTT_1.0828)

Reason:

- It establishes the current legal TTT pattern and shows the score lift from
  score-first TTT.

Success criterion:

- One shortened single-H100 run proves the code path works.
- One full `8xH100` run, when credits allow, lands in the same qualitative
  score regime as the official target.

### Phase 3: Port The Current SOTA Pieces

After the base stack is reproduced, port the top-record deltas in order.

Order:

1. Parallel residuals from late layers.
2. QK gain `5.0` to `5.25`.
3. Legal score-first TTT.
4. 3-layer depth recurrence.
5. LZMA/Brotli/byte-shuffle packaging improvements as needed for artifact
   headroom.

Success criterion:

- Each added feature has a before/after score and artifact-size comparison.
- No feature is kept solely because it is present in a top record; it must
  survive in our reproduced path.

### Phase 4: Add A Real Contribution

Only after reproduction is credible, add novelty.

Best current candidate:

- Hessian-aware or group-aware SDClip allocation.

Reason:

- The official Hessian-aware SDClip submission reports stable group-level
  Hessian traces across seeds, while row-level importance was noisy. That points
  to group-level clipping as a plausible improvement area without requiring a
  completely new architecture.

Other candidates:

- Better compression diagnostics that predict Brotli artifact size before full
  packaging.
- Cleaner TTT implementation or a smaller legal TTT variant with similar gain.
- Rotation/orthogonalization only as a post-GPTQ side ablation, not as the base
  compression method.

Success criterion:

- The contribution improves a reproduced SP8192 baseline, not the obsolete
  SP1024 baseline.
- It has a clean ablation and enough logs to justify a record or non-record PR.

## Immediate Next Actions

1. Create a clean SP8192 reproduction branch.
2. Pull the latest official upstream code and record folders into a local
   comparison area.
3. Download the SP8192 cached FineWeb data on the pod:

   ```bash
   MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
   python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 1
   ```

4. Run a single-H100 smoke job with reduced training, just to validate the
   tokenizer, model shape, and exporter path.
5. Port GPTQ/SDClip export and verify the artifact is legal or tunable.
6. Reproduce the April 5 SP8192 base stack before implementing TTT or new
   compression ideas.
7. When credits allow, use `8xH100` for a full reproduction run.

## Phase 1 Status

Branch:

- `codex/sp8192-upstream-base`

Carry-forward policy:

- Keep `AGENTS.md` and this roadmap.
- Use official `upstream/main` for code, records, data scripts, and
  requirements.
- Do not carry forward old SP1024 dense-int8 trainer modifications, minified
  scripts, or RunPod commands as reproduction code.

Official target now present in-tree:

- `records/track_10min_16mb/2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2`

Local check script:

```bash
python3 scripts/sp8192_apr05_phase1_check.py
```

Pod data/bootstrap check:

```bash
python3 scripts/sp8192_apr05_phase1_check.py --download-data --train-shards 1 --require-local-data
```

H100 smoke check after dependencies and data are present:

```bash
python3 scripts/sp8192_apr05_phase1_check.py --require-local-data --loader-forward
python3 scripts/sp8192_apr05_phase1_check.py --require-local-data --train-step
```

Current H100 pod result:

- Clean checkout: `/workspace/parameter-golf-sp8192`
- Official upstream commit: `75700cb`
- GPU: `1x NVIDIA H100 80GB HBM3`
- Runtime stack: `torch 2.11.0+cu130`, CUDA `13.0`, `flash_attn_interface`
  import OK
- Data: SP8192 tokenizer present, `1` train shard present, `1` validation shard
  present
- Manifest check: `fineweb10B_sp8192` found in `kevclark/parameter-golf`
  with `128` train shards and `1` validation shard
- Loader-forward smoke: passed, loss around `9.01`
- Single backward/optimizer train-step smoke: passed, loss around `9.03`

Phase 1 is complete when judged by the original success criterion: the April 5
SP8192 record-family code path can load the tokenizer/data, import the required
runtime dependencies, execute Flash Attention on H100, and run a real model
training step.

## Phase 2 Smoke Status

Goal:

- Prove the April 5 SP8192 record-family stack can run end-to-end on the H100
  before spending time on longer reproduction runs.

Smoke data:

- Created `/workspace/parameter-golf-sp8192/data_smoke`.
- Reused the SP8192 tokenizer from the real download.
- Wrote tiny valid SP8192-format train and validation shards with `8192` tokens
  each.

Human-readable April 5 script:

```bash
cd /workspace/parameter-golf-sp8192/phase2_runs/apr05_smoke_e2e_001
DATA_DIR=/workspace/parameter-golf-sp8192/data_smoke \
RUN_ID=phase2_apr05_smoke_e2e_001 \
SEED=1337 ITERATIONS=1 WARMUP_STEPS=0 \
TRAIN_SEQ_LEN=128 ROPE_TRAIN_SEQ_LEN=128 EVAL_SEQ_LEN=128 \
TRAIN_BATCH_TOKENS=1024 VAL_BATCH_TOKENS=1024 \
VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=1 MAX_WALLCLOCK_SECONDS=0 \
GPTQ_CALIBRATION_BATCHES=1 GPTQ_RESERVE_SECONDS=0 \
SLIDING_WINDOW_ENABLED=1 \
python3 /workspace/parameter-golf-sp8192/records/track_10min_16mb/2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2/train_gpt_human.py
```

Result:

- Training reached `1/1` step.
- Pre-quant post-EMA eval completed: `val_bpb 3.51007658`.
- GPTQ collected `67` Hessians.
- Quantized weights used the expected split: int6 matrix GPTQ, int8 token
  embedding GPTQ, float16 passthrough controls.
- Brotli artifact was written: `15,948,251` bytes.
- Human-script counted total was `16,006,618` bytes because the readable script
  is larger.
- Quantized eval completed: `val_bpb 3.51013157`.
- Quantized sliding-window eval completed: `val_bpb 3.51012659`.

Compressed submission April 5 script:

```bash
cd /workspace/parameter-golf-sp8192/phase2_runs/apr05_submission_smoke_e2e_001
DATA_DIR=/workspace/parameter-golf-sp8192/data_smoke \
RUN_ID=phase2_apr05_submission_smoke_e2e_001 \
SEED=1337 ITERATIONS=1 WARMUP_STEPS=0 \
TRAIN_SEQ_LEN=128 ROPE_TRAIN_SEQ_LEN=128 EVAL_SEQ_LEN=128 \
TRAIN_BATCH_TOKENS=1024 VAL_BATCH_TOKENS=1024 \
VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=1 MAX_WALLCLOCK_SECONDS=0 \
GPTQ_CALIBRATION_BATCHES=1 GPTQ_RESERVE_SECONDS=0 \
SLIDING_WINDOW_ENABLED=1 \
python3 /workspace/parameter-golf-sp8192/records/track_10min_16mb/2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2/train_gpt.py
```

Result:

- Training reached `1/1` step.
- Pre-quant post-EMA eval completed: `val_bpb 3.51007658`.
- GPTQ collected `67` Hessians.
- Brotli artifact was written: `15,948,251` bytes.
- Counted code size was `15,516` bytes.
- Counted total size was `15,963,767` bytes, under the `16,000,000` byte cap.
- Quantized eval completed: `val_bpb 3.51013157`.
- Quantized sliding-window eval completed: `val_bpb 3.51012659`.

Interpretation:

- The April 5 SP8192 stack is operational on this single H100.
- The actual compressed submission script, not only the human-readable version,
  runs through training, GPTQ export, artifact reload, quantized eval, and
  sliding eval.
- The next useful run should use real SP8192 validation and a short real-data
  training schedule, not the tiny smoke shards.

Short real-data run:

```bash
cd /workspace/parameter-golf-sp8192/phase2_runs/apr05_submission_realdata_2step_001
DATA_DIR=/workspace/parameter-golf-sp8192/data \
RUN_ID=phase2_apr05_submission_realdata_2step_001 \
SEED=1337 ITERATIONS=2 WARMUP_STEPS=0 \
VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=1 MAX_WALLCLOCK_SECONDS=0 \
GPTQ_CALIBRATION_BATCHES=2 GPTQ_RESERVE_SECONDS=0 \
SLIDING_WINDOW_ENABLED=0 \
python3 /workspace/parameter-golf-sp8192/records/track_10min_16mb/2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2/train_gpt.py
```

Result:

- Used real SP8192 data root with `1` downloaded training shard and the full
  validation shard.
- Validation covered `40,540,160` tokens.
- Training reached `2/2` steps with default `2048` sequence length and default
  April 5 batch shape.
- Peak memory: `35,482 MiB` allocated, `35,542 MiB` reserved on a single H100.
- Last-step validation: `val_bpb 4.2938`.
- Pre-quant post-EMA full validation: `val_bpb 3.47672552`,
  `eval_time 19,579ms`.
- GPTQ collected `67` Hessians from `2` calibration batches.
- Brotli artifact: `15,951,844` bytes.
- Counted code size: `15,516` bytes.
- Counted total size: `15,967,360` bytes, under the `16,000,000` byte cap.
- Quantized full validation: `val_bpb 3.47670741`, `eval_time 34,578ms`.

Interpretation:

- The record-style April 5 script works on real SP8192 validation on a single
  H100.
- Full validation without sliding is fast enough for short iteration loops.
- The next meaningful single-H100 experiment should use a longer schedule with
  recurrence activation less pathological than the `2`-step smoke run.

## Phase 2 Bounded-Run Status

Goal:

- Use the single H100 to find which parts of the April 5 SP8192 stack are
  trustworthy before spending credits on a full reproduction.

Current pod data constraint:

- The pod currently has `38` SP8192 train shards plus the full validation shard.
- A full `128`-shard download failed at train shard `38` with `Disk quota
  exceeded`.
- Removed partial Hugging Face caches and the zero-byte partial shard after the
  failure.
- Treat this pod as a reduced-data experimentation box unless the volume/quota
  changes.

Ten-minute bounded run with official EMA:

```bash
cd /workspace/parameter-golf-sp8192/phase2_runs/apr05_submission_realdata_10min_001
DATA_DIR=/workspace/parameter-golf-sp8192/data \
RUN_ID=phase2_apr05_submission_realdata_10min_001 \
SEED=1337 ITERATIONS=1000 WARMUP_STEPS=0 \
VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=1 MAX_WALLCLOCK_SECONDS=600 \
GPTQ_RESERVE_SECONDS=120 GPTQ_CALIBRATION_BATCHES=8 \
SLIDING_WINDOW_ENABLED=0 \
python3 /workspace/parameter-golf-sp8192/records/track_10min_16mb/2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2/train_gpt.py
```

Result:

- Stopped at step `504/1000` after `480,740ms` effective training time.
- Recurrence activated at step `290`, frac `0.501`.
- Last-step full validation: `val_bpb 1.2569`.
- Post-EMA pre-quant full validation: `val_bpb 1.59685649`.
- Quantized full validation: `val_bpb 1.60313390`.
- Counted total size: `16,005,588` bytes, over the cap by `5,588` bytes.

Interpretation:

- Official `EMA_DECAY=0.997` is harmful for this shortened single-H100 schedule.
  The EMA is too stale when the run stops around eight minutes of training.

Ten-minute bounded run with `EMA_DECAY=0`:

```bash
cd /workspace/parameter-golf-sp8192/phase2_runs/apr05_submission_realdata_10min_ema0_001
DATA_DIR=/workspace/parameter-golf-sp8192/data \
RUN_ID=phase2_apr05_submission_realdata_10min_ema0_001 \
SEED=1337 ITERATIONS=1000 WARMUP_STEPS=0 \
VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=20 MAX_WALLCLOCK_SECONDS=600 \
GPTQ_RESERVE_SECONDS=120 GPTQ_CALIBRATION_BATCHES=8 \
SLIDING_WINDOW_ENABLED=0 EMA_DECAY=0 \
python3 /workspace/parameter-golf-sp8192/records/track_10min_16mb/2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2/train_gpt.py
```

Result:

- Stopped at step `507/1000` after `480,439ms` effective training time.
- Recurrence activated at step `292`.
- Post-EMA pre-quant full validation: `val_bpb 1.25628546`.
- Quantized full validation: `val_bpb 1.26491281`.
- Counted total size: `16,001,260` bytes, over the cap by `1,260` bytes.

Interpretation:

- `EMA_DECAY=0` fixes the short-run EMA failure.
- The score is already within the old local SP1024 legal result regime, but it
  needs a legal artifact and still trails current official SP8192 records by a
  large margin.

Legal export from the `EMA_DECAY=0` checkpoint:

```bash
cd /workspace/parameter-golf-sp8192/phase2_runs/apr05_submission_realdata_10min_ema0_export_k13_001
DATA_DIR=/workspace/parameter-golf-sp8192/data \
RUN_ID=phase2_apr05_submission_realdata_10min_ema0_export_k13_001 \
LOAD_CHECKPOINT=/workspace/parameter-golf-sp8192/phase2_runs/apr05_submission_realdata_10min_ema0_001/final_model.pt \
ITERATIONS=0 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=0 \
GPTQ_RESERVE_SECONDS=0 GPTQ_CALIBRATION_BATCHES=8 \
SLIDING_WINDOW_ENABLED=0 EMA_DECAY=0 MATRIX_CLIP_SIGMAS=13.0 \
python3 /workspace/parameter-golf-sp8192/records/track_10min_16mb/2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2/train_gpt.py
```

Result:

- Pre-quant full validation: `val_bpb 1.25628546`.
- Brotli artifact: `15,915,200` bytes.
- Counted total size: `15,930,716` bytes, legal under the `16,000,000` byte
  cap.
- Quantized full validation: `val_bpb 1.26511741`.

Interpretation:

- `MATRIX_CLIP_SIGMAS=13.0` gives enough legal artifact headroom with negligible
  score movement relative to the over-cap export.

Best current single-H100 sanity run:

```bash
cd /workspace/parameter-golf-sp8192/phase2_runs/apr05_submission_38shards_10min_ema0_k13_001
DATA_DIR=/workspace/parameter-golf-sp8192/data \
RUN_ID=phase2_apr05_submission_38shards_10min_ema0_k13_001 \
SEED=1337 ITERATIONS=1000 WARMUP_STEPS=0 \
VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=20 MAX_WALLCLOCK_SECONDS=600 \
GPTQ_RESERVE_SECONDS=120 GPTQ_CALIBRATION_BATCHES=8 \
SLIDING_WINDOW_ENABLED=0 EMA_DECAY=0 MATRIX_CLIP_SIGMAS=13.0 \
python3 /workspace/parameter-golf-sp8192/records/track_10min_16mb/2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2/train_gpt.py
```

Result:

- Used `38` train shards and the full `40,540,160`-token validation shard.
- Stopped at step `462/1000` after `481,003ms` effective training time.
- Recurrence activated at step `250`, frac `0.501`.
- Post-EMA pre-quant full validation: `val_bpb 1.25640200`.
- Brotli artifact: `15,915,766` bytes.
- Counted total size: `15,931,282` bytes, legal under the cap.
- Quantized full validation: `val_bpb 1.26396070`.

Interpretation:

- The current single-H100 baseline is legal and operational.
- `38` train shards did not materially improve dense validation over the
  one-shard bounded run, but it slightly improved legal quantized score.
- The main value of this run is operational: it confirms the current recipe,
  artifact size, and full-validation loop are stable on the live pod.

Export sweep from the `38`-shard checkpoint:

| Run | Matrix clip | GPTQ calibration batches | Total bytes | Quantized BPB | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `apr05_38shards_export_k12p85_cal8` | `12.850` | `8` | `16,001,261` | `1.26375852` | over cap |
| `apr05_38shards_export_k12p852_cal8` | `12.852` | `8` | `16,000,139` | `1.26376026` | over cap |
| `apr05_38shards_export_k12p853_cal8` | `12.853` | `8` | `15,999,590` | `1.26376547` | legal |
| `apr05_38shards_export_k12p854_cal8` | `12.854` | `8` | `15,999,743` | `1.26376420` | legal |
| `apr05_38shards_export_k12p855_cal8` | `12.855` | `8` | `15,999,298` | `1.26375981` | legal, best |
| `apr05_38shards_export_k12p856_cal8` | `12.856` | `8` | `15,998,332` | `1.26376272` | legal |
| `apr05_38shards_export_k12p858_cal8` | `12.858` | `8` | `15,997,308` | `1.26376813` | legal |
| `apr05_38shards_export_k12p86_cal8` | `12.860` | `8` | `15,996,777` | `1.26377308` | legal |

GPTQ calibration sweep at the best clip:

| Run | Matrix clip | GPTQ calibration batches | Total bytes | Quantized BPB | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `apr05_38shards_export_k12p855_cal8` | `12.855` | `8` | `15,999,298` | `1.26375981` | best |
| `apr05_38shards_export_k12p855_cal16` | `12.855` | `16` | `15,998,168` | `1.26379644` | worse |
| `apr05_38shards_export_k12p855_cal32` | `12.855` | `32` | `15,998,731` | `1.26379855` | worse |
| `apr05_38shards_export_k12p855_cal64` | `12.855` | `64` | `15,999,249` | `1.26379062` | worse |
| `apr05_38shards_export_k12p860_cal64` | `12.860` | `64` | `15,996,487` | `1.26381171` | worse |

Interpretation:

- The best current legal single-H100 artifact is
  `apr05_38shards_export_k12p855_cal8`: `15,999,298` bytes and
  `1.26375981` full-validation quantized BPB.
- The legal clip boundary is around `12.853`; `12.852` was only `139` bytes
  over cap.
- More GPTQ calibration batches did not help this checkpoint. For this
  shortened run, keep `GPTQ_CALIBRATION_BATCHES=8` unless a later checkpoint
  shows different behavior.
- This is a small but useful exporter improvement over the first legal
  `MATRIX_CLIP_SIGMAS=13.0` run, which scored `1.26396070`.

Near-term decision:

- Keep using the single H100 for short ablations around EMA, clip sigma, GPTQ
  calibration, warmup/warmdown, and sliding/TTT plumbing.
- Do not treat the single-H100 bounded score as a leaderboard proxy. It is a
  correctness and triage tool.
- A real catch-up attempt still needs a larger-volume pod for all shards and
  eventually `8xH100` for official-scale reproduction.

## Promotion Rules

A branch can receive serious compute only if all are true:

- It uses SP8192 or a directly competitive tokenizer path.
- It has GPTQ/SDClip-style compression, not only dense int8.
- It has a legal artifact-size plan under `16,000,000` bytes.
- It has a reduced-run result proving the code path is correct.
- It can be compared against an official record-family baseline.

A branch should be abandoned or demoted to side work if any are true:

- It only improves the old SP1024 dense-int8 baseline.
- It needs many full runs before showing a plausible signal.
- It cannot explain artifact size or legal roundtrip quality.
- It competes with already-proven official techniques instead of building on
  them.

## PR Strategy

No PR is worth submitting yet.

Next acceptable PR target:

- a non-record submission if we produce a clean, interesting negative result or
  a simplified reproducible implementation;
- a record submission only if we beat the leaderboard threshold with the
  required statistical evidence and a self-contained record folder.

For now, focus on reproduction plus one defensible improvement. Submitting a PR
before that would create maintenance overhead without improving our competitive
position.

## Bottom Line

The project should continue, but the strategy must pivot.

The old path was useful for learning the challenge mechanics, artifact sizing,
and strict timing, but it is no longer competitive. The next work should be
SP8192-first, GPTQ/SDClip-first, and reproduction-first. Once that base is
working, the best original angle is group/Hessian-aware compression on top of
the official SOTA stack.
