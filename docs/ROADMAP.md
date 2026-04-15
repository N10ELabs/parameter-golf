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
