# Upstream Scan: 2026-03-19

This note summarizes the latest meaningful upstream commits from
`openai/parameter-golf` as of `2026-03-19` and turns them into concrete ideas
for this fork.

## Snapshot

- Fetched `upstream/main` on `2026-03-19`.
- Local fork state at scan time: `2` commits ahead of `origin/main`, `28`
  commits behind `upstream/main`.
- The newest upstream commit was `5e29bfd`, but most same-day commits were
  README/leaderboard updates. The substantive changes were in new record
  folders and a few code fixes.

## Highest-Value Ideas

| Commit | Score | Main idea | Why it matters |
| --- | --- | --- | --- |
| `555669e` | `1.1574` | train at `seq_len=2048`, use FP16 tied embeddings, keep some late `c_k` weights in FP16, and do final sliding-window eval at `seq_len=2048`, `stride=256` | strongest recent result; combines better compression behavior with richer-context eval |
| `9fbdf8c` | `1.1748` | sliding-window eval + FP16 embed export + `10` layers + Muon weight decay + overtone init | shows that compression savings can be reinvested into depth |
| `d84a3e8` + `3a6fec7` | `1.1925` | sliding-window eval, then fix the skipped final partial window | best low-complexity idea; most of the gain came from evaluation only |
| `bd2463a` | `1.1928` | document-aware LoRA test-time training | interesting, but the ablations say most of the win came from doc isolation and strided eval rather than LoRA itself |
| `78c24e2` | `1.2014` | `TRAIN_SEQ_LEN=4096`, lower LRs, higher Muon momentum, smaller batch | longer training context can beat shorter-context runs even with fewer steps |
| `9ac12c2` | `1.2147` | `10` layers plus mixed `int8`/`int6` compression on middle layers | useful pattern for buying depth under the `16 MB` cap |
| `a5eb9ed` | `1.2197` | keep `tok_emb.weight` in FP16 during export, offset the byte cost by shrinking MLP hidden | clear evidence that tied embeddings are unusually quantization-sensitive |
| `6a08c9d` | n/a | `MLX_EAGER_EVAL` flag | operational improvement for MLX memory pressure, not a score idea |

## What Upstream Is Really Doing

The recent upstream direction is not one trick. It is a consistent pattern:

1. Improve evaluation before touching architecture.
2. Optimize for post-quantized score, not just pre-quantized loss.
3. Spend saved bytes on extra context, extra depth, or wider MLPs.

The best repeated themes were:

- **Sliding-window evaluation**: score only the right edge of overlapping
  windows so each scored token gets near-max context.
- **Document isolation during eval**: avoid cross-document contamination.
- **FP16 passthrough for the tied embedding**: `tok_emb.weight` is both the
  input embedding and the output head, so quantization errors hurt twice.
- **Longer warmdown or lower LR regimes**: several records improved mainly by
  making weights easier to quantize.
- **Longer training or evaluation context**: `1408`, `2048`, and `4096`
  appeared repeatedly.
- **Selective mixed precision**: middle layers and non-critical tensors can be
  quantized harder than the most sensitive layers.

## Details Worth Preserving

### 1. Sliding-window eval is the easiest high-ROI port

The `Sliding Window Eval` record kept training effectively baseline-like and
still gained about `0.032` `val_bpb` post-quantized by:

- setting `EVAL_STRIDE=64`
- evaluating overlapping `1024`-token windows
- scoring only the rightmost `64` tokens from each window
- batching many windows together for throughput

The follow-up fix in `3a6fec7` matters: the first implementation skipped short
final windows and could silently drop up to `stride - 1` tokens.

### 2. FP16 tied embeddings are now a first-class compression trick

Two independent upstream records highlighted the same point:

- `a5eb9ed`: keeping `tok_emb.weight` in FP16 collapsed the quantization gap
  from about `0.007` `bpb` to about `0.0005`
- `555669e`: the strongest recent run also kept the tied embedding in FP16

The recurring tradeoff was:

- spend about `500 KB` on FP16 embedding storage
- recover that budget elsewhere, often by shrinking MLP hidden or using more
  aggressive quantization on less sensitive weights

### 3. Training for compressibility matters more than before

The most useful optimizer lesson was not simply "bigger LR" or "smaller LR."
It was that the best schedule depends on the target artifact format.

Observed upstream patterns:

- `a5eb9ed`: `WARMDOWN_ITERS=3600`, `MATRIX_LR=0.06`
- `9ac12c2`: lower LRs worked better for the `10`-layer mixed-precision path
- `555669e`: the saved log shows a lower-LR long-context recipe around
  `token_lr=0.03`, `matrix_lr=0.02`, `scalar_lr=0.02`

The unifying idea is that smoother weight distributions quantize better.

### 4. Longer context is paying for itself

Recent upstream runs used context in three different places:

- eval-only context extension via sliding windows
- moderate NTK-style eval extension such as `EVAL_SEQ_LEN=1408`
- full training-context increases such as `TRAIN_SEQ_LEN=2048` or `4096`

This suggests the old "maximize steps at `1024`" mindset is no longer enough by
itself.

### 5. Compression savings are being reinvested aggressively

Upstream is using byte savings to buy:

- a `10th` transformer layer
- a `3x` MLP expansion
- FP16 passthrough for sensitive tensors
- richer eval procedures that cost time but not artifact bytes

That is the real strategic shift: compression is not just for legality now. It
is a budget source for model quality.

## Evidence Caveat

The `555669e` record needs special care:

- its commit message advertises `1.1574`
- its `submission.json` and `train.log` confirm `1.15744040`
- but its `README.md` still describes an older `1.2154` warmdown-only recipe

So for that record, trust the saved `submission.json` and `train.log` more than
the README prose.

## Recommended Port Order For This Fork

If the goal is to steal the highest-value ideas with the least risk, the port
order should be:

1. Add sliding-window evaluation and include the final-partial-window fix.
2. Add FP16 passthrough for `tok_emb.weight` during export.
3. Sweep a small compression-aware schedule family instead of generic tuning:
   lower LRs, longer warmdown, and possibly a moderate `EVAL_SEQ_LEN` increase.
4. Try longer context directly: `TRAIN_SEQ_LEN=2048` before more speculative
   architecture work.
5. Add selective FP16 or lower-bit passthrough for a tiny set of sensitive
   late-layer tensors if bytes still allow it.
6. Leave LoRA TTT, overtone init, and other more complex tricks for later.

## Practical Next Experiments

The fastest way to convert this upstream scan into actual fork progress would
be a small sequence like:

1. Baseline + sliding-window eval only.
2. Baseline + FP16 tied embedding export only.
3. Sliding-window eval + FP16 tied embedding export.
4. The combined recipe above with a longer warmdown or lower-LR sweep.
5. The best combined recipe above with `TRAIN_SEQ_LEN=2048`.

That order isolates the biggest upstream ideas before mixing in harder-to-debug
architectural changes.
