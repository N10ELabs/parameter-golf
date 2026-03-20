# Docs

This folder is for working notes, experiment plans, hypotheses, and logs while
you learn the challenge and iterate on your fork.

Suggested use:

- Keep planning and notes here.
- Keep actual challenge submissions under `records/`.
- Keep core training code changes in `train_gpt.py` or `train_gpt_mlx.py`.

For this fork, adding docs here is fine. These files do not matter for the
challenge artifact unless you intentionally copy them into a submission record.

Files:

- `current-frontier.md`: current best legal run, strongest active branch, what is working, what is not, and the highest-value next moves
- `first-five-runs.md`: beginner-friendly experiment plan for your first runs
- `next-five-runs.md`: batch 2 plan focused on depth, KV heads, and batch size
- `batch-three-runs.md`: batch 3 plan focused on cap calibration and small-batch promotion
- `batch-four-runs.md`: batch 4 plan focused on fine-grained cap-edge stopping-point search
- `batch-five-runs.md`: batch 5 plan and results for artifact engineering plus `925/926` cap-edge validation
- `batch-six-runs.md`: batch 6 plan focused on reallocating artifact budget into longer training
- `batch-seven-runs.md`: batch 7 plan focused on finding the best legal stop point for the `11-layer` regime
- `batch-eight-runs.md`: batch 8 plan focused on improving compressibility of the current best model family
- `batch-nine-runs.md`: batch 9 plan focused on improving low-bit fidelity with grouped `int4` scales and clipping
- `batch-ten-runs.md`: batch 10 plan focused on transferring the best low-bit recipe onto stronger dense checkpoints
- `batch-eleven-runs.md`: batch 11 plan and results focused on spending remaining low-bit budget on finer grouped scales
- `batch-twelve-runs.md`: batch 12 plan and results focused on transferring the gentler one-layer recipe onto near-cap dense frontiers
- `batch-thirteen-runs.md`: batch 13 plan and results focused on minimal-invasive attention targets for legalizing `run28`
- `batch-fourteen-runs.md`: batch 14 plan and results focused on dense-only code golf to legalize `run29` and probe `run28`
- `batch-fifteen-runs.md`: batch 15 plan and results focused on serializer and compressor changes for legalizing `run28`
- `batch-sixteen-runs.md`: batch 16 plan and results focused on spending the recovered `lzma` headroom on a new dense cap edge
- `batch-seventeen-runs.md`: batch 17 plan and results focused on upstream-inspired sliding eval promotion and the first upstream-style `10`-layer branch
- `upstream-scan-2026-03-19.md`: summary of the latest upstream `openai/parameter-golf` commits and the best ideas to port into this fork
- `strategy-reassessment.md`: retrospective on what the experiments taught us and the highest-value next path
- `run-log.md`: simple template for recording what you changed and what happened
