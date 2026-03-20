# Run Log

Use this file as a lightweight experiment journal.

## Summary Table

| Run | Hypothesis | Change | Final train_loss | Final val_loss | Final val_bpb | Final roundtrip val_loss | Final roundtrip val_bpb | Int8+zlib bytes | Step avg ms | Peak memory MiB | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run01_baseline | Baseline reference | None | 3.2278 | 3.2500 | 1.9249 | 3.41029634 | 2.01976959 | 7184469 | 491.33 | 10940 | Reference point |
| run02_width640 | More width helps early quality | `MODEL_DIM=640` | 3.4601 | 3.4790 | 2.0605 | 3.59490038 | 2.12910250 | 10141339 | 471.42 | 13610 | Worse score, larger artifact |
| run03_depth12 | More depth helps more than width | `NUM_LAYERS=12` | 3.1645 | 3.1866 | 1.8873 | 3.33824607 | 1.97709736 | 9443596 | 471.59 | 14391 | Best 200-step score |
| run04_kv2 | Fewer KV heads may be more efficient | `NUM_KV_HEADS=2` | 3.2405 | 3.2610 | 1.9313 | 3.40379212 | 2.01591742 | 8093489 | 338.95 | 10456 | Near-baseline score, much faster |
| run05_extend_best | Best short run stays best longer | `NUM_LAYERS=12`, `ITERATIONS=500` | 2.4037 | 2.4338 | 1.4414 | 2.44999506 | 1.45102507 | 13125796 | 531.55 | 14391 | Depth winner held and improved |
| run06_depth12_kv2 | Depth and fewer KV heads may combine well | `NUM_LAYERS=12`, `NUM_KV_HEADS=2` | 3.1626 | 3.1829 | 1.8851 | 3.32248944 | 1.96776539 | 10659516 | 483.44 | 13756 | Won 200-step promotion on score |
| run07_depth12_batch262k_eqtokens | Smaller batch may improve sample efficiency | `NUM_LAYERS=12`, `TRAIN_BATCH_TOKENS=262144`, `ITERATIONS=400` | 2.5860 | 2.5959 | 1.5374 | 2.61739137 | 1.55016660 | 12048828 | 427.14 | 7380 | Strong diagnostic, not promoted by rule |
| run08_depth12_matrixlr003 | Deeper model may want lower matrix LR | `NUM_LAYERS=12`, `MATRIX_LR=0.03` | 3.2703 | 3.2914 | 1.9494 | 3.52538185 | 2.08792971 | 8895818 | 482.57 | 14391 | Worse score, lower LR hurt |
| run09_depth12_kv2_500 | Better 200-step variant should hold at 500 steps | `NUM_LAYERS=12`, `NUM_KV_HEADS=2`, `ITERATIONS=500` | 2.4188 | 2.4472 | 1.4494 | 2.46178746 | 1.45800920 | 14117703 | 524.97 | 13756 | Close, but `run05` stayed better balanced |
| run10_depth12_1000 | Best 500-step depth variant should hold at 1000 steps | `NUM_LAYERS=12`, `ITERATIONS=1000` | 2.3242 | 2.2878 | 1.3549 | 2.29031424 | 1.35645310 | 16316309 | 494.55 | 14391 | Best raw score, but over 16 MB cap |

## Per-Run Notes

### run01_baseline

- Run ID: `run01_baseline`
- Log path: `/workspace/parameter-golf/logs/run01_baseline.txt`
- Command: baseline config, `ITERATIONS=200`
- What I expected: clean reference curve and moderate compressed size
- What happened: steady convergence, `2.01976959` roundtrip val_bpb, `7.18 MB` int8+zlib model
- What I learned: the default config is a strong reference and gives a stable comparison point

### run02_width640

- Run ID: `run02_width640`
- Log path: `/workspace/parameter-golf/logs/run02_width640.txt`
- Command: baseline plus `MODEL_DIM=640`
- What I expected: width might buy better quality at some memory cost
- What happened: score got worse, compressed size jumped to `10.14 MB`, memory rose to `13.6 GiB`
- What I learned: extra width was not worth it on this short budget

### run03_depth12

- Run ID: `run03_depth12`
- Log path: `/workspace/parameter-golf/logs/run03_depth12.txt`
- Command: baseline plus `NUM_LAYERS=12`
- What I expected: more depth might help more cleanly than width
- What happened: best 200-step result, `1.97709736` roundtrip val_bpb, better than baseline and width
- What I learned: depth is the most promising of the first-round architecture changes

### run04_kv2

- Run ID: `run04_kv2`
- Log path: `/workspace/parameter-golf/logs/run04_kv2.txt`
- Command: baseline plus `NUM_KV_HEADS=2`
- What I expected: fewer KV heads might give a useful efficiency trade
- What happened: almost baseline-level score with much faster steps and lower memory
- What I learned: KV-head reduction is a strong efficiency lever even if it did not beat the depth run on score

### run05_extend_best

- Run ID: `run05_extend_best`
- Log path: `/workspace/parameter-golf/logs/run05_extend_best.txt`
- Command: `NUM_LAYERS=12`, `ITERATIONS=500`
- What I expected: if depth was a real winner, the lead should survive longer training
- What happened: it improved sharply to `1.45102507` roundtrip val_bpb and stayed under the `16 MB` cap
- What I learned: the 200-step winner was not a fluke; deeper was genuinely the best direction in this first batch

### run06_depth12_kv2

- Run ID: `run06_depth12_kv2`
- Log path: `/workspace/parameter-golf/logs/run06_depth12_kv2.txt`
- Command: `NUM_LAYERS=12`, `NUM_KV_HEADS=2`, `ITERATIONS=200`
- What I expected: depth plus fewer KV heads might keep most of the depth win while improving efficiency
- What happened: it edged out `run03` at the same 200-step budget with `1.96776539` roundtrip val_bpb, though the artifact was larger than plain depth
- What I learned: depth and KV reduction do compose, and this was the right batch-2 promotion candidate on score

### run07_depth12_batch262k_eqtokens

- Run ID: `run07_depth12_batch262k_eqtokens`
- Log path: `/workspace/parameter-golf/logs/run07_depth12_batch262k_eqtokens.txt`
- Command: `NUM_LAYERS=12`, `TRAIN_BATCH_TOKENS=262144`, `ITERATIONS=400`
- What I expected: a smaller batch might improve sample efficiency for the deeper model if total token budget stayed fixed
- What happened: it produced a very strong `1.55016660` roundtrip val_bpb with only `7.38 GiB` peak memory and a faster `427.14 ms` step average
- What I learned: batch size is now a first-class lever for this repo and deserves direct promotion treatment in the next batch

### run08_depth12_matrixlr003

- Run ID: `run08_depth12_matrixlr003`
- Log path: `/workspace/parameter-golf/logs/run08_depth12_matrixlr003.txt`
- Command: `NUM_LAYERS=12`, `MATRIX_LR=0.03`, `ITERATIONS=200`
- What I expected: slightly lowering `MATRIX_LR` might help the deeper model quantize more cleanly
- What happened: it regressed badly to `2.08792971` roundtrip val_bpb despite a smaller artifact
- What I learned: the default matrix LR is safer here; lowering it this much hurt quality more than it helped compression

### run09_depth12_kv2_500

- Run ID: `run09_depth12_kv2_500`
- Log path: `/workspace/parameter-golf/logs/run09_depth12_kv2_500.txt`
- Command: `NUM_LAYERS=12`, `NUM_KV_HEADS=2`, `ITERATIONS=500`
- What I expected: if the 200-step combo was real, it should stay competitive against `run05` at the same 500-step budget
- What happened: it finished at `1.45800920` roundtrip val_bpb, very close to `run05`, but still worse and with a larger artifact
- What I learned: KV reduction remains viable, but pure depth stayed the better balanced 500-step choice in this batch

### run10_depth12_1000

- Run ID: `run10_depth12_1000`
- Log path: `/workspace/parameter-golf/logs/run10_depth12_1000.txt`
- Command: `NUM_LAYERS=12`, `ITERATIONS=1000`
- What I expected: the 500-step incumbent might keep improving on score if we gave it a longer schedule
- What happened: it reached the best raw batch-2 score at `1.35645310` roundtrip val_bpb, but the int8+zlib artifact grew to `16,316,309` bytes and broke the `16 MB` cap
- What I learned: longer training on the depth winner still helps, but artifact growth is now the limiting constraint

## Batch 2 Summary

- `run06` validated the `NUM_LAYERS=12 + NUM_KV_HEADS=2` combination, `run07` showed that smaller batches are a serious efficiency and quality lever, `run08` ruled out a lower `MATRIX_LR`, `run09` proved the KV-reduced depth model stays close but does not beat pure depth at 500 steps, and `run10` showed that pure depth keeps improving until artifact size becomes the blocker.
- For batch 3, keep `NUM_LAYERS=12` as the architectural base, but promote `TRAIN_BATCH_TOKENS=262144` from “diagnostic” to a first-class search dimension. Pure depth at 500 steps is still the best directly promoted valid scorer, but smaller-batch depth is the most important new direction to explore.

## Batch 3

| Run | Hypothesis | Change | Final train_loss | Final val_loss | Final val_bpb | Final roundtrip val_loss | Final roundtrip val_bpb | Int8+zlib bytes | Step avg ms | Peak memory MiB | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run11_depth12_950 | Near-cap pure depth should retain most of the 1000-step gain | `NUM_LAYERS=12`, `ITERATIONS=950` | 2.1956 | 2.2925 | 1.3578 | 2.29530613 | 1.35940958 | 16072708 | 541.51 | 14391 | Over total cap by `120,350` bytes |
| run12_depth12_925 | Slightly earlier pure depth should safely clear the cap | `NUM_LAYERS=12`, `ITERATIONS=925` | 2.1690 | 2.2933 | 1.3582 | 2.29648552 | 1.36010808 | 15953418 | 520.46 | 14391 | Over total cap by `1,060` bytes |
| run13_depth12_batch262k_600 | Smaller batch should earn a fair wallclock-matched promotion test | `NUM_LAYERS=12`, `TRAIN_BATCH_TOKENS=262144`, `ITERATIONS=600` | 2.4240 | 2.4352 | 1.4422 | 2.44350054 | 1.44717865 | 13917660 | 507.26 | 7380 | Valid under total cap, but worse than pure depth |
| run14_depth12_batch262k_800 | Smaller batch may keep improving without hitting the cap | `NUM_LAYERS=12`, `TRAIN_BATCH_TOKENS=262144`, `ITERATIONS=800` | 2.2973 | 2.3555 | 1.3951 | 2.35893899 | 1.39709654 | 15242441 | 431.74 | 7380 | Valid under total cap, still behind pure depth |
| run15_depth12_batch262k_900 | Smaller batch near the cap may become the best valid scorer | `NUM_LAYERS=12`, `TRAIN_BATCH_TOKENS=262144`, `ITERATIONS=900` | 2.3083 | 2.3372 | 1.3842 | 2.33975195 | 1.38573290 | 15760592 | 515.01 | 7380 | Best valid batch-3 run under the true cap |

### run11_depth12_950

- Run ID: `run11_depth12_950`
- Log path: `/workspace/parameter-golf/logs/run11_depth12_950.txt`
- Command: `NUM_LAYERS=12`, `ITERATIONS=950`
- What I expected: this should keep most of the `1000`-step score while slipping just under the size cap
- What happened: it reached `1.35940958` roundtrip val_bpb, but total submission size int8+zlib landed at `16,120,350` bytes
- What I learned: the pure-depth near-cap slope is steep, and `950` steps is still too aggressive once total artifact bytes are counted

### run12_depth12_925

- Run ID: `run12_depth12_925`
- Log path: `/workspace/parameter-golf/logs/run12_depth12_925.txt`
- Command: `NUM_LAYERS=12`, `ITERATIONS=925`
- What I expected: this should be the safer near-cap pure-depth checkpoint with only a small score penalty versus `run11`
- What happened: it finished at `1.36010808` roundtrip val_bpb with `15,953,418` model bytes, but total submission size int8+zlib was `16,001,060` bytes
- What I learned: `925` steps is still too large once code bytes are counted; the pure-depth line needed one more step trimmed

### run13_depth12_batch262k_600

- Run ID: `run13_depth12_batch262k_600`
- Log path: `/workspace/parameter-golf/logs/run13_depth12_batch262k_600.txt`
- Command: `NUM_LAYERS=12`, `TRAIN_BATCH_TOKENS=262144`, `ITERATIONS=600`
- What I expected: this would be the first fair promotion test for the smaller-batch regime at a comparable wallclock budget
- What happened: it produced `1.44717865` roundtrip val_bpb at `13,917,660` model bytes and `13,965,302` total bytes, with only `7.38 GiB` peak memory
- What I learned: smaller batch remains efficient and safe, but `600` steps is not enough to beat the pure-depth line

### run14_depth12_batch262k_800

- Run ID: `run14_depth12_batch262k_800`
- Log path: `/workspace/parameter-golf/logs/run14_depth12_batch262k_800.txt`
- Command: `NUM_LAYERS=12`, `TRAIN_BATCH_TOKENS=262144`, `ITERATIONS=800`
- What I expected: the smaller-batch branch might need a longer schedule before it became competitive with the pure-depth winner
- What happened: it improved to `1.39709654` roundtrip val_bpb at `15,242,441` model bytes and `15,290,083` total bytes
- What I learned: longer training helps this branch a lot, but it is still not good enough to displace the pure-depth line

### run15_depth12_batch262k_900

- Run ID: `run15_depth12_batch262k_900`
- Log path: `/workspace/parameter-golf/logs/run15_depth12_batch262k_900.txt`
- Command: `NUM_LAYERS=12`, `TRAIN_BATCH_TOKENS=262144`, `ITERATIONS=900`
- What I expected: this near-cap smaller-batch probe would tell us whether the branch could actually overtake the pure-depth winner
- What happened: it reached `1.38573290` roundtrip val_bpb at `15,760,592` model bytes and `15,808,234` total bytes
- What I learned: smaller batch is a real efficiency direction, but even its best valid batch-3 run did not catch the pure-depth line

## Batch 3 Summary

- `run11` and `run12` mapped the pure-depth cap edge cleanly, but both were still invalid once total submission bytes were counted.
- `run13`, `run14`, and `run15` proved the smaller-batch branch is efficient and memory-light, improving from `1.44717865` to `1.38573290`, and `run15` became the best valid batch-3 run under the true total-size cap.
- The batch-4 base shifted back to pure depth, but with corrected cap accounting and a finer one-step stopping-point search.

## Batch 4

| Run | Hypothesis | Change | Final train_loss | Final val_loss | Final val_bpb | Final roundtrip val_loss | Final roundtrip val_bpb | Int8+zlib bytes | Step avg ms | Peak memory MiB | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run16_depth12_935 | Tiny step-count increase may improve score while staying valid | `NUM_LAYERS=12`, `ITERATIONS=935` | 2.1692 | 2.2926 | 1.3578 | 2.29598159 | 1.35980963 | 16002541 | 483.20 | 14391 | Over total cap by `50,183` bytes |
| run17_depth12_934 | One-step trim may recover validity with almost no score loss | `NUM_LAYERS=12`, `ITERATIONS=934` | 2.1699 | 2.2930 | 1.3580 | 2.29616103 | 1.35991590 | 15998392 | 550.70 | 14391 | Over total cap by `46,034` bytes |
| run18_depth12_924 | Corrected pure-depth stop should restore legality | `NUM_LAYERS=12`, `ITERATIONS=924` | 2.1674 | 2.2932 | 1.3581 | 2.29614836 | 1.35990839 | 15948381 | 554.55 | 14391 | New best valid run overall |
| run19_depth12_925_codecut | Artifact engineering should legalize the old `925` stop | `NUM_LAYERS=12`, `ITERATIONS=925`, trimmed `train_gpt.py`, hybrid serializer | 2.1680 | 2.2932 | 1.3581 | 2.29581476 | 1.35971082 | 15952925 | 519.68 | 14391 | New best valid run overall |
| run20_depth12_926_codecut | Recovered headroom may permit one more pure-depth step | `NUM_LAYERS=12`, `ITERATIONS=926`, trimmed `train_gpt.py`, hybrid serializer | 2.1694 | 2.2936 | 1.3584 | 2.29635686 | 1.36003188 | 15958008 | 569.61 | 14391 | Legal, but slightly worse than `run19` |
| run21_depth11_1100_codecut | One less layer may spend artifact budget better if trained longer | `NUM_LAYERS=11`, `ITERATIONS=1100`, trimmed `train_gpt.py`, hybrid serializer | 2.1830 | 2.2903 | 1.3564 | 2.29244696 | 1.35771622 | 15433606 | 545.55 | 13241 | New best valid run overall |
| run22_dim480_1100_codecut | Modest width reduction may buy a better size-to-quality trade | `MODEL_DIM=480`, `NUM_HEADS=8`, `ITERATIONS=1100`, trimmed `train_gpt.py`, hybrid serializer | 2.2274 | 2.3149 | 1.3710 | 2.31731774 | 1.37244609 | 11274049 | 528.50 | 10531 | Much smaller, but clearly worse on score |
| run23_depth11_1400_codecut | The winning smaller-depth branch may improve further on a longer schedule | `NUM_LAYERS=11`, `ITERATIONS=1400`, trimmed `train_gpt.py`, hybrid serializer | 2.1131 | 2.2728 | 1.3461 | 2.27479807 | 1.34726355 | 16665279 | 533.72 | 13241 | Best raw score so far, but over total cap by `705,633` bytes |
| run24_dim480_1400_codecut | The width-reduced branch may catch up if given more steps | `MODEL_DIM=480`, `NUM_HEADS=8`, `ITERATIONS=1400`, trimmed `train_gpt.py`, hybrid serializer | 2.1658 | 2.2964 | 1.3600 | 2.29857691 | 1.36134672 | 12179721 | 437.67 | 10529 | Valid, but still worse than `run21` |
| run25_depth11_1700_codecut | The winning 11-layer branch may keep improving if pushed to the batch limit | `NUM_LAYERS=11`, `ITERATIONS=1700`, trimmed `train_gpt.py`, hybrid serializer | 2.1015 | 2.2584 | 1.3375 | 2.26014718 | 1.33858647 | 17595783 | 475.74 | 13241 | Best raw score so far, but over total cap by `1,636,137` bytes |

### run16_depth12_935

- Run ID: `run16_depth12_935`
- Log path: `/workspace/parameter-golf/logs/run16_depth12_935.txt`
- Command: `NUM_LAYERS=12`, `ITERATIONS=935`
- What I expected: this was the highest-value cap-edge probe because it might improve on `924/925` while staying legal
- What happened: it reached `1.35980963` roundtrip val_bpb, but total submission size int8+zlib landed at `16,050,183` bytes
- What I learned: the total-size cap is tight enough that even a 10-step bump from `925` was too much once code bytes were counted

### run17_depth12_934

- Run ID: `run17_depth12_934`
- Log path: `/workspace/parameter-golf/logs/run17_depth12_934.txt`
- Command: `NUM_LAYERS=12`, `ITERATIONS=934`
- What I expected: trimming one step from `run16` might be enough to recover validity
- What happened: it scored `1.35991590` roundtrip val_bpb, but total submission size int8+zlib was still `16,046,034` bytes
- What I learned: the boundary was lower than expected, and the earlier batch-3 accounting mistake mattered

### run18_depth12_924

- Run ID: `run18_depth12_924`
- Log path: `/workspace/parameter-golf/logs/run18_depth12_924.txt`
- Command: `NUM_LAYERS=12`, `ITERATIONS=924`
- What I expected: one more step trimmed from the pure-depth line should restore legality while keeping nearly all of the score
- What happened: it finished at `1.35990839` roundtrip val_bpb with total submission size int8+zlib of `15,996,023` bytes
- What I learned: `924` is the current best legal pure-depth stopping point and the best valid run overall so far

## Batch 4 Summary

- Correct cap accounting changed the search: the true constraint is total submission bytes, not just model bytes.
- `run16` and `run17` both landed just over the total-size limit, but `run18_depth12_924` came in valid and replaced `run15` as the best legal run overall.
- The current best base is `NUM_LAYERS=12`, `ITERATIONS=924`, with `final_int8_zlib_roundtrip_exact val_bpb: 1.35990839` and total submission size int8+zlib of `15,996,023` bytes.

## Batch 5

| Run | Hypothesis | Change | Final train_loss | Final val_loss | Final val_bpb | Final roundtrip val_loss | Final roundtrip val_bpb | Int8+zlib bytes | Step avg ms | Peak memory MiB | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run19_depth12_925_codecut | Artifact engineering should legalize the old `925` stop | `NUM_LAYERS=12`, `ITERATIONS=925`, trimmed `train_gpt.py`, hybrid serializer | 2.1680 | 2.2932 | 1.3581 | 2.29581476 | 1.35971082 | 15952925 | 519.68 | 14391 | New best valid run overall |
| run20_depth12_926_codecut | Recovered headroom may permit one more pure-depth step | `NUM_LAYERS=12`, `ITERATIONS=926`, trimmed `train_gpt.py`, hybrid serializer | 2.1694 | 2.2936 | 1.3584 | 2.29635686 | 1.36003188 | 15958008 | 569.61 | 14391 | Legal, but slightly worse than `run19` |

### run19_depth12_925_codecut

- Run ID: `run19_depth12_925_codecut`
- Log path: `/workspace/parameter-golf/logs/run19_depth12_925_codecut.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run19_depth12_925_codecut.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run19_depth12_925_codecut.final_model.int8.ptz`
- Command: `NUM_LAYERS=12`, `ITERATIONS=925`, trimmed `train_gpt.py`, hybrid serializer
- What I expected: the smaller script plus metadata-light serializer should make the old `925` pure-depth checkpoint legal again
- What happened: it finished at `1.35971082` roundtrip val_bpb with `15,952,925` model bytes and `15,993,279` total submission bytes
- What I learned: artifact engineering was enough to recover `925`, and this is now the best valid run overall

### run20_depth12_926_codecut

- Run ID: `run20_depth12_926_codecut`
- Log path: `/workspace/parameter-golf/logs/run20_depth12_926_codecut.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run20_depth12_926_codecut.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run20_depth12_926_codecut.final_model.int8.ptz`
- Command: `NUM_LAYERS=12`, `ITERATIONS=926`, trimmed `train_gpt.py`, hybrid serializer
- What I expected: the reclaimed headroom might allow one more pure-depth step while staying under the true total-size cap
- What happened: it stayed legal at `15,998,362` total submission bytes, but it scored slightly worse than `run19` at `1.36003188` roundtrip val_bpb
- What I learned: `926` is not the new winner; `925` is the cleaner cap-edge stopping point for the trimmed script

## Batch 5 Summary

- A pure code-size reduction from `47,642` to `40,354` bytes plus a hybrid serializer was the right intervention once the search became artifact-bound.
- `run19_depth12_925_codecut` is now the best valid run overall at `1.35971082` roundtrip val_bpb and `15,993,279` total submission bytes.

## Batch 17

| Run | Hypothesis | Change | Final train_loss | Final val_loss | Final val_bpb | Final roundtrip val_loss | Final roundtrip val_bpb | Int8+zlib bytes | Step avg ms | Peak memory MiB | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run95_screen_slide1024s64_from_run92 | Sliding-window eval should be the highest-ROI upstream port | `run92` checkpoint, `EVAL_SEQ_LEN=1024`, `EVAL_STRIDE=64` | n/a | 2.2270 | 1.3190 | 2.22891179 | 1.32008711 | 15974060 | 0.02 | 818 | Huge screened gain, not yet code-size-legal on `run92` |
| run96_screen_slide2048s256_from_run92 | Longer-context sliding eval may beat `1024/64` with RoPE extrapolation | `run92` checkpoint, `EVAL_SEQ_LEN=2048`, `EVAL_STRIDE=256` | n/a | partial | 1.3212 pre-quant | n/a | n/a | n/a | n/a | 11449 | Stopped early after losing to `run95` at pre-quant score |
| run97_min_upstream_slide64_from_run28 | The best winning eval policy should produce a legal promoted run on a higher-headroom checkpoint | minified upstream-capable script, `run28` checkpoint, `EVAL_SEQ_LEN=1024`, `EVAL_STRIDE=64` | n/a | 2.2291 | 1.3202 | 2.23128314 | 1.32149156 | 15876848 | 0.03 | 11444 | New best legal run overall |
| run98_min_upstream_slide64_ck9fp16_from_run28 | One late `c_k` FP16 passthrough may improve the legal sliding-eval frontier | `run28` checkpoint plus `INT8_KEEP_FLOAT_LARGE_NAME_PATTERNS=blocks.9.attn.c_k.weight` | n/a | n/a | n/a | n/a | n/a | partial | n/a | n/a | Stopped early to avoid paying the old double-eval cost |
| run99_min_upstream_slide64_ck9ck10_fast_from_run28 | Two late `c_k` FP16 passthrough tensors may recover more score under the fast export-only path | `run28` checkpoint plus `blocks.9/10.attn.c_k.weight` in FP16, compile off, skip pre-quant eval | n/a | skipped | skipped | n/a | n/a | 16142248 | n/a | 80 | Over total cap at `16,174,242` bytes, stopped early |
| run100_10l_slide64_fp16emb_latek_muwd | First upstream-style `10`-layer branch may buy quality per byte with FP16-sensitive export | `NUM_LAYERS=10`, `TIED_EMBED_LR=0.1`, `MUON_WEIGHT_DECAY=0.02`, slide64 eval, FP16 embed + late `c_k` export | n/a | n/a | n/a | n/a | n/a | n/a | n/a | Failed on first attempt due Muon weight-decay implementation bug |
| run101_10l_slide64_fp16emb_latek_muwd | The corrected `10`-layer branch should be operationally viable and competitive enough to continue | same as `run100`, after Muon-WD fix | n/a | 2.2493 | 1.3321 | 2.40777710 | 1.42602122 | 10882226 | 49.33 | 11706 | Completed later on `8xH100`; export gap killed the branch |

## Batch 17 Summary

- `run95` proved that sliding-window eval is the highest-ROI upstream port for this fork so far, and `run97` turned that into a legal promoted run at `1.32149156`.
- The `run28` family does not have enough artifact budget for two FP16 late-`c_k` tensors, so the export-side branch is mostly exhausted.
- The later `8xH100` follow-up on `run101` showed that the `10`-layer branch was operationally fine but not competitive after export, so it did not replace the dense `11`-layer line.
- `run20_depth12_926_codecut` proved that one more step still fits, but not every extra step helps; it stayed legal and still lost to `run19` on score.
- The current best base is `NUM_LAYERS=12`, `ITERATIONS=925`, with the trimmed `train_gpt.py` and hybrid serializer.

## Batch 18

| Run | Hypothesis | Change | Final train_loss | Final val_loss | Final val_bpb | Final roundtrip val_loss | Final roundtrip val_bpb | Int8+zlib bytes | Step avg ms | Peak memory MiB | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run101_10l_slide64_fp16emb_latek_muwd_8xh100_20260320 | The upstream-style `10`-layer branch might buy enough quality per byte to justify the first `8xH100` push | `NUM_LAYERS=10`, `TIED_EMBED_LR=0.1`, `MUON_WEIGHT_DECAY=0.02`, `1024/64` eval, FP16 embed plus late `c_k` passthrough | n/a | 2.2493 | 1.3321 | 2.40777710 | 1.42602122 | 10882226 | 49.33 | 11706 | Training was healthy, but the quantization gap was too large |
| run101_dense_export_control_8xh100_20260320 | Dense export-only control should tell us whether `run101` merely picked the wrong exporter | saved `run101` checkpoint, `ITERATIONS=0`, `INT4_NAME_PATTERNS=` | n/a | skipped | skipped | 3.65026163 | 2.16189055 | 16094636 | n/a | n/a | Dense control was even worse and over the `16 MB` cap |
| run102_11l_dense_slide64_lzma_p4_8xh100_20260320 | A safer dense `11`-layer family should use the pod budget more sensibly than `run101` | dense `11`-layer family, `1024/64`, `lzma+p4`, periodic validation still enabled | partial | n/a | 1.2198 at step 5400 | n/a | n/a | n/a | ~65.62 by step 5600 | n/a | Strong research signal, but not a strict leaderboard-faithful run |
| run103_track_dense11_train_strictwall_8xh100_20260320 | First strict split run should prove real `10` minute training once wallclock is measured correctly | `STRICT_WALLCLOCK=1`, `MAX_WALLCLOCK_SECONDS=600`, `VAL_LOSS_EVERY=0`, `SKIP_POST_TRAIN_EVAL=1` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | Overran because training-only mode still forced a final validation |
| run106_track_dense11_train_strict588_clean_8xh100_20260320d | Reducing the internal cap should keep the full process under `10` real minutes | same as `run103`, but `MAX_WALLCLOCK_SECONDS=588` after fixing the last-step-validation bug | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | First clean competition-faithful training run; external wallclock `598s` |
| run107_eval_dense_slide64_from_run106_8xh100_20260320d | Dense export from the strict timed checkpoint might still be promotable | export-only eval from `run106`, `INT4_NAME_PATTERNS=` | n/a | skipped | skipped | 3.71032873 | 2.19746567 | 19260692 | n/a | n/a | Eval time was legal, but artifact bytes and score were both bad |
| run108_eval_mixedint4_from_run106_8xh100_20260320d | Default mixed-`int4` might legalize the strict timed checkpoint | export-only eval from `run106`, default mixed-`int4` | n/a | skipped | skipped | 4.08271236 | 2.41801223 | 13188288 | n/a | n/a | Fully legal on time and bytes, but quality collapsed |
| run109_eval_mixedint4_tokemb_ck_from_run106_8xh100_20260320d | Selective float passthrough might recover some of the mixed-`int4` loss | export-only eval from `run106`, mixed-`int4` plus `tok_emb` and late `c_k` float passthrough | n/a | skipped | skipped | 4.05366229 | 2.40080714 | 13700420 | n/a | n/a | Best legal artifact from the timed checkpoint, still far from frontier |

### run101_10l_slide64_fp16emb_latek_muwd_8xh100_20260320

- Run ID: `run101_10l_slide64_fp16emb_latek_muwd_8xh100_20260320`
- Command family: upstream-style `10`-layer branch on `8xH100`
- What I expected: the smaller branch might preserve enough quality to justify more serious multi-GPU budget
- What happened: pre-quant came in at `1.3321`, but final exact roundtrip fell to `1.42602122`
- What I learned: the branch was healthy to train, but not healthy to export

### run101_dense_export_control_8xh100_20260320

- Run ID: `run101_dense_export_control_8xh100_20260320`
- Command family: dense export-only control from the saved `run101` checkpoint
- What I expected: this would tell us whether the failure was caused mainly by the mixed-`int4` default path
- What happened: final exact roundtrip collapsed to `2.16189055` and total submission size rose above the cap
- What I learned: `run101` itself was the problem, not just the exporter choice

### run102_11l_dense_slide64_lzma_p4_8xh100_20260320

- Run ID: `run102_11l_dense_slide64_lzma_p4_8xh100_20260320`
- Command family: safer dense `11`-layer `8xH100` research run
- What I expected: this would give better signal than `run101` while we clarified the timing rule
- What happened: validation improved from `1.4070` at step `600` to `1.2198` at step `5400`, but periodic validation made the run unsuitable as a clean competition rehearsal
- What I learned: the model family was healthier than `run101`, but the trainer still needed a real wallclock split mode

### run103_track_dense11_train_strictwall_8xh100_20260320

- Run ID: `run103_track_dense11_train_strictwall_8xh100_20260320`
- Command family: first strict training-only attempt with true wallclock
- What I expected: this would be the first clean `10` minute training run
- What happened: the process still ran too long because training-only mode accidentally forced a final validation; internal timing reached `644757ms` and external wallclock reached `656s`
- What I learned: the training-only path also had to disable last-step validation, and wrapper overhead needed headroom

### run104 and run105

- Status: intentionally aborted cleanup attempts while fixing the strict-timing process flow
- What I learned: treat them as operational cleanup, not modeling evidence

### run106_track_dense11_train_strict588_clean_8xh100_20260320d

- Run ID: `run106_track_dense11_train_strict588_clean_8xh100_20260320d`
- Command family: corrected strict training-only run with wrapper headroom
- What I expected: this should finally produce a competition-faithful training process
- What happened: internal training-only exit landed at `588125ms`, external wallclock landed at `598s`, and the checkpoint was saved successfully
- What I learned: the split workflow now works for real, and `588` internal seconds is the safe practical cap with the current launcher

### run107 to run109

- Run IDs:
  - `run107_eval_dense_slide64_from_run106_8xh100_20260320d`
  - `run108_eval_mixedint4_from_run106_8xh100_20260320d`
  - `run109_eval_mixedint4_tokemb_ck_from_run106_8xh100_20260320d`
- Command family: separate export/eval runs from the strict `run106` checkpoint
- What I expected: one of the export policies might produce a competitive legal artifact
- What happened:
  - dense eval was time-valid but byte-illegal and still poor on score
  - mixed-`int4` evals were legal on time and bytes but catastrophically worse on score
- What I learned: `run106` was an infrastructure success, not a new frontier checkpoint

## Batch 18 Summary

- The session successfully turned the fork into a real split-budget `8xH100` workflow: training under `10` real minutes and evaluation under `10` real minutes both now work.
- The safe strict-training cap with the current `torchrun` launch pattern is about `588` internal seconds, not `600`, because startup overhead costs real wallclock before `main()` starts timing.
- `run101` is no longer a serious frontier candidate. It was operationally healthy but failed badly on both mixed-export and dense-control paths.
- `run106` proved that a dense `11`-layer family can train within the competition rule, but its checkpoint exported terribly: dense was byte-illegal and mixed-`int4` destroyed quality.
- The best legal result in the repo is still `run97` at `1.32149156`.

## Batch 6

| Run | Hypothesis | Change | Final train_loss | Final val_loss | Final val_bpb | Final roundtrip val_loss | Final roundtrip val_bpb | Int8+zlib bytes | Step avg ms | Peak memory MiB | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run21_depth11_1100_codecut | One less layer may spend artifact budget better if trained longer | `NUM_LAYERS=11`, `ITERATIONS=1100`, trimmed `train_gpt.py`, hybrid serializer | 2.1830 | 2.2903 | 1.3564 | 2.29244696 | 1.35771622 | 15433606 | 545.55 | 13241 | New best valid run overall |
| run22_dim480_1100_codecut | Modest width reduction may buy a better size-to-quality trade | `MODEL_DIM=480`, `NUM_HEADS=8`, `ITERATIONS=1100`, trimmed `train_gpt.py`, hybrid serializer | 2.2274 | 2.3149 | 1.3710 | 2.31731774 | 1.37244609 | 11274049 | 528.50 | 10531 | Much smaller, but clearly worse on score |
| run23_depth11_1400_codecut | The winning smaller-depth branch may improve further on a longer schedule | `NUM_LAYERS=11`, `ITERATIONS=1400`, trimmed `train_gpt.py`, hybrid serializer | 2.1131 | 2.2728 | 1.3461 | 2.27479807 | 1.34726355 | 16665279 | 533.72 | 13241 | Best raw score so far, but over total cap by `705,633` bytes |
| run24_dim480_1400_codecut | The width-reduced branch may catch up if given more steps | `MODEL_DIM=480`, `NUM_HEADS=8`, `ITERATIONS=1400`, trimmed `train_gpt.py`, hybrid serializer | 2.1658 | 2.2964 | 1.3600 | 2.29857691 | 1.36134672 | 12179721 | 437.67 | 10529 | Valid, but still worse than `run21` |
| run25_depth11_1700_codecut | The winning 11-layer branch may keep improving if pushed to the batch limit | `NUM_LAYERS=11`, `ITERATIONS=1700`, trimmed `train_gpt.py`, hybrid serializer | 2.1015 | 2.2584 | 1.3375 | 2.26014718 | 1.33858647 | 17595783 | 475.74 | 13241 | Best raw score so far, but over total cap by `1,636,137` bytes |

### run21_depth11_1100_codecut

- Run ID: `run21_depth11_1100_codecut`
- Log path: `/workspace/parameter-golf/logs/run21_depth11_1100_codecut.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run21_depth11_1100_codecut.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run21_depth11_1100_codecut.final_model.int8.ptz`
- Command: `NUM_LAYERS=11`, `ITERATIONS=1100`, trimmed `train_gpt.py`, hybrid serializer
- What I expected: one fewer layer might save enough bytes to let a longer run beat the `12 x 512 @ 925` control
- What happened: it reached `1.35771622` roundtrip val_bpb with `15,473,960` total submission bytes
- What I learned: dropping one layer is a better artifact-budget reallocation move than the previous `12-layer` cap-edge strategy

### run22_dim480_1100_codecut

- Run ID: `run22_dim480_1100_codecut`
- Log path: `/workspace/parameter-golf/logs/run22_dim480_1100_codecut.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run22_dim480_1100_codecut.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run22_dim480_1100_codecut.final_model.int8.ptz`
- Command: `MODEL_DIM=480`, `NUM_HEADS=8`, `ITERATIONS=1100`, trimmed `train_gpt.py`, hybrid serializer
- What I expected: a modest width cut might free more bytes per quality point than dropping a whole layer
- What happened: it was dramatically smaller at `11,314,403` total submission bytes, but much worse on score at `1.37244609`
- What I learned: width reduction was too costly in quality for this model family

### run23_depth11_1400_codecut

- Run ID: `run23_depth11_1400_codecut`
- Log path: `/workspace/parameter-golf/logs/run23_depth11_1400_codecut.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run23_depth11_1400_codecut.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run23_depth11_1400_codecut.final_model.int8.ptz`
- Command: `NUM_LAYERS=11`, `ITERATIONS=1400`, trimmed `train_gpt.py`, hybrid serializer
- What I expected: the winning depth-reduced branch might keep improving if trained substantially longer
- What happened: it reached `1.34726355` roundtrip val_bpb, but total submission size int8+zlib jumped to `16,705,633`
- What I learned: the 11-layer branch has a better raw frontier, but it still hits the same artifact wall when pushed too far

### run24_dim480_1400_codecut

- Run ID: `run24_dim480_1400_codecut`
- Log path: `/workspace/parameter-golf/logs/run24_dim480_1400_codecut.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run24_dim480_1400_codecut.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run24_dim480_1400_codecut.final_model.int8.ptz`
- Command: `MODEL_DIM=480`, `NUM_HEADS=8`, `ITERATIONS=1400`, trimmed `train_gpt.py`, hybrid serializer
- What I expected: the faster width-reduced branch might recover enough quality on a longer schedule to become competitive
- What happened: it improved to `1.36134672` roundtrip val_bpb at `12,220,075` total submission bytes, but still lost clearly to `run21`
- What I learned: width reduction buys speed and bytes, but not enough score

### run25_depth11_1700_codecut

- Run ID: `run25_depth11_1700_codecut`
- Log path: `/workspace/parameter-golf/logs/run25_depth11_1700_codecut.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run25_depth11_1700_codecut.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run25_depth11_1700_codecut.final_model.int8.ptz`
- Command: `NUM_LAYERS=11`, `ITERATIONS=1700`, trimmed `train_gpt.py`, hybrid serializer
- What I expected: the winning 11-layer branch might keep improving enough to define a better raw frontier, even if it risked the cap
- What happened: it reached the best raw batch-6 score at `1.33858647` roundtrip val_bpb, but total submission size int8+zlib rose to `17,636,137`
- What I learned: the new branch really is better, but `1700` steps is far beyond the legal stopping point

## Batch 6 Summary

- Batch 6 found a better model-size regime: `11` layers beats both the old `12-layer` control and the narrower `480`-dim branch when evaluated under the submission cap.
- `run21_depth11_1100_codecut` is the new best valid run overall at `1.35771622` roundtrip val_bpb and `15,473,960` total submission bytes.
- `run23` and `run25` showed that the 11-layer branch has a much better raw frontier than the old 12-layer line, but it still becomes invalid when trained too long.
- The current best base is `NUM_LAYERS=11`, `ITERATIONS=1100`, with the trimmed `train_gpt.py` and hybrid serializer.

## Batch 7

| Run | Hypothesis | Change | Final train_loss | Final val_loss | Final val_bpb | Final roundtrip val_loss | Final roundtrip val_bpb | Int8+zlib bytes | Step avg ms | Peak memory MiB | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run26_depth11_1200_codecut | A moderate step bump should improve the new `11-layer` baseline while staying comfortably legal | `NUM_LAYERS=11`, `ITERATIONS=1200`, trimmed `train_gpt.py`, hybrid serializer | 2.1519 | 2.2874 | 1.3547 | 2.28964658 | 1.35605768 | 15878055 | 415.89 | 13241 | New best valid run overall |
| run27_depth11_1250_codecut | The cap edge may still be above `1250` in the `11-layer` regime | `NUM_LAYERS=11`, `ITERATIONS=1250`, trimmed `train_gpt.py`, hybrid serializer | 2.3227 | 2.2831 | 1.3522 | 2.28557398 | 1.35364565 | 16084846 | 431.97 | 13241 | Over total cap by `125,200` bytes |
| run28_depth11_1225_codecut | The legal edge may sit near the original interpolation estimate | `NUM_LAYERS=11`, `ITERATIONS=1225`, trimmed `train_gpt.py`, hybrid serializer | 2.2403 | 2.2851 | 1.3534 | 2.28732429 | 1.35468228 | 15984416 | 462.49 | 13241 | Over total cap by `24,770` bytes |
| run29_depth11_1220_codecut | A smaller refinement under the new bracket may land exactly on the legal edge | `NUM_LAYERS=11`, `ITERATIONS=1220`, trimmed `train_gpt.py`, hybrid serializer | 2.2386 | 2.2858 | 1.3538 | 2.28772143 | 1.35491749 | 15964997 | 530.00 | 13241 | Over total cap by `5,351` bytes |
| run30_depth11_1218_codecut | Two steps below the last miss should recover legality with almost all of the score | `NUM_LAYERS=11`, `ITERATIONS=1218`, trimmed `train_gpt.py`, hybrid serializer | 2.1530 | 2.2858 | 1.3538 | 2.28774078 | 1.35492895 | 15952102 | 506.34 | 13241 | New best valid run overall |

### run26_depth11_1200_codecut

- Run ID: `run26_depth11_1200_codecut`
- Log path: `/workspace/parameter-golf/logs/run26_depth11_1200_codecut.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run26_depth11_1200_codecut.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run26_depth11_1200_codecut.final_model.int8.ptz`
- Command: `NUM_LAYERS=11`, `ITERATIONS=1200`, trimmed `train_gpt.py`, hybrid serializer
- What I expected: a moderate step increase should improve the new `11-layer` winner without threatening the cap
- What happened: it reached `1.35605768` roundtrip val_bpb with `15,918,409` total submission bytes
- What I learned: the `11-layer` branch still had real legal headroom above `1100`, and `1200` immediately became the new best valid checkpoint

### run27_depth11_1250_codecut

- Run ID: `run27_depth11_1250_codecut`
- Log path: `/workspace/parameter-golf/logs/run27_depth11_1250_codecut.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run27_depth11_1250_codecut.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run27_depth11_1250_codecut.final_model.int8.ptz`
- Command: `NUM_LAYERS=11`, `ITERATIONS=1250`, trimmed `train_gpt.py`, hybrid serializer
- What I expected: this would test whether the legal edge was higher than the rough `1228` estimate
- What happened: it improved raw score to `1.35364565` roundtrip val_bpb, but total submission size int8+zlib landed at `16,125,200`
- What I learned: the cap edge is definitely below `1250`, so the search needed to move back downward

### run28_depth11_1225_codecut

- Run ID: `run28_depth11_1225_codecut`
- Log path: `/workspace/parameter-golf/logs/run28_depth11_1225_codecut.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run28_depth11_1225_codecut.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run28_depth11_1225_codecut.final_model.int8.ptz`
- Command: `NUM_LAYERS=11`, `ITERATIONS=1225`, trimmed `train_gpt.py`, hybrid serializer
- What I expected: this should test the original interpolation estimate directly
- What happened: it finished at `1.35468228` roundtrip val_bpb, but total submission size int8+zlib was `16,024,770`, only `24,770` bytes over the cap
- What I learned: the true legal edge is just below `1225`, so the final search should use tiny downward refinements instead of another broad probe

### run29_depth11_1220_codecut

- Run ID: `run29_depth11_1220_codecut`
- Log path: `/workspace/parameter-golf/logs/run29_depth11_1220_codecut.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run29_depth11_1220_codecut.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run29_depth11_1220_codecut.final_model.int8.ptz`
- Command: `NUM_LAYERS=11`, `ITERATIONS=1220`, trimmed `train_gpt.py`, hybrid serializer
- What I expected: a five-step drop from `1225` might land exactly on the legal boundary
- What happened: it came in at `1.35491749` roundtrip val_bpb and `16,005,351` total submission bytes, missing legality by just `5,351` bytes
- What I learned: one more tiny downward move was enough; the final legal answer had to be below `1220`
- Note: the original queue silently launched an obsolete `1230`-step branch after `run28`; I terminated it once the `1225` result made that run obviously uninformative

### run30_depth11_1218_codecut

- Run ID: `run30_depth11_1218_codecut`
- Log path: `/workspace/parameter-golf/logs/run30_depth11_1218_codecut.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run30_depth11_1218_codecut.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run30_depth11_1218_codecut.final_model.int8.ptz`
- Command: `NUM_LAYERS=11`, `ITERATIONS=1218`, trimmed `train_gpt.py`, hybrid serializer
- What I expected: two steps below the `1220` miss should safely recover validity while keeping almost all of the score gain
- What happened: it finished at `1.35492895` roundtrip val_bpb with `15,992,456` total submission bytes
- What I learned: `1218` is the new best legal stop point for the `11-layer` regime and the best valid run overall so far

## Batch 7 Summary

- Batch 7 mapped the `11-layer` cap edge tightly enough to stop guessing: `1250` was clearly invalid, `1225` was close but still over, `1220` missed by only `5,351` bytes, and `1218` restored legality.
- `run30_depth11_1218_codecut` is the new best valid run overall at `1.35492895` roundtrip val_bpb and `15,992,456` total submission bytes.
- The original `1230` probe was superseded by the live bracket and was terminated once it became obvious that it would only waste GPU time.
- The current best base is `NUM_LAYERS=11`, `ITERATIONS=1218`, with the trimmed `train_gpt.py` and hybrid serializer.

## Batch 8

| Run | Hypothesis | Change | Final train_loss | Final val_loss | Final val_bpb | Final roundtrip val_loss | Final roundtrip val_bpb | Int8+zlib bytes | Step avg ms | Peak memory MiB | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run31_scale_quant_only | Scale coding alone may reduce metadata without harming score | log-quantized per-row scales, same `int8` weights, export-only on `run30` checkpoint | N/A | N/A | N/A | 3.05657698 | 1.81027706 | 15953939 | N/A | N/A | Failed badly; slightly larger total artifact and much worse score |
| run32_mixed_int4_int8_ptq | Large MLP matrices may be over-serialized at `int8` | packed `int4` for `mlp.fc.weight` and `mlp.proj.weight`, export-only on `run30` checkpoint | N/A | N/A | N/A | 3.79081733 | 2.24513555 | 10239663 | N/A | N/A | Massive size win, but catastrophic quality loss |
| run33_outlier_aware_mixed_precision | A safer carveout may recover enough quality to make low-bit export viable | packed `int4` only for `mlp.fc.weight`, export-only on `run30` checkpoint | N/A | N/A | N/A | 3.40179935 | 2.01473719 | 13091783 | N/A | N/A | Better than `run32`, still far too weak |
| run34_qat_tail_lowbit | A short QAT tail may teach the model to survive the low-bit exporter | resume `run30`, `200`-step low-LR tail, fake `int4` on `mlp.fc.weight` during training, mixed export at eval | 2.5421 | 2.6838 | 1.5895 | 2.63197345 | 1.55880293 | 13094308 | 1036.83 | 16691 | Large recovery over PTQ-only, but still not competitive with the `int8` control |
| run35_qat_tail_mid5_fc | A gentler low-bit target set may preserve most of the QAT gain while recovering more quality | resume `run30`, `200`-step low-LR tail, fake `int4` only on middle `5` `mlp.fc.weight` layers, mixed export at eval | 2.4751 | 2.5790 | 1.5274 | 2.57480238 | 1.52494300 | 14662556 | 973.85 | 16691 | Best low-bit result so far, still behind the `int8` control |
| run36_qat_tail_mid5_fc_frozen | Freezing non-target params during QAT may preserve the dense model while adapting only the low-bit layers | resume `run30`, `200`-step low-LR tail, fake `int4` on middle `5` `mlp.fc.weight` layers, freeze all non-target params | 2.9057 | 2.9630 | 1.7549 | 2.99661092 | 1.77476178 | 14661399 | 690.11 | 16691 | Much worse; freezing blocked the needed co-adaptation |
| run37_qat_tail_mid3_fc | An even narrower low-bit target set may recover more quality while staying under the cap | resume `run30`, `200`-step low-LR tail, fake `int4` only on middle `3` `mlp.fc.weight` layers, mixed export at eval | 2.4619 | 2.5649 | 1.5191 | 2.56462259 | 1.51891396 | 15179072 | 1015.73 | 16691 | Narrower targeting helped again and slightly beat `run35` |
| run38_qat_tail_mid3_fc_extend | The best selective-QAT branch may keep improving under a continuation tail | resume `run37`, `200`-step lower-LR continuation, same middle `3` target set | 2.4047 | 2.5133 | 1.4885 | 2.51148655 | 1.48744380 | 15180121 | 990.31 | 16691 | Strong continuation win; best low-bit result so far at that point |
| run39_qat_tail_mid3_fc_extend2 | A second lower-LR continuation may keep reducing the low-bit gap | resume `run38`, `200`-step lower-LR continuation, same middle `3` target set | 2.3849 | 2.4961 | 1.4783 | 2.49324550 | 1.47664042 | 15180433 | 1009.56 | 16691 | New best low-bit result overall |
| run40_export_mid2_from_run39 | Spending extra headroom by exporting fewer low-bit layers from the adapted checkpoint may buy back score cheaply | export-only from `run39`, quantize only `2` of the adapted middle layers | N/A | N/A | N/A | 2.49689159 | 1.47879984 | 15436564 | N/A | 1333 | Slightly worse than `run39`; export-mask-only tweak did not help |
| run41_qat_tail_mid2_fc | A smaller `2`-layer target set may have a better long-run quality ceiling than the `3`-layer branch | resume `run30`, `200`-step low-LR tail, fake `int4` only on `blocks.5-6.mlp.fc.weight` | 2.4546 | 2.5574 | 1.5146 | 2.55921892 | 1.51571360 | 15437580 | 1003.83 | 16691 | Good new branch start, but still behind the continued `run39` line |
| run42_qat_tail_mid2_fc_extend | The new `2`-layer branch may need the same continuation treatment that made the `3`-layer branch work | resume `run41`, `200`-step lower-LR continuation, same middle `2` target set | 2.3982 | 2.5063 | 1.4844 | 2.50701062 | 1.48479290 | 15437916 | 1011.16 | 16691 | Strong continuation win; now close to `run39` |
| run43_qat_tail_mid2_fc_extend2 | A second continuation may decide whether the `2`-layer branch can overtake the current best low-bit line | resume `run42`, `200`-step lower-LR continuation, same middle `2` target set | 2.3785 | 2.4888 | 1.4740 | 2.48890649 | 1.47407062 | 15438498 | 1009.92 | 16691 | New best low-bit result overall |
| run44_qat_tail_mid2_fc_extend3 | One more ultra-low-LR continuation may still squeeze out a capstone gain before the branch flattens | resume `run43`, `200`-step ultra-low-LR continuation, same middle `2` target set | 2.3698 | 2.4813 | 1.4696 | 2.48108463 | 1.46943807 | 15438410 | 1010.33 | 16691 | Best low-bit result so far; gains are smaller but still real |
| run45_qat_tail_mid2_fc_extend4 | Another ultra-low-LR continuation can confirm whether the branch is still improving or has effectively plateaued | resume `run44`, `200`-step ultra-low-LR continuation, same middle `2` target set | 2.3657 | 2.4781 | 1.4677 | 2.47742446 | 1.46727031 | 15438213 | 1015.70 | 16691 | New best low-bit result overall, but the gain is now small |

### run31_scale_quant_only

- Run ID: `run31_scale_quant_only`
- Log path: `/workspace/parameter-golf/logs/run31_scale_quant_only.txt`
- Artifact path:
  - `/workspace/parameter-golf/artifacts/run31_scale_quant_only.final_model.int8.ptz`
- Checkpoint source: `/workspace/parameter-golf/artifacts/run30_depth11_1218_codecut.final_model.pt`
- Command style: export-only evaluation of the saved `run30` checkpoint
- What I expected: quantizing scale metadata might recover bytes with almost no quality change
- What happened: total submission size was `15,995,663`, only a tiny size change, while roundtrip score collapsed to `1.81027706`
- What I learned: this form of scale coding is not viable; metadata tricks alone are not enough here

### run32_mixed_int4_int8_ptq

- Run ID: `run32_mixed_int4_int8_ptq`
- Log path: `/workspace/parameter-golf/logs/run32_mixed_int4_int8_ptq.txt`
- Artifact path:
  - `/workspace/parameter-golf/artifacts/run32_mixed_int4_int8_ptq.final_model.int8.ptz`
- Checkpoint source: `/workspace/parameter-golf/artifacts/run30_depth11_1218_codecut.final_model.pt`
- Command style: export-only evaluation of the saved `run30` checkpoint
- What I expected: packed `int4` on the largest MLP matrices might buy a large size win, even if some quality was lost
- What happened: total submission size dropped to `10,282,161`, but roundtrip score collapsed to `2.24513555`
- What I learned: the model is highly compressible in principle, but naive post-training `int4` export is far too destructive

### run33_outlier_aware_mixed_precision

- Run ID: `run33_outlier_aware_mixed_precision`
- Log path: `/workspace/parameter-golf/logs/run33_outlier_aware_mixed_precision.txt`
- Artifact path:
  - `/workspace/parameter-golf/artifacts/run33_outlier_aware_mixed_precision.final_model.int8.ptz`
- Checkpoint source: `/workspace/parameter-golf/artifacts/run30_depth11_1218_codecut.final_model.pt`
- Command style: export-only evaluation of the saved `run30` checkpoint
- What I expected: keeping `mlp.proj.weight` on the safer path while sending only `mlp.fc.weight` to `int4` might recover enough quality to make mixed precision plausible
- What happened: total submission size landed at `13,134,281`, but roundtrip score was still only `2.01473719`
- What I learned: a simple tensor-level carveout helps relative to `run32`, but plain PTQ low-bit export is still nowhere near competitive

### run34_qat_tail_lowbit

- Run ID: `run34_qat_tail_lowbit`
- Log path: `/workspace/parameter-golf/logs/run34_qat_tail_lowbit.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run34_qat_tail_lowbit.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run34_qat_tail_lowbit.final_model.int8.ptz`
- Initialization source: `/workspace/parameter-golf/artifacts/run30_depth11_1218_codecut.final_model.pt`
- Command style: `200`-step low-LR fine-tuning tail with fake `int4` applied during training to `mlp.fc.weight`, then exported with the same mixed low-bit serializer as `run33`
- What I expected: if the low-bit path is salvageable, the model should recover a substantial fraction of the PTQ-only loss once it is trained to expect quantization noise
- What happened: the final dense checkpoint reached `1.5895 val_bpb`, and the exported mixed low-bit artifact improved sharply to `1.55880293` roundtrip val_bpb at `13,138,419` total submission bytes
- What I learned: QAT is directionally correct and recovers a lot of the PTQ-only damage, but this specific low-bit recipe is still far from threatening the `1.35492895` `int8` control

### run35_qat_tail_mid5_fc

- Run ID: `run35_qat_tail_mid5_fc`
- Log path: `/workspace/parameter-golf/logs/run35_qat_tail_mid5_fc.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run35_qat_tail_mid5_fc.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run35_qat_tail_mid5_fc.final_model.int8.ptz`
- Initialization source: `/workspace/parameter-golf/artifacts/run30_depth11_1218_codecut.final_model.pt`
- Command style: `200`-step low-LR fine-tuning tail with fake `int4` applied during training only to the middle `5` `mlp.fc.weight` layers, then exported with the same mixed low-bit serializer
- What I expected: if the low-bit branch is mostly a targeting problem, a smaller set of fake-quantized layers should retain much more quality while still preserving real size savings
- What happened: the dense checkpoint reached `1.5274 val_bpb`, and the exported mixed low-bit artifact improved further to `1.52494300` roundtrip val_bpb at `14,706,667` total submission bytes
- What I learned: the low-bit branch is not dead. Gentler targeting helps substantially, but the current recipe is still not close enough to displace the `int8` control

### run36_qat_tail_mid5_fc_frozen

- Run ID: `run36_qat_tail_mid5_fc_frozen`
- Log path: `/workspace/parameter-golf/logs/run36_qat_tail_mid5_fc_frozen.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run36_qat_tail_mid5_fc_frozen.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run36_qat_tail_mid5_fc_frozen.final_model.int8.ptz`
- Initialization source: `/workspace/parameter-golf/artifacts/run30_depth11_1218_codecut.final_model.pt`
- Command style: `200`-step low-LR QAT tail with fake `int4` on the middle `5` `mlp.fc.weight` layers, but freeze all non-target parameters during training
- What I expected: if most of the dense model is already good, adapting only the quantized targets might preserve the base while still teaching those layers to survive export
- What happened: the dense checkpoint fell to `1.7549 val_bpb`, and the exported artifact regressed to `1.77476178` roundtrip val_bpb at `14,706,446` total submission bytes
- What I learned: freezing non-target parameters is the wrong direction here. The rest of the network needs to move during the QAT tail

### run37_qat_tail_mid3_fc

- Run ID: `run37_qat_tail_mid3_fc`
- Log path: `/workspace/parameter-golf/logs/run37_qat_tail_mid3_fc.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run37_qat_tail_mid3_fc.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run37_qat_tail_mid3_fc.final_model.int8.ptz`
- Initialization source: `/workspace/parameter-golf/artifacts/run30_depth11_1218_codecut.final_model.pt`
- Command style: `200`-step low-LR QAT tail with fake `int4` applied only to the middle `3` `mlp.fc.weight` layers, then exported with the mixed low-bit serializer
- What I expected: cutting the target set from `5` layers to `3` might recover more quality while still preserving useful size savings
- What happened: the dense checkpoint reached `1.5191 val_bpb`, and the exported artifact improved slightly to `1.51891396` roundtrip val_bpb at `15,224,119` total submission bytes
- What I learned: narrowing the target set again helped, and the middle `3` layers became the new best low-bit starting point

### run38_qat_tail_mid3_fc_extend

- Run ID: `run38_qat_tail_mid3_fc_extend`
- Log path: `/workspace/parameter-golf/logs/run38_qat_tail_mid3_fc_extend.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run38_qat_tail_mid3_fc_extend.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run38_qat_tail_mid3_fc_extend.final_model.int8.ptz`
- Initialization source: `/workspace/parameter-golf/artifacts/run37_qat_tail_mid3_fc.final_model.pt`
- Command style: `200`-step lower-LR continuation of `run37` with the same middle `3` target set
- What I expected: if the selective-QAT branch is still undertrained, a continuation tail should keep reducing the low-bit gap without changing the byte footprint much
- What happened: the dense checkpoint improved to `1.4885 val_bpb`, and the exported artifact improved materially to `1.48744380` roundtrip val_bpb at `15,225,168` total submission bytes
- What I learned: the middle-`3` branch is not just cleaner at step `200`; it continues to improve with more low-LR QAT

### run39_qat_tail_mid3_fc_extend2

- Run ID: `run39_qat_tail_mid3_fc_extend2`
- Log path: `/workspace/parameter-golf/logs/run39_qat_tail_mid3_fc_extend2.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run39_qat_tail_mid3_fc_extend2.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run39_qat_tail_mid3_fc_extend2.final_model.int8.ptz`
- Initialization source: `/workspace/parameter-golf/artifacts/run38_qat_tail_mid3_fc_extend.final_model.pt`
- Command style: second `200`-step lower-LR continuation of the middle `3` target set
- What I expected: if the slope is still real, one more continuation should keep improving the low-bit branch
- What happened: the dense checkpoint reached `1.4783 val_bpb`, and the exported artifact improved again to `1.47664042` roundtrip val_bpb at `15,225,480` total submission bytes
- What I learned: continuation is working. This is now the best low-bit result overall, though still behind the dense `int8` control

### run40_export_mid2_from_run39

- Run ID: `run40_export_mid2_from_run39`
- Log path: `/workspace/parameter-golf/logs/run40_export_mid2_from_run39.txt`
- Checkpoint source: `/workspace/parameter-golf/artifacts/run39_qat_tail_mid3_fc_extend2.final_model.pt`
- Command style: export-only evaluation of the saved `run39` checkpoint, but quantize only `2` of the previously adapted middle layers
- What I expected: because `run39` still had ample headroom, exporting fewer low-bit layers might buy back score without another training pass
- What happened: total submission size rose to `15,481,611`, but the exported score was `1.47879984`, slightly worse than the original `run39` export
- What I learned: export-mask-only tweaks are not enough. If we want to change the low-bit target set, that change needs its own QAT tail

### run41_qat_tail_mid2_fc

- Run ID: `run41_qat_tail_mid2_fc`
- Log path: `/workspace/parameter-golf/logs/run41_qat_tail_mid2_fc.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run41_qat_tail_mid2_fc.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run41_qat_tail_mid2_fc.final_model.int8.ptz`
- Initialization source: `/workspace/parameter-golf/artifacts/run30_depth11_1218_codecut.final_model.pt`
- Command style: fresh `200`-step low-LR QAT tail with fake `int4` applied only to `blocks.5-6.mlp.fc.weight`
- What I expected: a smaller `2`-layer target set might have a better quality ceiling than the current best `3`-layer branch, even if it starts from a similar place
- What happened: the dense checkpoint reached `1.5146 val_bpb`, and the exported artifact landed at `1.51571360` roundtrip val_bpb with `15,482,627` total submission bytes
- What I learned: the `2`-layer branch is viable and its first-step result is stronger than the earlier `3`-layer and `5`-layer starts, but it still trails the fully continued `run39` branch

### run42_qat_tail_mid2_fc_extend

- Run ID: `run42_qat_tail_mid2_fc_extend`
- Log path: `/workspace/parameter-golf/logs/run42_qat_tail_mid2_fc_extend.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run42_qat_tail_mid2_fc_extend.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run42_qat_tail_mid2_fc_extend.final_model.int8.ptz`
- Initialization source: `/workspace/parameter-golf/artifacts/run41_qat_tail_mid2_fc.final_model.pt`
- Command style: `200`-step lower-LR continuation of `run41` with the same middle `2` target set
- What I expected: if the `2`-layer branch is real, the first continuation should close most of the gap to `run39`
- What happened: the dense checkpoint improved to `1.4844 val_bpb`, and the exported artifact improved to `1.48479290` roundtrip val_bpb at `15,482,963` total submission bytes
- What I learned: the `2`-layer branch does benefit from continuation in exactly the same way the `3`-layer branch did, and it became a serious challenger

### run43_qat_tail_mid2_fc_extend2

- Run ID: `run43_qat_tail_mid2_fc_extend2`
- Log path: `/workspace/parameter-golf/logs/run43_qat_tail_mid2_fc_extend2.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run43_qat_tail_mid2_fc_extend2.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run43_qat_tail_mid2_fc_extend2.final_model.int8.ptz`
- Initialization source: `/workspace/parameter-golf/artifacts/run42_qat_tail_mid2_fc_extend.final_model.pt`
- Command style: second `200`-step lower-LR continuation of the middle `2` target set
- What I expected: this was the decisive comparison run for the `2`-layer branch, because the second continuation is where the `3`-layer line reached `run39`
- What happened: the dense checkpoint reached `1.4740 val_bpb`, and the exported artifact improved to `1.47407062` roundtrip val_bpb at `15,483,545` total submission bytes
- What I learned: the middle-`2` branch has the better continuation frontier. It overtook `run39` and became the best low-bit result overall

### run44_qat_tail_mid2_fc_extend3

- Run ID: `run44_qat_tail_mid2_fc_extend3`
- Log path: `/workspace/parameter-golf/logs/run44_qat_tail_mid2_fc_extend3.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run44_qat_tail_mid2_fc_extend3.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run44_qat_tail_mid2_fc_extend3.final_model.int8.ptz`
- Initialization source: `/workspace/parameter-golf/artifacts/run43_qat_tail_mid2_fc_extend2.final_model.pt`
- Command style: third `200`-step ultra-low-LR continuation of the middle `2` target set
- What I expected: the gains should be smaller now, but one more continuation might still squeeze out a real improvement before the branch flattens
- What happened: the dense checkpoint improved again to `1.4696 val_bpb`, and the exported artifact improved to `1.46943807` roundtrip val_bpb at `15,483,457` total submission bytes
- What I learned: the branch is not flat yet. The improvement is smaller, but it is still real, and the middle-`2` line is now clearly the best low-bit recipe

### run45_qat_tail_mid2_fc_extend4

- Run ID: `run45_qat_tail_mid2_fc_extend4`
- Log path: `/workspace/parameter-golf/logs/run45_qat_tail_mid2_fc_extend4.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run45_qat_tail_mid2_fc_extend4.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run45_qat_tail_mid2_fc_extend4.final_model.int8.ptz`
- Initialization source: `/workspace/parameter-golf/artifacts/run44_qat_tail_mid2_fc_extend3.final_model.pt`
- Command style: fourth `200`-step ultra-low-LR continuation of the middle `2` target set
- What I expected: this was the plateau check. If the branch still had meaningful life, it would improve again, but likely only by a small amount
- What happened: the dense checkpoint improved to `1.4677 val_bpb`, and the exported artifact improved again to `1.46727031` roundtrip val_bpb at `15,483,260` total submission bytes
- What I learned: the branch is still improving, but only incrementally now. Continued gains are likely to be small unless we change the recipe rather than just extending it again

## Batch 8 Summary So Far

- The first three compression runs were enough to rule out the easy paths.
- `run31` showed that scale-metadata tricks alone are not a useful lever in this form.
- `run32` and `run33` showed that low-bit storage has huge byte upside, but plain post-training quantization destroys too much quality even with a conservative carveout.
- `run34` then showed that QAT does help a lot, improving the mixed low-bit path from `2.01473719` to `1.55880293`, but not enough to beat or even approach the `int8` control.
- `run35` improved the mixed low-bit path again to `1.52494300`, showing that gentler target selection is a real lever.
- `run36` ruled out freezing non-target parameters during the QAT tail; that shortcut prevented the model from adapting properly.
- `run37`, `run38`, and `run39` showed that a narrower middle-`3` target set plus continued low-LR QAT is the strongest low-bit recipe so far, improving the exported score to `1.47664042`.
- `run40` ruled out a cheap export-mask-only trick on the adapted checkpoint.
- `run41` showed that a fresh middle-`2` branch is plausible, and `run42`, `run43`, and `run44` showed that it has the best continuation frontier, improving the exported score to `1.46943807`.
- `run45` confirmed the plateau shape: the branch still improved to `1.46727031`, but only by a small increment.
- The current conclusion is that low-bit export is only worth pursuing further through selective QAT-aware targeting and continuation; plain PTQ is ruled out, and the present best low-bit run is `run45_qat_tail_mid2_fc_extend4`.

## Batch 9

| Run | Hypothesis | Change | Final train_loss | Final val_loss | Final val_bpb | Final roundtrip val_loss | Final roundtrip val_bpb | Int8+zlib bytes | Step avg ms | Peak memory MiB | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run46_export_group128_from_run45 | Grouped `int4` scales may improve fidelity without more training | export-only from `run45`, `INT4_GROUP_SIZE=128` | N/A | N/A | N/A | 2.47742595 | 1.46727119 | 15438297 | N/A | 1333 | Essentially flat versus `run45`; grouped export alone was not enough |
| run47_export_group64_from_run45 | A finer grouped recipe may buy more fidelity than `128` | export-only from `run45`, `INT4_GROUP_SIZE=64` | N/A | N/A | N/A | 2.47743549 | 1.46727684 | 15438378 | N/A | 1333 | Also flat and slightly worse than `128` |
| run48_qat_group128_mid2_resume_run45 | Grouped `int4` may need QAT adaptation before it helps | resume `run45`, `200`-step QAT tail, `INT4_GROUP_SIZE=128` | 2.3618 | 2.4748 | 1.4657 | 2.47390291 | 1.46518465 | 15438767 | 1014.83 | 16691 | Real improvement; grouped scales helped after adaptation |
| run49_qat_group128_mid2_clip9999 | The grouped recipe may want a tighter `int4` clip than the rowwise default | resume `run48`, `200`-step QAT tail, `INT4_GROUP_SIZE=128`, `INT4_CLIP_PERCENTILE=99.99` | 2.3583 | 2.4715 | 1.4638 | 2.47081476 | 1.46335568 | 15438655 | 1005.10 | 16691 | Better again; tighter clip improved the grouped recipe |
| run50_qat_group128_mid2_clip9999_extend | The best grouped recipe may still have continuation headroom | resume `run49`, `200`-step lower-LR continuation, same grouped+clipped recipe | 2.3566 | 2.4701 | 1.4629 | 2.46896276 | 1.46225881 | 15438655 | 1006.08 | 16691 | New best low-bit result overall |

### run46_export_group128_from_run45

- Run ID: `run46_export_group128_from_run45`
- Log path: `/workspace/parameter-golf/logs/run46_export_group128_from_run45.txt`
- Artifact path:
  - `/workspace/parameter-golf/artifacts/run46_export_group128_from_run45.final_model.int8.ptz`
- Checkpoint source: `/workspace/parameter-golf/artifacts/run45_qat_tail_mid2_fc_extend4.final_model.pt`
- Command style: export-only evaluation of the saved `run45` checkpoint with grouped `int4` scales and `INT4_GROUP_SIZE=128`
- What I expected: grouped scales might improve fidelity enough to buy a free score win at modest metadata cost
- What happened: the exported score was `1.46727119`, effectively identical to `run45`, at `15,484,582` total submission bytes
- What I learned: grouped scales are not a free exporter win on their own. The model needs to adapt to the grouped recipe

### run47_export_group64_from_run45

- Run ID: `run47_export_group64_from_run45`
- Log path: `/workspace/parameter-golf/logs/run47_export_group64_from_run45.txt`
- Artifact path:
  - `/workspace/parameter-golf/artifacts/run47_export_group64_from_run45.final_model.int8.ptz`
- Checkpoint source: `/workspace/parameter-golf/artifacts/run45_qat_tail_mid2_fc_extend4.final_model.pt`
- Command style: export-only evaluation of the saved `run45` checkpoint with grouped `int4` scales and `INT4_GROUP_SIZE=64`
- What I expected: a finer grouped setting might buy more fidelity than `128`, even if it cost a little more scale metadata
- What happened: the exported score was `1.46727684`, slightly worse than `run46`, and the total submission size was also slightly larger at `15,484,663`
- What I learned: `64` is not the better grouped export setting here. `128` wins the probe on both score and bytes

### run48_qat_group128_mid2_resume_run45

- Run ID: `run48_qat_group128_mid2_resume_run45`
- Log path: `/workspace/parameter-golf/logs/run48_qat_group128_mid2_resume_run45.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run48_qat_group128_mid2_resume_run45.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run48_qat_group128_mid2_resume_run45.final_model.int8.ptz`
- Initialization source: `/workspace/parameter-golf/artifacts/run45_qat_tail_mid2_fc_extend4.final_model.pt`
- Command style: `200`-step grouped-QAT continuation with `INT4_GROUP_SIZE=128` on the same middle `2` target set
- What I expected: if grouped scales are the right fidelity lever, they should help once the model is allowed to adapt to the new quantization noise
- What happened: the dense checkpoint improved to `1.4657 val_bpb`, and the exported artifact improved to `1.46518465` roundtrip val_bpb at `15,485,052` total submission bytes
- What I learned: grouped `int4` is useful, but only after QAT adaptation. This was the first real batch-9 recipe win

### run49_qat_group128_mid2_clip9999

- Run ID: `run49_qat_group128_mid2_clip9999`
- Log path: `/workspace/parameter-golf/logs/run49_qat_group128_mid2_clip9999.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run49_qat_group128_mid2_clip9999.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run49_qat_group128_mid2_clip9999.final_model.int8.ptz`
- Initialization source: `/workspace/parameter-golf/artifacts/run48_qat_group128_mid2_resume_run45.final_model.pt`
- Command style: `200`-step grouped-QAT continuation with `INT4_GROUP_SIZE=128` and `INT4_CLIP_PERCENTILE=99.99`
- What I expected: grouped scales might still be overreacting to rare outliers, so a slightly tighter `int4` clip could improve post-export fidelity
- What happened: the dense checkpoint improved to `1.4638 val_bpb`, and the exported artifact improved to `1.46335568` roundtrip val_bpb at `15,484,940` total submission bytes
- What I learned: the grouped recipe does want its own clip setting. `99.99` is clearly better than the inherited rowwise default on this branch

### run50_qat_group128_mid2_clip9999_extend

- Run ID: `run50_qat_group128_mid2_clip9999_extend`
- Log path: `/workspace/parameter-golf/logs/run50_qat_group128_mid2_clip9999_extend.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run50_qat_group128_mid2_clip9999_extend.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run50_qat_group128_mid2_clip9999_extend.final_model.int8.ptz`
- Initialization source: `/workspace/parameter-golf/artifacts/run49_qat_group128_mid2_clip9999.final_model.pt`
- Command style: `200`-step lower-LR continuation of the grouped+clipped recipe
- What I expected: if the batch-9 recipe change is real, it should still have some continuation headroom beyond the initial adaptation
- What happened: the dense checkpoint improved to `1.4629 val_bpb`, and the exported artifact improved again to `1.46225881` roundtrip val_bpb at `15,484,940` total submission bytes
- What I learned: batch 9 was a real recipe success. Grouped `128` scales plus `99.99` clipping beat the old rowwise low-bit branch and still left continuation headroom

## Batch 9 Summary

- `run46` and `run47` showed that grouped scales are not a free exporter win, and that `128` is the better grouped setting of the two quick probes.
- `run48` proved that grouped scales do help once QAT adapts to them, improving the low-bit branch from `1.46727031` to `1.46518465`.
- `run49` showed that the grouped recipe wants a tighter clip setting, improving the exported score further to `1.46335568`.
- `run50` confirmed that the grouped+clipped recipe still has continuation headroom and established the new best low-bit result at `1.46225881`.
- The low-bit branch is still well behind the dense/int8 control, but batch 9 was successful because it improved the branch through a recipe change rather than just more of the old continuation schedule.

## Batch 10

| Run | Hypothesis | Change | Final train_loss | Final val_loss | Final val_bpb | Final roundtrip val_loss | Final roundtrip val_bpb | Int8+zlib bytes | Step avg ms | Peak memory MiB | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run51_qat_group128_mid2_clip9999_from_run23 | The best grouped low-bit recipe may transfer to a stronger dense frontier | start from `run23`, grouped `128`, clip `99.99`, fresh `200`-step adaptation tail | 2.4692 | 2.5762 | 1.5258 | 2.57860518 | 1.52719524 | 16148547 | 996.42 | 16691 | Invalid and much worse; frontier transfer failed |
| run52_export_group128_mid2_clip9999_from_run25 | The best saved dense checkpoint may survive the grouped low-bit recipe without adaptation | export-only from `run25`, grouped `128`, clip `99.99` | N/A | N/A | N/A | 3.08749565 | 1.82858883 | 17069528 | N/A | 1333 | Catastrophic and invalid |
| run53_qat_group128_mid1_block5_from_run50 | Spending headroom on fidelity by quantizing only one middle layer may beat the two-layer grouped branch | resume `run50`, quantize only `blocks.5.mlp.fc.weight`, grouped `128`, clip `99.99` | 2.3516 | 2.4654 | 1.4602 | 2.46540708 | 1.46015294 | 15695619 | 1005.12 | 16691 | Real improvement and still legal |
| run54_qat_group128_mid1_block6_from_run50 | The other middle layer may be the better one-layer target | resume `run50`, quantize only `blocks.6.mlp.fc.weight`, grouped `128`, clip `99.99` | 2.3520 | 2.4644 | 1.4596 | 2.46527763 | 1.46007628 | 15694819 | 1012.92 | 16691 | Slightly better than `run53` on score and size |
| run55_qat_group128_mid1_block6_extend | The winning one-layer branch may still have continuation headroom | resume `run54`, lower-LR continuation, same one-layer grouped+clipped recipe | 2.3500 | 2.4628 | 1.4586 | 2.46360394 | 1.45908502 | 15695109 | 1010.56 | 16691 | New best low-bit result overall |

### run51_qat_group128_mid2_clip9999_from_run23

- Run ID: `run51_qat_group128_mid2_clip9999_from_run23`
- Log path: `/workspace/parameter-golf/logs/run51_qat_group128_mid2_clip9999_from_run23.txt`
- Initialization source: `/workspace/parameter-golf/artifacts/run23_depth11_1400_codecut.final_model.pt`
- Command style: fresh `200`-step grouped-QAT adaptation from the stronger dense `run23` checkpoint
- What I expected: a better dense frontier might transfer into a better low-bit branch if the grouped recipe is strong enough
- What happened: the exported score regressed to `1.52719524`, and total submission size ballooned to `16,194,832`, making the branch invalid
- What I learned: stronger dense checkpoints can transfer much worse compressibility under the low-bit recipe. This frontier-transfer path is not promising in its current form

### run52_export_group128_mid2_clip9999_from_run25

- Run ID: `run52_export_group128_mid2_clip9999_from_run25`
- Log path: `/workspace/parameter-golf/logs/run52_export_group128_mid2_clip9999_from_run25.txt`
- Artifact path:
  - `/workspace/parameter-golf/artifacts/run52_export_group128_mid2_clip9999_from_run25.final_model.int8.ptz`
- Checkpoint source: `/workspace/parameter-golf/artifacts/run25_depth11_1700_codecut.final_model.pt`
- Command style: export-only grouped+clipped evaluation of the strongest saved dense checkpoint
- What I expected: if frontier transfer was viable at all, the raw `run25` checkpoint might at least show a useful export-only floor
- What happened: it collapsed to `1.82858883` roundtrip val_bpb and `17,115,813` total submission bytes
- What I learned: aggressive frontier transfer is not even close. The dense frontier is not the lever we should be pulling next

### run53_qat_group128_mid1_block5_from_run50

- Run ID: `run53_qat_group128_mid1_block5_from_run50`
- Log path: `/workspace/parameter-golf/logs/run53_qat_group128_mid1_block5_from_run50.txt`
- Initialization source: `/workspace/parameter-golf/artifacts/run50_qat_group128_mid2_clip9999_extend.final_model.pt`
- Command style: `200`-step QAT tail that keeps the grouped+clipped recipe but quantizes only `blocks.5.mlp.fc.weight`
- What I expected: if the low-bit branch still has spare size budget, quantizing only one layer should buy back score more effectively than a two-layer target
- What happened: the exported score improved to `1.46015294` at `15,741,904` total submission bytes
- What I learned: spending more of the low-bit budget on fidelity works. One-layer targeting beat the batch-9 winner

### run54_qat_group128_mid1_block6_from_run50

- Run ID: `run54_qat_group128_mid1_block6_from_run50`
- Log path: `/workspace/parameter-golf/logs/run54_qat_group128_mid1_block6_from_run50.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run54_qat_group128_mid1_block6_from_run50.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run54_qat_group128_mid1_block6_from_run50.final_model.int8.ptz`
- Initialization source: `/workspace/parameter-golf/artifacts/run50_qat_group128_mid2_clip9999_extend.final_model.pt`
- Command style: same one-layer idea as `run53`, but quantize only `blocks.6.mlp.fc.weight`
- What I expected: the exact middle layer might matter, and the second candidate could be slightly more robust under low-bit compression
- What happened: the exported score improved to `1.46007628` at `15,741,104` total submission bytes, slightly beating `run53` on both metrics
- What I learned: the one-layer idea is real, and `blocks.6.mlp.fc.weight` is the better target of the two

### run55_qat_group128_mid1_block6_extend

- Run ID: `run55_qat_group128_mid1_block6_extend`
- Log path: `/workspace/parameter-golf/logs/run55_qat_group128_mid1_block6_extend.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run55_qat_group128_mid1_block6_extend.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run55_qat_group128_mid1_block6_extend.final_model.int8.ptz`
- Initialization source: `/workspace/parameter-golf/artifacts/run54_qat_group128_mid1_block6_from_run50.final_model.pt`
- Command style: lower-LR continuation of the winning one-layer grouped+clipped branch
- What I expected: if the one-layer path is the right new base, it should still have some continuation headroom
- What happened: the exported score improved again to `1.45908502` at `15,741,394` total submission bytes
- What I learned: batch 10 found a better low-bit direction than batch 9. The current best low-bit base is now the one-layer grouped+clipped branch centered on `blocks.6.mlp.fc.weight`

## Batch 10 Summary

- The original batch-10 hypothesis failed: stronger dense frontiers did not survive the grouped low-bit recipe and often became invalid due to much worse compressibility.
- The replacement idea worked: spending the spare low-bit budget on fidelity by quantizing only one middle layer beat the batch-9 winner.
- `run54` identified `blocks.6.mlp.fc.weight` as the better single-layer target, and `run55` improved it further to `1.45908502`.
- The new best low-bit run is `run55_qat_group128_mid1_block6_extend` at `1.45908502`, with total submission size `15,741,394`.

## Batch 11

| Run | Hypothesis | Change | Final train_loss | Final val_loss | Final val_bpb | Final roundtrip val_loss | Final roundtrip val_bpb | Int8+zlib bytes | Step avg ms | Peak memory MiB | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run56_export_group32_block6_from_run55 | The one-layer branch may mostly be exporter-limited, and finer grouped scales could recover fidelity without retraining | export-only from `run55`, one-layer `blocks.6.mlp.fc.weight`, `INT4_GROUP_SIZE=32`, `INT4_CLIP_PERCENTILE=99.99` | N/A | 2.3213 | 1.3748 | 2.32207600 | 1.37526421 | 15695265 | 0.02 | 690 | Massive exporter win; much better than `run55` |
| run57_export_group16_block6_from_run55 | An even finer group size may beat `32` on the same checkpoint with only a tiny metadata cost | export-only from `run55`, one-layer `blocks.6.mlp.fc.weight`, `INT4_GROUP_SIZE=16`, `INT4_CLIP_PERCENTILE=99.99` | N/A | 2.3213 | 1.3748 | 2.32205343 | 1.37525084 | 15695322 | 0.01 | 690 | Slightly better than `run56`; `group16` won the probe |
| run58_qat_group16_block6_from_run55 | The winning `group16` exporter may still benefit from a short adaptation tail | resume `run55`, `200`-step QAT tail, one-layer `blocks.6.mlp.fc.weight`, `INT4_GROUP_SIZE=16`, `INT4_CLIP_PERCENTILE=99.99` | 2.1874 | 2.3197 | 1.3738 | 2.31903252 | 1.37346168 | 15695546 | 747.69 | 13241 | Real improvement over export-only `group16` |
| run59_qat_group16_block6_clip999 | Finer grouped scales may want a tighter clip than `99.99` | resume `run58`, `200`-step QAT tail, one-layer `blocks.6.mlp.fc.weight`, `INT4_GROUP_SIZE=16`, `INT4_CLIP_PERCENTILE=99.9` | 2.1864 | 2.3183 | 1.3730 | 2.31824024 | 1.37299245 | 15695364 | 734.25 | 13241 | Small but real improvement; tighter clip helped |
| run60_qat_group16_block6_clip999_extend | The best batch-11 branch may still have continuation headroom at lower LR | resume `run59`, lower-LR `200`-step QAT tail, one-layer `blocks.6.mlp.fc.weight`, `INT4_GROUP_SIZE=16`, `INT4_CLIP_PERCENTILE=99.9` | 2.1858 | 2.3177 | 1.3727 | 2.31784492 | 1.37275832 | 15695238 | 639.79 | 13241 | New best low-bit result overall |

### run56_export_group32_block6_from_run55

- Run ID: `run56_export_group32_block6_from_run55`
- Log path: `/workspace/parameter-golf/logs/run56_export_group32_block6_from_run55.txt`
- Artifact path:
  - `/workspace/parameter-golf/artifacts/run56_export_group32_block6_from_run55.final_model.int8.ptz`
- Checkpoint source: `/workspace/parameter-golf/artifacts/run55_qat_group128_mid1_block6_extend.final_model.pt`
- Command style: export-only evaluation of the saved `run55` dense checkpoint with one-layer `int4`, `INT4_GROUP_SIZE=32`, and `INT4_CLIP_PERCENTILE=99.99`
- What I expected: the batch-10 branch might have been bottlenecked more by coarse grouped scales than by the underlying weights
- What happened: the exported score jumped from `1.45908502` in `run55` to `1.37526421` at `15,741,550` total submission bytes
- What I learned: the saved `run55` weights were much better than the old QAT-time evaluation suggested. The exporter, not the learned weights, was the main problem

### run57_export_group16_block6_from_run55

- Run ID: `run57_export_group16_block6_from_run55`
- Log path: `/workspace/parameter-golf/logs/run57_export_group16_block6_from_run55.txt`
- Artifact path:
  - `/workspace/parameter-golf/artifacts/run57_export_group16_block6_from_run55.final_model.int8.ptz`
- Checkpoint source: `/workspace/parameter-golf/artifacts/run55_qat_group128_mid1_block6_extend.final_model.pt`
- Command style: export-only evaluation of the same one-layer branch, but with `INT4_GROUP_SIZE=16`
- What I expected: if the exporter was still too coarse at `32`, `16` should buy a little more fidelity at a tiny metadata cost
- What happened: it improved again to `1.37525084`, slightly beating `run56` while staying well under the cap
- What I learned: `group16` is the better grouped setting for the current one-layer low-bit recipe

### run58_qat_group16_block6_from_run55

- Run ID: `run58_qat_group16_block6_from_run55`
- Log path: `/workspace/parameter-golf/logs/run58_qat_group16_block6_from_run55.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run58_qat_group16_block6_from_run55.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run58_qat_group16_block6_from_run55.final_model.int8.ptz`
- Initialization source: `/workspace/parameter-golf/artifacts/run55_qat_group128_mid1_block6_extend.final_model.pt`
- Command style: `200`-step low-LR QAT tail with one-layer `int4`, `INT4_GROUP_SIZE=16`, and `INT4_CLIP_PERCENTILE=99.99`
- What I expected: the better exporter might still benefit from a small adaptation tail once the model sees the new quantization noise during training
- What happened: the exported score improved to `1.37346168` at `15,741,831` total submission bytes
- What I learned: the finer grouped recipe still has adaptation headroom, and the improvement is not just an export-only fluke

### run59_qat_group16_block6_clip999

- Run ID: `run59_qat_group16_block6_clip999`
- Log path: `/workspace/parameter-golf/logs/run59_qat_group16_block6_clip999.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run59_qat_group16_block6_clip999.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run59_qat_group16_block6_clip999.final_model.int8.ptz`
- Initialization source: `/workspace/parameter-golf/artifacts/run58_qat_group16_block6_from_run55.final_model.pt`
- Command style: `200`-step continuation of the `group16` branch with a tighter `INT4_CLIP_PERCENTILE=99.9`
- What I expected: once scales are fine-grained enough, the best clip setting may become slightly tighter than the `99.99` value that worked for `group128`
- What happened: the exported score improved to `1.37299245`, and the total submission size also dropped slightly to `15,741,649`
- What I learned: the finer-group recipe wants its own clip tuning. `99.9` is better than `99.99` for this branch

### run60_qat_group16_block6_clip999_extend

- Run ID: `run60_qat_group16_block6_clip999_extend`
- Log path: `/workspace/parameter-golf/logs/run60_qat_group16_block6_clip999_extend.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run60_qat_group16_block6_clip999_extend.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run60_qat_group16_block6_clip999_extend.final_model.int8.ptz`
- Initialization source: `/workspace/parameter-golf/artifacts/run59_qat_group16_block6_clip999.final_model.pt`
- Command style: lower-LR `200`-step continuation of the best batch-11 recipe
- What I expected: if the branch is still optimization-limited rather than exporter-limited, one more lower-LR tail should still improve the final roundtrip score
- What happened: the exported score improved again to `1.37275832` at `15,741,523` total submission bytes
- What I learned: the finer-scale one-layer branch still has continuation headroom. Batch 11 did not just find a better exporter; it found a better low-bit regime

## Batch 11 Summary

- Batch 11 was a major success. The missing lever was finer grouped scales, not a brand-new compression family.
- `run56` and `run57` showed that the saved `run55` dense checkpoint was already strong, and that most of the old low-bit loss came from coarse `group128` scale granularity.
- `run58`, `run59`, and `run60` showed that once the exporter is good enough, QAT can still refine the branch further.
- The new best low-bit run is `run60_qat_group16_block6_clip999_extend` at `1.37275832`, with total submission size `15,741,523`.
- The low-bit branch is now only about `0.01783 val_bpb` behind the dense/int8 control `run30_depth11_1218_codecut` at `1.35492895`.

## Batch 12

| Run | Hypothesis | Change | Final train_loss | Final val_loss | Final val_bpb | Final roundtrip val_loss | Final roundtrip val_bpb | Int8+zlib bytes | Step avg ms | Peak memory MiB | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run61_export_group16_block6_from_run30 | The gentler one-layer low-bit recipe may transfer cleanly to the current dense/int8 control | export-only from `run30`, one-layer `blocks.6.mlp.fc.weight`, `group16`, clip `99.9` | N/A | 2.2858 | 1.3538 | 2.29781591 | 1.36089601 | 15696155 | 0.05 | 690 | Legal, but transfer loss was too large |
| run62_export_group16_block6_from_run29 | The same recipe may work even better on the slightly stronger but invalid `run29` checkpoint | export-only from `run29`, same one-layer MLP target | N/A | 2.2858 | 1.3538 | 2.29815861 | 1.36109898 | 15708682 | 0.02 | 690 | Worse than `run61` |
| run63_export_group16_block6_from_run28 | The strongest near-cap dense checkpoint may be the best transfer source | export-only from `run28`, same one-layer MLP target | N/A | 2.2851 | 1.3534 | 2.29744621 | 1.36067705 | 15726902 | 0.01 | 690 | Best export-only dense transfer |
| run64_qat_group16_block6_from_run28 | A short full-model QAT tail may recover enough transfer loss to make the `run28` branch competitive | `200`-step QAT adaptation from `run28`, same one-layer MLP target | 2.1456 | 2.2851 | 1.3534 | 2.29633147 | 1.36001684 | 15726973 | 562.75 | 13241 | Improved over `run63`, but still too far from `run30` |

## Batch 12 Summary

- Batch 12 showed that the batch-11 one-layer recipe does transfer to stronger dense checkpoints, but not cleanly enough on a full `mlp.fc.weight` target.
- `run28` was the best dense source, and `run64` confirmed that adaptation helps, but only by about `0.00066 val_bpb`.
- The real lesson was about target size, not transfer viability: the low-bit recipe was gentle enough, but the chosen tensor was still too invasive.
- That directly motivated batch 13: search for the smallest tensor that buys the `24.8 KB` needed to legalize `run28`.

## Batch 13

| Run | Hypothesis | Change | Final train_loss | Final val_loss | Final val_bpb | Final roundtrip val_loss | Final roundtrip val_bpb | Int8+zlib bytes | Step avg ms | Peak memory MiB | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run65_export_group16_ck_block6_from_run28 | A smaller `c_k` matrix may save enough bytes with less harm than an MLP target | export-only, `blocks.6.attn.c_k.weight`, `group16`, clip `99.9` | N/A | 2.2851 | 1.3534 | 2.28834991 | 1.35528972 | 15921002 | 0.01 | 690 | Huge improvement over the MLP target |
| run66_export_group16_cv_block6_from_run28 | The matching `c_v` target may be even less harmful | export-only, `blocks.6.attn.c_v.weight`, `group16`, clip `99.9` | N/A | 2.2851 | 1.3534 | 2.28899035 | 1.35566902 | 15920265 | 0.02 | 690 | Worse than `c_k` |
| run67_export_group16_ck_block5_from_run28 | The best `c_k` target may be one block earlier | export-only, `blocks.5.attn.c_k.weight`, `group16`, clip `99.9` | N/A | 2.2851 | 1.3534 | 2.28842970 | 1.35533697 | 15921140 | 0.01 | 690 | Slightly worse than block 6 |
| run68_export_group16_ck_block7_from_run28 | The best `c_k` target may be one block later | export-only, `blocks.7.attn.c_k.weight`, `group16`, clip `99.9` | N/A | 2.2851 | 1.3534 | 2.28824450 | 1.35522728 | 15921161 | 0.05 | 690 | Better than blocks 5 and 6 |
| run69_export_group8_ck_block7_from_run28 | Finer group size may improve the best `c_k` target | export-only, `blocks.7.attn.c_k.weight`, `group8`, clip `99.9` | N/A | 2.2851 | 1.3534 | 2.28824687 | 1.35522869 | 15921302 | 0.08 | 690 | No meaningful gain over `group16` |
| run70_export_group8_ck_block6_from_run28 | Finer group size may improve the block-6 `c_k` target too | export-only, `blocks.6.attn.c_k.weight`, `group8`, clip `99.9` | N/A | 2.2851 | 1.3534 | 2.28833272 | 1.35527953 | 15921140 | 0.02 | 690 | No meaningful gain over `group16` |
| run71_qat_only_ck_block7_from_run28 | Target-only QAT may improve the best small target without perturbing the rest of the checkpoint | `200`-step target-only QAT on `blocks.7.attn.c_k.weight` | 2.1373 | 2.2851 | 1.3534 | 2.28824597 | 1.35522815 | 15921168 | 535.20 | 2865 | No real improvement |
| run72_export_group16_ck_block8_from_run28 | Later decoder `c_k` targets may be even less harmful | export-only, `blocks.8.attn.c_k.weight`, `group16`, clip `99.9` | N/A | 2.2851 | 1.3534 | 2.28810700 | 1.35514585 | 15921715 | 0.06 | 690 | New leader at the time |
| run73_export_group16_ck_block9_from_run28 | The next decoder `c_k` target may be the real sweet spot | export-only, `blocks.9.attn.c_k.weight`, `group16`, clip `99.9` | N/A | 2.2851 | 1.3534 | 2.28806880 | 1.35512322 | 15921569 | 0.02 | 690 | Best target-family result |
| run74_export_group16_ck_block10_from_run28 | The final decoder `c_k` target may continue the trend | export-only, `blocks.10.attn.c_k.weight`, `group16`, clip `99.9` | N/A | 2.2851 | 1.3534 | 2.28957299 | 1.35601409 | 15921836 | 0.01 | 690 | Clearly worse |
| run75_qat_ck_block9_from_run28 | Full-model QAT may recover the last few ten-thousandths on the best `c_k` target | `200`-step full-model QAT on `blocks.9.attn.c_k.weight` | 2.1363 | 2.2858 | 1.3538 | 2.28850519 | 1.35538168 | 15921656 | 766.23 | 13241 | QAT hurt this branch |
| run76_export_group16_ck_block9_clip9999_from_run28 | The best `c_k` target may want a looser clip | export-only, `blocks.9.attn.c_k.weight`, `group16`, clip `99.99` | N/A | 2.2851 | 1.3534 | 2.28805884 | 1.35511732 | 15921586 | 0.02 | 690 | Slightly better than `99.9` |
| run77_export_group16_ck_block9_clip995_from_run28 | The best `c_k` target may want a slightly tighter clip instead | export-only, `blocks.9.attn.c_k.weight`, `group16`, clip `99.5` | N/A | 2.2851 | 1.3534 | 2.28805381 | 1.35511435 | 15921493 | 0.01 | 690 | Best batch-13 result |
| run78_export_group16_ck_block9_clip990_from_run28 | Keep tightening clip to see whether the exporter still has slack | export-only, `blocks.9.attn.c_k.weight`, `group16`, clip `99.0` | N/A | 2.2851 | 1.3534 | 2.28809354 | 1.35513788 | 15921403 | 0.01 | 690 | Slight regression from `99.5` |

## Batch 13 Summary

- Batch 13 found the right tensor family: late decoder key projections, not MLPs.
- The best target in this sweep was `blocks.9.attn.c_k.weight`, which made `run28` legal at `15,967,778` bytes with `1.35511435` roundtrip `val_bpb`.
- `group16` was already sufficient, and QAT did not help this minimal target line.
- The current best low-bit legalization of `run28` is `run77_export_group16_ck_block9_clip995_from_run28`.
- The overall best valid run in the repo is still the dense/int8 control `run30_depth11_1218_codecut` at `1.35492895`, but the gap is now only about `0.000185 val_bpb`.

## Post-Batch 13 Notes

- `run79_export_group16_proj_block9_from_run28`
  - export-only, `blocks.9.attn.proj.weight`, `group16`, clip `99.5`
  - roundtrip `val_bpb: 1.35518553`
  - total submission size int8+zlib: `15,903,025`
  - result: worse than the `blocks.9.attn.c_k.weight` line

- `run80_export_group16_cq_block9_from_run28`
  - export-only, `blocks.9.attn.c_q.weight`, `group16`, clip `99.5`
  - roundtrip `val_bpb: 1.35524874`
  - total submission size int8+zlib: `15,903,833`
  - result: also worse than the `blocks.9.attn.c_k.weight` line

- Code-golf feasibility check
  - current [train_gpt.py](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/train_gpt.py) is `46,285` bytes
  - automated `python-minifier` on the current script gets it to about `25,344` bytes
  - a crude dense-only/QAT-stripped estimate drops the raw script to about `38,964` bytes before minification
  - takeaway: a submission-specific dense-only `run28` trainer looks plausible enough to justify a dedicated code-golf batch

## Batch 14

| Run | Hypothesis | Change | Final train_loss | Final val_loss | Final val_bpb | Final roundtrip val_loss | Final roundtrip val_bpb | Int8+zlib bytes | Step avg ms | Peak memory MiB | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run81b_current_from_run29 | The current repo may already reproduce the saved dense frontier directly | export-only from `run29` with current `train_gpt.py` defaults | N/A | 2.2858 | 1.3538 | 3.04070474 | 1.80087662 | 10252545 | 0.01 | 690 | Failed dense-fidelity diagnostic; default export was still using mixed `int4` |
| run81c_current_dense_from_run29 | Clearing the `int4` target list should restore the historical dense exporter exactly | export-only from `run29` with `INT4_NAME_PATTERNS=` | N/A | 2.2858 | 1.3538 | 2.28772143 | 1.35491749 | 15964994 | 0.02 | 690 | Exact dense control restored |
| run82_minified_dense_from_run29 | Minified dense-only packaging should legalize `run29` by code size alone | `train_gpt_min.py`, export-only from `run29`, `INT4_NAME_PATTERNS=` | N/A | 2.2858 | 1.3538 | 2.28772143 | 1.35491749 | 15964994 | 0.02 | 690 | New best valid run overall |
| run83_minified_dense_from_run28 | The same dense-only code-golf path may be enough for the stronger `run28` checkpoint too | `train_gpt_min.py`, export-only from `run28`, `INT4_NAME_PATTERNS=` | N/A | 2.2851 | 1.3534 | 2.28732429 | 1.35468228 | 15984432 | 0.01 | 690 | Still over the cap by `9,778` bytes |

### run81b_current_from_run29

- Run ID: `run81b_current_from_run29`
- Log path: `/workspace/parameter-golf/logs/run81b_current_from_run29.txt`
- Artifact path: `/workspace/parameter-golf/artifacts/run81b_current_from_run29.final_model.int8.ptz`
- Command style: export-only from saved `run29` checkpoint using the current repo defaults
- What I expected: a quick control check before committing to code golf
- What happened: the artifact shrank to `10,252,545` bytes, but roundtrip score collapsed to `1.80087662`
- What I learned: the current exporter had drifted away from the historical dense path because it still applied default mixed `int4` export unless `INT4_NAME_PATTERNS=` was cleared

### run81c_current_dense_from_run29

- Run ID: `run81c_current_dense_from_run29`
- Log path: `/workspace/parameter-golf/logs/run81c_current_dense_from_run29.txt`
- Artifact path: `/workspace/parameter-golf/artifacts/run81c_current_dense_from_run29.final_model.int8.ptz`
- Command style: export-only from saved `run29` checkpoint using current `train_gpt.py` with `INT4_NAME_PATTERNS=`
- What I expected: disabling the implicit `int4` target list should restore the saved dense frontier
- What happened: it reproduced the exact historical `run29` dense result, including `1.35491749` roundtrip `val_bpb`
- What I learned: the dense frontier was still intact; the problem was exporter drift, not checkpoint quality

### run82_minified_dense_from_run29

- Run ID: `run82_minified_dense_from_run29`
- Log path: `/workspace/parameter-golf/logs/run82_minified_dense_from_run29.txt`
- Artifact path: `/workspace/parameter-golf/artifacts/run82_minified_dense_from_run29.final_model.int8.ptz`
- Command style: export-only from saved `run29` checkpoint using `train_gpt_min.py` with `INT4_NAME_PATTERNS=`
- What I expected: once the dense exporter was restored, code-size reduction alone should make `run29` legal
- What happened: it preserved the exact `1.35491749` dense roundtrip score and landed at `15,990,340` total submission bytes
- What I learned: `run29` is now the best valid dense run overall, and dense-only code golf is a real path rather than just a feasibility idea

### run83_minified_dense_from_run28

- Run ID: `run83_minified_dense_from_run28`
- Log path: `/workspace/parameter-golf/logs/run83_minified_dense_from_run28.txt`
- Artifact path: `/workspace/parameter-golf/artifacts/run83_minified_dense_from_run28.final_model.int8.ptz`
- Command style: export-only from saved `run28` checkpoint using `train_gpt_min.py` with `INT4_NAME_PATTERNS=`
- What I expected: the same minified dense exporter might also be enough for the strongest saved dense checkpoint
- What happened: it preserved the exact `1.35468228` dense roundtrip score, but total submission size was still `16,009,778` bytes
- What I learned: `run28` remains the best dense checkpoint, and the remaining gap is now a much smaller packaging problem of `9,778` bytes

## Batch 14 Summary

- Batch 14 validated the dense-only code-golf strategy and immediately produced a new best valid run: `run82_minified_dense_from_run29` at `1.35491749` roundtrip `val_bpb`.
- The decisive diagnostic was that the current repo had silently drifted into mixed `int4` export by default; clearing `INT4_NAME_PATTERNS=` restored the historical dense frontier exactly.
- `run83` shows that the higher-upside dense target is still `run28`, and the remaining gap is now only `9,778` bytes.
- The best next move is a purpose-built dense-only submission script that removes the low-bit/QAT branches cleanly rather than relying on a generic minifier.

## Batch 15

| Run | Hypothesis | Change | Final train_loss | Final val_loss | Final val_bpb | Final roundtrip val_loss | Final roundtrip val_bpb | Int8+compressed bytes | Step avg ms | Peak memory MiB | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run84_minified_dense_p5_from_run28 | A newer pickle protocol may save enough bytes to legalize `run28` | `train_gpt_min_p5.py`, `pickle_protocol=5`, `zlib` | N/A | 2.2851 | 1.3534 | N/A | N/A | 15979730 | 0.02 | 690 | Artifact improved, but still `5,094` bytes over and blocked by `weights_only=True` |
| run85_minified_dense_oldser_from_run28 | Old Torch serialization may compress better than the default zip format | `train_gpt_min_oldser.py`, legacy Torch save, `zlib` | N/A | 2.2851 | 1.3534 | 2.28732429 | 1.35468228 | 15996414 | 0.02 | 690 | Worse than the default minified dense exporter |
| run86_minified_dense_oldser_p5_from_run28 | Combining legacy serialization with protocol 5 may beat either one alone | `train_gpt_min_oldser_p5.py`, legacy Torch save, `pickle_protocol=5`, `zlib` | N/A | 2.2851 | 1.3534 | N/A | N/A | 15987492 | 0.06 | 690 | Improved over `run85`, still worse than plain `p5`, blocked by `weights_only=True` |
| run87_minified_dense_p4_from_run28 | Protocol 4 may match the size win while staying closer to Torch’s supported loaders | `train_gpt_min_p4.py`, `pickle_protocol=4`, `zlib` | N/A | 2.2851 | 1.3534 | N/A | N/A | 15979727 | 0.02 | 690 | Best `zlib` artifact, still `5,091` bytes over and blocked by `weights_only=True` |
| run88_minified_dense_oldser_p4_from_run28 | Combining legacy serialization with protocol 4 may reveal the best of both | `train_gpt_min_oldser_p4.py`, legacy Torch save, `pickle_protocol=4`, `zlib` | N/A | 2.2851 | 1.3534 | N/A | N/A | 15995519 | 0.02 | 690 | Worse than the best zip-based protocol variants |
| run89_minified_dense_lzma_p4_from_run28 | A stronger lossless compressor may turn the best dense checkpoint legal without touching the model | `train_gpt_min_lzma_p4.py`, `pickle_protocol=4`, `lzma` | N/A | 2.2851 | 1.3534 | 2.28732429 | 1.35468228 | 15876848 | 0.02 | 690 | New best valid run overall |

### run84_minified_dense_p5_from_run28

- Run ID: `run84_minified_dense_p5_from_run28`
- Log path: `/workspace/parameter-golf/logs/run84_minified_dense_p5_from_run28.txt`
- Artifact path: `/workspace/parameter-golf/artifacts/run84_minified_dense_p5_from_run28.final_model.int8.ptz`
- Command style: export-only from saved `run28` checkpoint using `train_gpt_min_p5.py`
- What I expected: protocol `5` might shave the remaining bytes if Torch zip serialization overhead was the main blocker
- What happened: it improved the artifact to `15,979,730` bytes, but the total was still `16,005,094` and the roundtrip reload failed under PyTorch 2.6’s default `weights_only=True`
- What I learned: serializer format mattered, but not enough to clear the gap on its own

### run85_minified_dense_oldser_from_run28

- Run ID: `run85_minified_dense_oldser_from_run28`
- Log path: `/workspace/parameter-golf/logs/run85_minified_dense_oldser_from_run28.txt`
- Artifact path: `/workspace/parameter-golf/artifacts/run85_minified_dense_oldser_from_run28.final_model.int8.ptz`
- Command style: export-only from saved `run28` checkpoint using legacy Torch serialization plus `zlib`
- What I expected: the older serialization format might trade metadata for better compressibility
- What happened: it preserved the score exactly, but artifact size regressed to `15,996,414` bytes
- What I learned: the legacy save format was not a free win for the dense path

### run86_minified_dense_oldser_p5_from_run28

- Run ID: `run86_minified_dense_oldser_p5_from_run28`
- Log path: `/workspace/parameter-golf/logs/run86_minified_dense_oldser_p5_from_run28.txt`
- Artifact path: `/workspace/parameter-golf/artifacts/run86_minified_dense_oldser_p5_from_run28.final_model.int8.ptz`
- Command style: export-only from saved `run28` checkpoint using legacy Torch serialization plus protocol `5`
- What I expected: the combination might beat both individual changes
- What happened: it improved on plain legacy save, but still landed at `16,012,893` total bytes and hit the same `weights_only=True` load blocker
- What I learned: the zip-based serializer remained the better base format

### run87_minified_dense_p4_from_run28

- Run ID: `run87_minified_dense_p4_from_run28`
- Log path: `/workspace/parameter-golf/logs/run87_minified_dense_p4_from_run28.txt`
- Artifact path: `/workspace/parameter-golf/artifacts/run87_minified_dense_p4_from_run28.final_model.int8.ptz`
- Command style: export-only from saved `run28` checkpoint using `pickle_protocol=4`
- What I expected: protocol `4` might match or slightly beat protocol `5`
- What happened: it was the best `zlib` artifact at `15,979,727` bytes, but total size was still `16,005,091` and the same PyTorch load issue appeared
- What I learned: protocol `4` was the best serializer variant, reducing the remaining gap to about `5.1 KB`

### run88_minified_dense_oldser_p4_from_run28

- Run ID: `run88_minified_dense_oldser_p4_from_run28`
- Log path: `/workspace/parameter-golf/logs/run88_minified_dense_oldser_p4_from_run28.txt`
- Artifact path: `/workspace/parameter-golf/artifacts/run88_minified_dense_oldser_p4_from_run28.final_model.int8.ptz`
- Command style: export-only from saved `run28` checkpoint using legacy serialization plus protocol `4`
- What I expected: if protocol `4` was the right serializer, it was worth checking the final cross-term with the legacy format
- What happened: it stayed clearly worse than the zip-based `p4` path
- What I learned: the best packaging line was now obvious: zip-based save, protocol `4`, and a better outer compressor

### run89_minified_dense_lzma_p4_from_run28

- Run ID: `run89_minified_dense_lzma_p4_from_run28`
- Log path: `/workspace/parameter-golf/logs/run89_minified_dense_lzma_p4_from_run28.txt`
- Artifact path: `/workspace/parameter-golf/artifacts/run89_minified_dense_lzma_p4_from_run28.final_model.int8.ptz`
- Command style: export-only from saved `run28` checkpoint using `train_gpt_min_lzma_p4.py`, `pickle_protocol=4`, `lzma`, and `weights_only=False` on reload
- What I expected: a stronger lossless compressor might be the cleanest way to spend the remaining few kilobytes
- What happened: it preserved the exact dense roundtrip score and dropped total submission size to `15,902,232`
- What I learned: the packaging barrier on `run28` is gone. The next bottleneck is model quality again, not artifact legality

## Batch 15 Summary

- Batch 15 showed that serializer tweaks alone were useful but insufficient: protocol `4/5` cut the dense `run28` gap roughly in half.
- The decisive move was swapping the outer compressor from `zlib` to `lzma` while keeping the dense quantizer unchanged.
- `run89_minified_dense_lzma_p4_from_run28` is now the best valid run overall at `1.35468228` roundtrip `val_bpb` and `15,902,232` total submission bytes.
- With roughly `97.8 KB` of new headroom, the next step is to reopen the `11`-layer cap-edge step search above `1225`.

## Batch 16

| Run | Hypothesis | Change | Final train_loss | Final val_loss | Final val_bpb | Final roundtrip val_loss | Final roundtrip val_bpb | Int8+compressed bytes | Step avg ms | Peak memory MiB | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run90_depth11_1250_lzma_p4 | The recovered `lzma` headroom may support a materially better dense stop above the old `1225` edge | `NUM_LAYERS=11`, `ITERATIONS=1250`, dense `lzma+p4` packaging | 2.1560 | 2.2833 | 1.3523 | 2.28538953 | 1.35353641 | 15988208 | 532.69 | 13241 | Best raw score in the batch, but over by `13,592` bytes |
| run91_depth11_1246_lzma_p4 | A slightly smaller step count should turn the first lzma probe legal while keeping most of the score gain | `NUM_LAYERS=11`, `ITERATIONS=1246`, dense `lzma+p4` packaging | 2.1533 | 2.2838 | 1.3526 | 2.28586673 | 1.35381904 | 15972748 | 531.67 | 13241 | Legal and better than `run89` |
| run92_depth11_1247_lzma_p4 | The true lzma-adjusted edge may be one step higher than the safe fallback | `NUM_LAYERS=11`, `ITERATIONS=1247`, dense `lzma+p4` packaging | 2.1527 | 2.2829 | 1.3521 | 2.28490481 | 1.35324933 | 15974060 | 536.23 | 13241 | New best valid run overall |
| run93_depth11_1248_lzma_p4 | One more step may still be legal and improve the score again | `NUM_LAYERS=11`, `ITERATIONS=1248`, dense `lzma+p4` packaging | 2.1549 | 2.2833 | 1.3523 | 2.28531651 | 1.35349316 | 15984108 | 534.99 | 13241 | Over by `9,492` bytes and slightly worse than `run92` |

### run90_depth11_1250_lzma_p4

- Run ID: `run90_depth11_1250_lzma_p4`
- Log path: `/workspace/parameter-golf/logs/run90_depth11_1250_lzma_p4.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run90_depth11_1250_lzma_p4.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run90_depth11_1250_lzma_p4.final_model.int8.ptz`
- Command style: fresh dense retraining above the old `1225` edge using the `lzma+p4` packaging path
- What I expected: the recovered packaging headroom should support a meaningfully better dense checkpoint, even if `1250` turned out slightly too large
- What happened: it improved the score sharply to `1.35353641`, but total submission size was `16,013,592`
- What I learned: the packaging win was real, and the new edge moved upward materially, but `1250` itself was still too aggressive

### run91_depth11_1246_lzma_p4

- Run ID: `run91_depth11_1246_lzma_p4`
- Log path: `/workspace/parameter-golf/logs/run91_depth11_1246_lzma_p4.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run91_depth11_1246_lzma_p4.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run91_depth11_1246_lzma_p4.final_model.int8.ptz`
- Command style: safer cap-edge retraining under the same `lzma+p4` packaging path
- What I expected: trimming a few steps from `1250` should restore legality while keeping most of the score gain
- What happened: it landed legal at `15,998,132` total bytes and improved on `run89` to `1.35381904`
- What I learned: the new edge was definitely above `1225`, and the remaining search range was now very tight

### run92_depth11_1247_lzma_p4

- Run ID: `run92_depth11_1247_lzma_p4`
- Log path: `/workspace/parameter-golf/logs/run92_depth11_1247_lzma_p4.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run92_depth11_1247_lzma_p4.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run92_depth11_1247_lzma_p4.final_model.int8.ptz`
- Command style: one-step probe above the safe `1246` fallback
- What I expected: this was the most likely new legal edge
- What happened: it came in legal by only `556` bytes and produced the new best valid run overall at `1.35324933`
- What I learned: `1247` is the current best dense cap edge under the `lzma+p4` packaging path

### run93_depth11_1248_lzma_p4

- Run ID: `run93_depth11_1248_lzma_p4`
- Log path: `/workspace/parameter-golf/logs/run93_depth11_1248_lzma_p4.txt`
- Artifact paths:
  - `/workspace/parameter-golf/artifacts/run93_depth11_1248_lzma_p4.final_model.pt`
  - `/workspace/parameter-golf/artifacts/run93_depth11_1248_lzma_p4.final_model.int8.ptz`
- Command style: final one-step probe above the new `1247` leader
- What I expected: it might barely miss the cap, or reveal that `1247` was not quite the true edge
- What happened: it was over by `9,492` bytes and also slightly worse on score than `run92`
- What I learned: `1247` is a clean stopping point for this neighborhood; the next search should move to a different axis instead of blindly stepping upward again

## Batch 16 Summary

- Batch 16 successfully converted the recovered `lzma` headroom into a stronger dense checkpoint.
- `run90` proved the new packaging path bought real quality, `run91` restored legality, `run92` found the new best valid edge, and `run93` showed the next step up was both invalid and not better.
- The new best valid run overall is `run92_depth11_1247_lzma_p4` at `1.35324933` roundtrip `val_bpb` and `15,999,444` total submission bytes.
- The immediate neighborhood is now mapped tightly enough that the next search should shift away from raw step count and onto a new model-side lever.
