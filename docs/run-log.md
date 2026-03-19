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
