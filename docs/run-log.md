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

## Planned Batch 3

| Run | Hypothesis | Change | Final train_loss | Final val_loss | Final val_bpb | Final roundtrip val_loss | Final roundtrip val_bpb | Int8+zlib bytes | Step avg ms | Peak memory MiB | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run11_depth12_950 | Near-cap pure depth should retain most of the 1000-step gain | `NUM_LAYERS=12`, `ITERATIONS=950` | 2.1956 | 2.2925 | 1.3578 | 2.29530613 | 1.35940958 | 16072708 | 541.51 | 14391 | Strong score, over cap by 72,708 bytes |
| run12_depth12_925 | Slightly earlier pure depth should safely clear the cap | `NUM_LAYERS=12`, `ITERATIONS=925` | 2.1690 | 2.2933 | 1.3582 | 2.29648552 | 1.36010808 | 15953418 | 520.46 | 14391 | New best valid pure-depth run |
| run13_depth12_batch262k_600 | Smaller batch should earn a fair wallclock-matched promotion test | `NUM_LAYERS=12`, `TRAIN_BATCH_TOKENS=262144`, `ITERATIONS=600` | 2.4240 | 2.4352 | 1.4422 | 2.44350054 | 1.44717865 | 13917660 | 507.26 | 7380 | Valid and fast, but worse than `run12` |
| run14_depth12_batch262k_800 | Smaller batch may keep improving without hitting the cap | `NUM_LAYERS=12`, `TRAIN_BATCH_TOKENS=262144`, `ITERATIONS=800` | 2.2973 | 2.3555 | 1.3951 | 2.35893899 | 1.39709654 | 15242441 | 431.74 | 7380 | Strong improvement, still behind `run12` |
| run15_depth12_batch262k_900 | Smaller batch near the cap may become the best valid scorer | `NUM_LAYERS=12`, `TRAIN_BATCH_TOKENS=262144`, `ITERATIONS=900` | 2.3083 | 2.3372 | 1.3842 | 2.33975195 | 1.38573290 | 15760592 | 515.01 | 7380 | Best small-batch result, still behind `run12` |

### run11_depth12_950

- Run ID: `run11_depth12_950`
- Log path: `/workspace/parameter-golf/logs/run11_depth12_950.txt`
- Command: `NUM_LAYERS=12`, `ITERATIONS=950`
- What I expected: this should keep most of the `1000`-step score while slipping just under the size cap
- What happened: it reached `1.35940958` roundtrip val_bpb, but the int8+zlib artifact still landed at `16,072,708` bytes
- What I learned: the pure-depth near-cap slope is steep, and `950` steps is still too aggressive by `72,708` bytes

### run12_depth12_925

- Run ID: `run12_depth12_925`
- Log path: `/workspace/parameter-golf/logs/run12_depth12_925.txt`
- Command: `NUM_LAYERS=12`, `ITERATIONS=925`
- What I expected: this should be the safer under-cap pure-depth checkpoint with only a small score penalty versus `run11`
- What happened: it finished at `1.36010808` roundtrip val_bpb with an int8+zlib artifact of `15,953,418` bytes
- What I learned: `925` steps is a valid near-cap pure-depth checkpoint and a clear improvement over `run05_extend_best`

### run13_depth12_batch262k_600

- Run ID: `run13_depth12_batch262k_600`
- Log path: `/workspace/parameter-golf/logs/run13_depth12_batch262k_600.txt`
- Command: `NUM_LAYERS=12`, `TRAIN_BATCH_TOKENS=262144`, `ITERATIONS=600`
- What I expected: this would be the first fair promotion test for the smaller-batch regime at a comparable wallclock budget
- What happened: it produced `1.44717865` roundtrip val_bpb at `13,917,660` bytes with only `7.38 GiB` peak memory
- What I learned: smaller batch remains efficient and safe, but `600` steps is not enough to beat the new pure-depth baseline

### run14_depth12_batch262k_800

- Run ID: `run14_depth12_batch262k_800`
- Log path: `/workspace/parameter-golf/logs/run14_depth12_batch262k_800.txt`
- Command: `NUM_LAYERS=12`, `TRAIN_BATCH_TOKENS=262144`, `ITERATIONS=800`
- What I expected: the smaller-batch branch might need a longer schedule before it became competitive with the pure-depth winner
- What happened: it improved to `1.39709654` roundtrip val_bpb at `15,242,441` bytes and stayed comfortably under the cap
- What I learned: longer training helps this branch a lot, but it is still not good enough to displace `run12_depth12_925`

### run15_depth12_batch262k_900

- Run ID: `run15_depth12_batch262k_900`
- Log path: `/workspace/parameter-golf/logs/run15_depth12_batch262k_900.txt`
- Command: `NUM_LAYERS=12`, `TRAIN_BATCH_TOKENS=262144`, `ITERATIONS=900`
- What I expected: this near-cap smaller-batch probe would tell us whether the branch could actually overtake the pure-depth winner
- What happened: it reached `1.38573290` roundtrip val_bpb at `15,760,592` bytes, which was the best small-batch result but still clearly behind `run12`
- What I learned: smaller batch is a real efficiency direction, but pure depth at `925` steps is still the stronger scoring base

## Batch 3 Summary

- `run11` and `run12` mapped the pure-depth cap edge cleanly: `950` steps was just over, and `925` steps became the best valid run so far at `1.36010808` roundtrip val_bpb and `15,953,418` bytes.
- `run13`, `run14`, and `run15` proved the smaller-batch branch is efficient and memory-light, improving from `1.44717865` to `1.38573290`, but it still did not catch the pure-depth `925`-step checkpoint on score.
- The batch-4 base should stay `NUM_LAYERS=12`, `ITERATIONS=925` as the strongest valid scorer. The smaller-batch branch remains useful, but right now it looks more like an efficiency fallback than the best leaderboard path.
