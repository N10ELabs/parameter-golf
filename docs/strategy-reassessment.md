# Strategy Reassessment

Current best-known position has moved beyond the point where this note was first
written. For the live frontier and the current decision tree, use
[current-frontier.md](/Users/anthonymarti/Desktop/N10E%20LABS%20Code/parameter-golf/docs/current-frontier.md).

## Where We Are

The project has now moved through four distinct phases:

1. model and training search
2. low-bit compression research
3. dense exporter recovery
4. serializer and compressor search

The current best valid run overall is:

- `run92_depth11_1247_lzma_p4`
- `final_int8_zlib_roundtrip_exact val_bpb: 1.35324933`
- total submission size int8+lzma: `15,999,444`

The most important comparison points are:

- best valid overall: `run92` at `1.35324933`
- dense fallback: `run91` at `1.35381904`
- recovered dense baseline before the new cap-edge sweep: `run89` at
  `1.35468228`
- low-bit fallback: `run77` at `1.35511435`

So the main line is still dense, and it is now materially stronger than it was
before the `lzma` packaging shift.

## What The Experiments Really Taught Us

### 1. Architecture search already paid off

The structural findings are stable:

- width increase was a bad trade
- deeper models were better than wider ones
- reducing depth from `12` to `11` and spending the budget on longer training
  was the right regime change
- the dense `11`-layer branch remains the best family

### 2. Low-bit work was valuable, but not the winning branch

Low-bit experiments were still useful because they taught us:

- plain PTQ low-bit export was not enough
- grouped scales mattered a lot
- QAT only helped for some larger targets
- minimal late-decoder attention targets were much better than whole-MLP
  quantization

But the strategic conclusion is unchanged:

- low-bit is the fallback path, not the main line

### 3. Dense exporter behavior mattered more than expected

One of the most important discoveries was that the current repo had drifted into
default mixed `int4` export behavior. Clearing `INT4_NAME_PATTERNS=` restored
the historical dense frontier exactly.

That turned several confusing results into a clean dense baseline again.

### 4. The real breakthrough was packaging

Batch 15 showed that:

- serializer choice mattered
- `pickle_protocol=4` was better than the default save path
- the decisive move was swapping the outer compressor from `zlib` to `lzma`

That made the old `run28` dense frontier legal with comfortable headroom.

### 5. Batch 16 already spent most of that headroom

The recovered margin was not theoretical. It bought real model quality:

- `run90` proved `1250` steps was now within reach qualitatively
- `run91` made the new neighborhood legal
- `run92` found the new dense edge at `1247` steps
- `run93` showed the next step up was both over the cap and slightly worse

That means the immediate `1246-1250` neighborhood is now mapped tightly enough.

## Current Bottleneck

The bottleneck is no longer “make `run28` legal.”

It is now:

- how to improve on `run92_depth11_1247_lzma_p4` without blindly pushing the
  same step-count axis

`run92` has only `556` bytes of remaining headroom, so this branch is back to a
true cap edge.

## Best Next Approach

The best next approach is:

1. keep the dense `11`-layer branch as the main line
2. keep the `lzma+p4` packaging path
3. treat `run92` as the new control
4. shift the next search to a different model-side lever

What is no longer the highest-value move:

- more blind step-count probing in the immediate `1246-1250` range
- more generic low-bit experiments
- deeper code golf for its own sake

## Recommended Execution Order

1. Freeze `run92_depth11_1247_lzma_p4` as the new dense control.
2. Keep `run91` as the safer dense fallback and `run77` as the low-bit fallback.
3. Design the next batch around one new model-side change on top of the `run92`
   regime, not around another raw step-count sweep.
4. Only revisit packaging if a clearly better checkpoint misses the cap by a
   small amount.

## Bottom Line

The bigger picture is now:

- dense is still the winning branch
- `lzma+p4` solved the previous packaging bottleneck
- `run92_depth11_1247_lzma_p4` is the new best valid run
- the next progress will come from a fresh model-side idea, not from squeezing
  one more nearby training step out of the same curve
