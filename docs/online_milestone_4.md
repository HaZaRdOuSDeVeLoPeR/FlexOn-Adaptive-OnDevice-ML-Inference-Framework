# Online Phase — Milestone 4: Adaptive Level and Resource Selection

## Goal

Implement the core reactive scheduling mechanics described in the FlexOn paper:

1. adaptive segmentation-level selection using the AD/MI rule;
2. degradation-aware per-segment CPU/CUDA resource selection;
3. background resource-capacity monitoring so scheduler decisions do not
   synchronously invoke monitoring commands.

Recovery allocation is deliberately deferred to the next milestone.

## Paper correspondence

For an inference period, the runtime records the measured segment execution
time `R` and compares it with the offline-profiled expected execution time
`E`.

With the paper parameters:

```text
alpha = 1.2
beta  = 1.6
```

the next level is selected as:

```text
if measured < alpha * expected and level >= 1:
    level = level - 1

else if measured > beta * expected:
    level = Lmax

else:
    level unchanged
```

The resource selector uses a linear degradation envelope between the two
anchor points `(U_min, max_dr)` and `(1, 1)`. For each exact
segment/resource pair, `max_dr` is computed offline as:

```text
max_dr(r, s_i) = contended_mean_ms(r, s_i) / mean_ms(r, s_i)
```

with `U_min = 0.01`. The live degradation for a resource that has not just
executed the segment is:

```text
U = clamp(U_r, U_min, 1)

d_r(U) = max_dr + (max_dr - 1) * (U_min - U) / (1 - U_min)
```

Thus `d_r(1) = 1` and `d_r(U_min) = max_dr`. The maximum ratio is stored
directly in the artifact, so the online scheduler does not need to divide a
contended execution time by the ideal execution time. The remaining capacity
is lower-clamped at `0.01`; using `min(0.01, U_r)` would incorrectly force
all larger capacities to `0.01`.

For the resource that just executed the current segment, the scheduler instead
uses the measured runtime ratio `R_i / C_r(s_i)`, as in the paper. The
supported resource minimizing `d_r * C_r(next_segment)` is selected.

The first segment has no preceding execution measurement, so the normalized
hyperbolic model is used for both resources.

## Runtime modes

Existing deterministic modes remain available:

```text
--resource cpu
--resource cuda
--resource-plan cpu,cuda,...
```

FlexOn dynamic resource selection is enabled with:

```text
--resource auto
```

Adaptive segmentation-level selection is enabled independently with:

```text
--adaptive-level
```

The intended full online mode is therefore:

```text
--resource auto --adaptive-level
```

`--level N` remains the initial segmentation level.

A fixed `--resource-plan` cannot be combined with `--adaptive-level`, because a
fixed plan is tied to one specific level.

## Resource monitoring

CPU remaining capacity is estimated from `/proc/stat`.

CUDA utilization is sampled through `nvidia-smi`. Monitoring occurs on a
background thread so scheduler decisions do not synchronously pay the
`nvidia-smi` startup cost.

If GPU monitoring is unavailable, the CUDA monitor conservatively falls back
to a remaining-capacity value of `1.0`; the scheduler still remains functional
using the offline profile.

## Deliberate limitation

The paper's recovery mechanism is not part of this milestone. In particular,
we do not yet launch a duplicate segment on another resource after detecting
that the selected resource has become unexpectedly slow.

That is the next milestone.

## Validation requirements

Milestone 4 must retain all previous tests:

- CPU-only execution;
- CUDA-only execution;
- mixed CPU↔CUDA execution;
- shared CPU/CUDA arena reuse.

Additional scheduler validation should verify:

- AD/MI level decrease;
- AD/MI jump to `Lmax`;
- no level change inside the hysteresis region;
- first-segment resource selection;
- degradation-aware resource selection;
- dynamic `--resource auto` execution;
- adaptive `--resource auto --adaptive-level` execution.
