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

The resource selector uses the paper's degradation factor:

```text
d_r = R_i / C_r(s_i)    if r is the resource running the current segment

d_r = 1 / U_r           otherwise
```

and selects the supported resource minimizing:

```text
d_r * C_r(next_segment)
```

where `U_r` is the monitored remaining capacity. For non-current resources,
the implementation uses `1 / (U_r + epsilon)` with a small numerical epsilon
only to guard the zero-capacity case. This epsilon is an implementation
safeguard, not a paper parameter.

The first segment has no preceding segment measurement, so the implementation
uses `1/U_r * C_r(first_segment)` as its initial resource-selection score.

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
