# Online Phase — Milestone 5: Recovery Allocation

## Goal

Implement the recovery allocation mechanism described by FlexOn Eq. (5).
Recovery is deliberately evaluated **during segment execution**, after the
normal scheduler has already selected the primary resource.

The scheduler therefore performs:

```text
level selection / resource selection
            ↓
       primary segment
          execution
            ↓
     recovery monitoring
            ↓
 optional speculative execution
```

Recovery is not a replacement for the normal resource-selection decision.

## Paper correspondence

For the currently executing segment `s_i^l`, FlexOn triggers recovery when:

```text
d_r* * C_r*(s_i^l)
    > gamma * min_{r != r*} { d_r * C_r(s_i^l) }
```

with `gamma = 1.3` in the paper.

During execution, the implementation uses the elapsed execution time as the
current estimate of the left-hand side. This follows the paper's degradation
relationship `d_r* = R_i^l / C_r*(s_i^l)`, for which
`d_r* * C_r*(s_i^l) = R_i^l`.

The alternative-resource score is continuously recomputed from its monitored
remaining capacity:

```text
d_r = 1 / U_r
score = d_r * C_r(s_i^l)
```

A small numerical epsilon is added to `U_r` only to avoid division by zero:

```text
1 / (U_r + epsilon)
```

This epsilon is an implementation safeguard, not a FlexOn paper parameter.

## Execution behavior

When recovery is triggered:

1. The already-running primary execution is not cancelled.
2. A second execution of the same pre-generated segment is launched on the
   selected alternative resource.
3. The first completed execution supplies the segment output.
4. The losing speculative execution is allowed to finish and is ignored.
5. Its resources are reclaimed after completion.

This preserves correctness with ONNX Runtime, which does not provide the
cancellation semantics required to safely kill an in-flight segment.

The runtime waits for speculative executions before destroying the current
iteration's tensor/arena state, but the winning result is selected as soon as
one execution completes.

## Configuration

```yaml
recovery:
  enabled: true
  gamma: 1.3
```

`gamma` is the paper's recovery sensitivity parameter.

No `remaining_capacity_floor` is exposed. For non-current resources the
scheduler uses `1 / (U + epsilon)` directly, with `epsilon` serving only as a
numerical zero-division safeguard.

## Resource selection vs recovery

These are intentionally separate:

- **Resource selection** happens before a segment starts and chooses its
  primary resource.
- **Recovery** happens while that segment is already executing and reacts to
  an unexpected slowdown.

Therefore a GPU workload introduced after a CUDA segment starts can trigger
recovery even though the original resource-selection decision was reasonable.
