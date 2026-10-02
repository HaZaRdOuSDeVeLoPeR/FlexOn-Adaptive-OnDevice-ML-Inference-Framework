# Online Phase — Milestone 3: Shared Buffer Arena

## Goal

Reduce repeated activation allocation by reusing segment-boundary CPU and CUDA
buffers when tensor shape, datatype, device, and lifetime are compatible.

The paper describes sharing buffers across segmentation levels when the same
operator/tensor storage can be reused. The ONNX Runtime implementation uses a
conservative equivalent at the FlexOn/ORT boundary: externally-backed
`Ort::Value` tensors are allocated from a reusable per-run arena and bound
through `IoBinding`. ORT internal activation allocations remain outside the
arena.

## Architecture

```text
segment inputs
      |
      v
Tensor Store <---- RuntimeTensor
      |                 |
      |                 +-- Ort::Value
      |                 +-- BufferLease
      v
BufferArena
   |         |
   v         v
 CPU pool   CUDA pool
```

CPU and CUDA allocations are deliberately kept in separate pools. A physical
allocation is never reused across devices.

## Allocation key

A buffer is reusable only when all of the following match:

- execution resource (CPU or CUDA);
- ONNX tensor element type;
- complete concrete tensor shape;
- resulting byte size.

Dynamic output shapes are not forced through the arena. They retain the ORT
allocation path until a concrete shape is available.

## Lifetime and reuse

`RuntimeTensor` owns both the `Ort::Value` and a `BufferLease`. The `Ort::Value`
is destroyed before the lease returns the backing allocation to the arena.
Segment input liveness is derived from the level's manifest input dependencies.
After the final consumer of a non-final tensor completes, its tensor is erased
from the tensor store and its arena allocation becomes available for reuse.

The arena persists across inference iterations, so released buffers can be
reused by later iterations without another `cudaMalloc`/`malloc`.

## Telemetry

Each segment reports:

- `arena_allocations`: number of backing allocations made by the arena;
- `arena_reuses`: number of allocations served from the free pool;
- `arena_bytes`: cumulative bytes owned by the arena;
- `arena_peak_live_bytes`: maximum simultaneously live arena bytes observed.

These metrics provide the basis for later memory-footprint comparisons.

## Deliberate limitations

This milestone does not implement:

- ORT internal allocator replacement;
- physical CPU/GPU memory sharing;
- adaptive resource selection;
- adaptive segmentation-level selection;
- AD/MI scheduling;
- resource monitoring;
- degradation/recovery.

## Validation

Existing Milestone-2 CPU-only, CUDA-only, and mixed-resource tests remain
regression requirements. In addition, a successful arena run should show
non-zero `arena_reuses` after the first iteration when compatible segment
buffers are released and requested again.
