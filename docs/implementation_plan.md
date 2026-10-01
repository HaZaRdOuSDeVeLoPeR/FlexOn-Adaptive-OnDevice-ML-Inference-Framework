# Incremental Implementation Plan

Each step follows:

1. implement one module;
2. write focused unit tests;
3. run tests;
4. add integration test;
5. only then move to the next module.

## Offline order

**Current module: Offline compiler pipeline (graph → segments → profiling → artifacts)**

1. Configuration loader
2. ONNX model loader
3. Graph analyzer + best-effort shape inference
4. Initial segmentation at configured CUDA fallback boundaries
5. Segment ONNX model generator
6. CPU/CUDA segment profiler
7. Multi-level balanced split policy
8. Offline artifact writer
9. Artifact validator
10. Offline facade

## Online order

1. Artifact loader
2. ORT environment/session factory
3. Session pool
4. Tensor/device transport
5. Segment executor
6. Resource monitor
7. Resource selector
8. Level selector
9. Recovery controller
10. Runtime telemetry
11. Online facade

## Performance gates

Before implementing recovery, demonstrate:

- one CPU segment executes correctly;
- one CUDA segment executes correctly;
- CPU and CUDA outputs agree within configured tolerance;
- measured segment cost is stable across repeated runs;
- scheduler overhead is independently measurable;
- no ORT session is constructed during inference;
- boundary-copy cost is visible separately from compute cost.

Only after these are true should the adaptive scheduler be benchmarked.
