# Online Phase — Milestone 1

## Goal

Execute a complete level from a frozen offline artifact using pre-created
ONNX Runtime sessions.

## Scope

This milestone establishes the correctness boundary for the online runtime:

- load and parse `manifest.yaml` (artifact format version 2)
- prepare every supported CPU/CUDA segment session during `load()`
- generate deterministic model inputs from the first segment's ONNX metadata
- execute all segments in level order
- preserve tensors needed by non-adjacent dependencies
- verify that the final segment produces the declared level outputs
- expose CPU, CUDA, and offline-cost-based `auto` resource selection

## Deliberate limitations

This is **not** the final FlexOn scheduler. In particular, this milestone does
not implement:

- I/O binding or explicit device-buffer management
- zero-copy/buffer sharing
- adaptive segmentation-level selection
- runtime resource monitoring
- AD/MI scheduling
- degradation/recovery
- contention experiments

The current executor intentionally uses ordinary ORT `Run()` and clones input
values when necessary to keep tensor lifetime correct across residual/skip
connections. This is a correctness-first implementation and is expected to be
replaced by explicit lifetime-aware buffer management in later milestones.

## CLI

```text
./apps/flexon_online --artifact <dir> [--level N] [--resource cpu|cuda|auto] [--iterations N]
./apps/flexon_online_tester --artifact <dir> [--level N] [--resource cpu|cuda|auto]
```

The default resource is CPU and the default level/iteration count is 0/1.


### Execution backend note

Segment execution uses ONNX Runtime I/O Binding so existing `Ort::Value` inputs can be bound without copying ownership, while outputs are allocated on the selected CPU/CUDA resource. This establishes the tensor/device boundary needed by later online resource switching and buffer-management work.
