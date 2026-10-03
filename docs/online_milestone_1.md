# Online Phase — Milestone 1 / 2

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

## Milestone 2 additions

Milestone 2 makes CPU/CUDA placement explicit at segment boundaries:

- tensors are inspected for their current ORT memory placement
- cross-resource inputs are explicitly copied into the destination resource
- boundary-copy time is measured separately from segment execution time
- segment outputs are allocated on the selected resource
- an explicit per-segment resource plan can exercise CPU/CUDA switching

The runtime still does not implement adaptive scheduling, monitoring, AD/MI,
degradation, or recovery. Those remain later milestones.

The current executor intentionally uses ordinary ORT `Run()` and clones input
values when necessary to keep tensor lifetime correct across residual/skip
connections. This is a correctness-first implementation and is expected to be
replaced by explicit lifetime-aware buffer management in later milestones.

## CLI

The online tester was consolidated into the runtime application. The single
experiment-facing executable is now:

```text
./apps/flexon_online --artifact <dir> [options]
```

Its defaults are 100 iterations, initial level `max_level / 2`, automatic
resource selection, automatic resource plan, and adaptive-level selection
disabled. Scheduler `alpha`, `beta`, and `gamma` may be overridden from the
command line; recovery is enabled explicitly with `--with-recovery`.

The offline tester was likewise consolidated into `flexon_offline`; use
`--validate` for validation-only runs.


### Execution backend note

Segment execution uses ONNX Runtime I/O Binding so existing `Ort::Value` inputs can be bound without copying ownership, while outputs are allocated on the selected CPU/CUDA resource. This establishes the tensor/device boundary needed by later online resource switching and buffer-management work.
