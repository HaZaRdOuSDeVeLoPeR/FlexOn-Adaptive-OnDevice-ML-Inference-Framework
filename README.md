# FlexOn — ONNX Runtime C++ Implementation

A modular C++ Implementation of the FlexOn framework proposed in:

> Minsung Kim et al., "Mitigating Resource Contention for Responsive On-device Machine Learning Inferences", IEEE/ACM ICCAD 2025.

## Scope of this repository

The implementation separates the framework into:

- **Offline phase** — model analysis, resource profiling, initial segmentation,
  multi-level segmentation, artifact generation and validation.
- **Online phase** — preloaded segment execution, resource monitoring,
  segmentation-level selection, resource selection, recovery and telemetry.
- **Core** — data-only structures shared by both phases.
- **API** — small public interfaces that integrate each phase.
- **Apps** — `offline_phase` and `flexon` executables.
- **Dashboard** — optional live monitoring/control service.

The execution backend is **ONNX Runtime C++** with CPU and NVIDIA CUDA
Execution Providers. ONNX graph manipulation is intentionally kept separate
from ONNX Runtime: the offline compiler works on ONNX `ModelProto` graphs,
while ONNX Runtime is used to execute and profile generated segment models.

## Important design rule

Do not confuse a runtime framework feature with a FlexOn feature.

ONNX Runtime provides execution-provider graph partitioning, CPU fallback,
CUDA execution, I/O binding, allocators and profiling. FlexOn's
**multi-level segmentation, level selection, degradation model and recovery
policy are implemented by this project**.

## Current status

The offline compiler is implemented through artifact generation:

- ONNX model loading and best-effort shape inference
- structural graph analysis
- configurable CUDA fallback boundaries
- exact ONNX subgraph generation for each segment
- CPU/CUDA segment profiling with ORT sessions
- multi-level balanced segmentation
- YAML offline artifact manifest generation
- artifact validation

The online runtime currently includes:
- preloaded CPU/CUDA segment sessions;
- explicit CPU↔CUDA boundary transfers;
- reusable CPU/CUDA segment-boundary buffer arenas;
- FlexOn AD/MI segmentation-level selection;
- degradation-aware dynamic resource selection with background resource monitoring.

Recovery allocation is the next online milestone.

The CUDA profiling path uses an ORT CUDA-enabled session, but this version does
not claim that successful session creation proves every node was assigned to
CUDA. Explicit fallback operators are therefore configurable in
`config/offline.yaml`.


## Target platform

Initial target:

- Linux / WSL2
- x86-64
- NVIDIA GPU
- CUDA Execution Provider
- CPU + CUDA resources
- C++17 or newer
- CMake
- one GPU initially
- batch-1/static-shape models initially

The design should remain extensible to additional execution providers later.

## Build

```bash
./setup.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFLEXON_ENABLE_WARNINGS=ON
cmake --build build -j"${nproc}"
ctest --test-dir build --output-on-failure
```

Compiler warnings are enabled by default. For GCC/Clang builds,
`FLEXON_ENABLE_WARNINGS=ON` enables `-Wall`, `-Wextra`, `-Wpedantic`,
`-Wunused-variable`, and `-Wunused-function`. Disable them with
`-DFLEXON_ENABLE_WARNINGS=OFF` when needed.

`setup.sh` is intended to provision/pin the third-party dependencies and
download experiment models. Exact dependency versions belong in the project
configuration; do not hard-code experimental constants in C++.

## Experimental reproducibility

Every benchmark should record:

- model hash
- FlexOn configuration hash
- ONNX Runtime version/commit
- ONNX opset/IR version
- CUDA/cuDNN versions
- GPU name and compute capability
- CPU information
- segmentation levels
- profiling iteration counts
- scheduler parameters
- workload parameters
- measured segment latency
- measured boundary-copy latency
- scheduler overhead
- end-to-end latency
- percentile statistics

# FlexOn experiment harness

The experiment layer has two responsibilities with a hard boundary between them.

## Contention environment

`contention.py` is responsible only for creating and removing external resource load.

- `ContentionSpec.none()` — no load
- `ContentionSpec.cpu(workers)` — CPU load
- `ContentionSpec.gpu(command, workers=...)` — GPU load using a caller-supplied long-running workload
- `ContentionSpec.both(...)` — simultaneous CPU and GPU load
- `ContentionSession` — start/stop lifecycle

It does not import or invoke the FlexOn runtime.

## Experiment execution and analysis

- `runner.py` — invokes `flexon_online` and parses its stdout.
- `measurements.py` — extracts authoritative iteration latency and computes statistics.
- `analysis.py` — converts records into comparable dataframes and builds runtime-path diagnostics (resource execution counts, segmentation usage, level changes and recovery activity).
- `plotting.py` — creates overview and focused research figures, including latency distributions, percentile views, resource-execution breakdowns and adaptive-level diagnostics.
- `config.py` — stores benchmark invocation settings.

The runner has no contention parameter. This is intentional: the same benchmark code is used under idle and contended environments.

## Notebook workflow

1. `contention_generator.ipynb`: configure/start/stop external contention.
2. `experiments.ipynb`: run the six FlexOn execution modes, summarize, plot, and save results.

Run the contention notebook in a separate live kernel when contention is required.

The experiment notebook intentionally provides both all-configuration and
FlexOn-only latency plots. The fixed CPU baseline is much slower than the
other configurations, so a focused view is used to keep low-latency boxplots
and percentile curves readable. Runtime-path diagnostics are presented
separately from user-facing latency metrics.

## Reference

The paper's workflow has an offline phase that profiles operators and creates
multi-level segments, followed by an online phase that selects a segmentation
level and resource dynamically and can trigger recovery execution.

## Priority-isolated contention experiments

FlexOn supports an experiment-only priority-isolation mode in which the
dedicated scheduler/control thread runs at the maximum Linux `SCHED_FIFO`
priority while actual inference execution remains at normal `SCHED_OTHER`
priority. Linux scheduling policy is per-thread, so this isolates the control
plane without promoting the ONNX Runtime/CUDA execution threads.

The runtime itself performs the promotion from `OnlineScheduler::start()`. It
first initializes CUDA/ONNX Runtime normally, then the scheduler thread invokes
a tiny, separately installed helper carrying only `CAP_SYS_NICE`. The helper
validates that the target TID belongs to the invoking FlexOn process before
calling `sched_setscheduler()`. `flexon_online` itself never receives
`CAP_SYS_NICE`, and should not be launched through `sudo`.

One-time setup after building:

```bash
./scripts/setup_priority_helper.sh
getcap /usr/local/libexec/flexon-scheduler-priority
```

Expected capability:

```text
/usr/local/libexec/flexon-scheduler-priority cap_sys_nice=ep
```

Run normally; no launcher is required:

```bash
apps/flexon_online \
    --artifact artifacts/resnet18 \
    --iterations 10 \
    --resource auto \
    --priority-isolation
```

The runtime reports the scheduler thread's TID and scheduling policy. If the
helper is missing, not privileged, or fails verification, priority-isolated
execution fails instead of silently running an incorrectly configured
experiment.
