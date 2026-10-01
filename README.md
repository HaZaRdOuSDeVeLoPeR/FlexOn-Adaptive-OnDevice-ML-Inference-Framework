# FlexOn — ONNX Runtime C++ Reimplementation

A modular C++ reimplementation of the FlexOn framework proposed in:

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

The online scheduler/runtime remains under construction.

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
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"${nproc}"
ctest --test-dir build --output-on-failure
```

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

## Reference

The paper's workflow has an offline phase that profiles operators and creates
multi-level segments, followed by an online phase that selects a segmentation
level and resource dynamically and can trigger recovery execution.
