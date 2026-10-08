# Standardized Contention Generator

FlexOn uses a closed-loop contention generator for controlled resource-contention experiments.
The generator is deliberately independent of the online scheduler and offline profiler so it can
be validated on its own before being integrated into offline profiling.

## Resource targets

The standalone executable accepts measured-load targets:

```bash
./apps/flexon_contention \
    --cpu 60 \
    --gpu 60 \
    --dram 60 \
    --vram 60 \
    --duration 300
```

The values are **target resource load**, not raw stressor duty cycles.

- `--cpu`: system-wide CPU utilization percentage.
- `--gpu`: GPU compute utilization percentage reported by NVML.
- `--dram`: percentage of the safe DRAM bandwidth ceiling.
- `--vram`: percentage of the safe VRAM bandwidth ceiling.

Both memory-bandwidth ceilings are **measured automatically on the current machine** before the
corresponding stressor starts. No DRAM or VRAM bandwidth value is required in
`config/hardware.yaml`.

The DRAM benchmark uses the same streaming `A + B -> C` access pattern as the DRAM stressor,
with randomized buffer contents, a working set larger than LLC, multiple independent worker
streams, and separate warmup/measured phases. Logical bandwidth is calculated as:

```text
(read A + read B + write C) / elapsed time
= (elements × 12 bytes) / elapsed time
```

The VRAM ceiling is measured on the selected CUDA device before the VRAM stressor starts. The
benchmark allocates a large device-memory source and destination buffer, warms up, then performs
repeated device-to-device copies. Bandwidth is calculated as:

```text
(bytes copied × 2 for device read + device write) / elapsed time
```

This mirrors the reference PyTorch measurement used to validate the RTX 4070. The benchmark
uses randomized source contents, measures only the copy phase, and adapts the buffer size if
the foreground process has already consumed significant VRAM. The generator then applies the
configurable safety factor (`0.85` by default) to each measured ceiling.

## Feedback architecture

A dedicated monitor thread samples CPU and GPU utilization and stores the latest measurements.
DRAM/VRAM workers independently measure their achieved bandwidth and publish it into the same
monitor state. Stressor/controller threads query this state and adjust their duty cycle.

The controller uses a deadband and a bounded proportional update. This avoids rapidly chasing
telemetry noise and prevents large duty-cycle jumps.

## Stress workloads

- CPU: randomized FP64/FMA-heavy arithmetic with per-worker PRNG state.
- GPU: randomized FP32/FMA-heavy CUDA kernels launched across all SMs.
- DRAM: randomized large host-memory working sets with read/read/write traffic.
- VRAM: randomized device-memory working sets with read/read/write traffic.
- DRAM ceiling benchmark: same streaming workload as the DRAM stressor, independently measured before stress.
- VRAM ceiling benchmark: large device-to-device copy, independently measured before stress.

Random values are generated outside the critical memory loops where possible so the stressor does
not accidentally benchmark the random-number generator itself.

## Validation procedure

Before using the generator for FlexOn profiling, validate each resource independently:

```text
0%, 20%, 40%, 60%, 80%, 90%
```

For each target, allow the controller to stabilize and record the measured value. Do not start
FlexOn profiling until the requested contention level has stabilized.

The benchmark-only commands are useful for validating the calibration independently:

```bash
./apps/flexon_contention --benchmark-dram
./apps/flexon_contention --benchmark-vram
./apps/flexon_contention --benchmark-memory
```

Only after the four independent stressors are stable should combined workloads be validated.

## Offline integration

The offline phase now reuses the same `flexon_contention_lib` after normal profiling and
segmentation. It waits for the configured stabilization period (10 seconds by default), then
re-profiles every generated segment on each supported resource using the same warmup and
measurement iteration settings as the ideal profile.

Each segment/resource entry stores its own offline maximum degradation ratio as `max_degradation_ratio`:

```text
ideal profiling
      ↓
standardized maximum contention
      ↓
10 s stabilization
      ↓
contended segment profiling
      ↓
per-segment/resource max_degradation_ratio
      ↓
manifest.yaml
```

The slowdown ratio is intentionally **not** collapsed into a model-wide or level-wide constant.
For a segment/resource pair it is:

```text
contended_mean_ms / mean_ms
```

This preserves the resource- and segment-specific behavior needed by the later online scheduler.
