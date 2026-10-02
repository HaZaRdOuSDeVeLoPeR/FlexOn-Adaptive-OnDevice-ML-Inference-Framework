# Online Phase — Milestone 2

## Goal

Establish explicit CPU/CUDA tensor placement at segment boundaries and
measure the cost of moving tensors when consecutive segments execute on
different resources.

## Scope

Milestone 2 adds:

- inspection of each runtime tensor's ORT memory placement
- explicit CPU/CUDA destination allocation for switched inputs
- ORT `CopyTensor` for cross-resource transfers
- separate `boundary_copy` timing from segment execution timing
- output allocation on the selected segment resource
- an explicit per-segment resource plan for deterministic switching tests

Example for a five-segment level:

```text
--resource-plan cpu,cuda,cuda,cpu,cuda
```

The plan length must exactly match the selected level's segment count.
`--resource` remains available for a uniform resource selection, and
`auto` continues to use offline measured costs.

## Deliberate limitations

This milestone does not implement:

- adaptive resource selection
- runtime resource monitoring
- adaptive segmentation-level selection
- AD/MI scheduling
- degradation/recovery
- adaptive buffer reuse/zero-copy optimization
- contention experiments

## Test commands

Build:

```bash
cmake -S . -B build
cmake --build build -j2
```

Uniform CPU:

```bash
./apps/flexon_online_tester \
  --artifact artifacts/resnet18 \
  --level 4 \
  --resource cpu \
  --iterations 10
```

Uniform CUDA:

```bash
./apps/flexon_online_tester \
  --artifact artifacts/resnet18 \
  --level 4 \
  --resource cuda \
  --iterations 10
```

Alternating resources:

```bash
./apps/flexon_online_tester \
  --artifact artifacts/resnet18 \
  --level 4 \
  --resource-plan cpu,cuda,cpu,cuda,cpu \
  --iterations 10
```

The alternating run should report non-zero `boundary_copy` time at every
resource transition while same-resource segment boundaries should report zero
copy time.


## Explicit cross-resource transfer layer

Milestone 2 uses an explicit CPU/CUDA transfer layer rather than relying on
`IoBinding::SynchronizeInputs()` to move tensors between execution resources.
This is required for ORT 1.20.0 because the tested CUDA-to-CPU path does not
provide the registered transfer used by the previous implementation.

The transfer layer allocates the destination tensor from the destination
resource's ORT allocator and performs the boundary copy explicitly with the
CUDA runtime. The copy is measured separately as `boundary_copy`, while the
segment execution measurement remains the ORT execution time.

This also establishes the memory-ownership boundary used by Milestone 3: an
activation arena can own reusable CPU and CUDA buffers and IoBinding can bind
those existing tensors to segment inputs/outputs. That is the ONNX Runtime
analogue of the paper's pointer-based buffer sharing.

The paper's CPU/GPU non-shareable-buffer observation remains relevant: a buffer
can only be reused while its shape/type/device and lifetime are compatible.
Therefore the future arena will maintain separate CPU and CUDA pools rather
than pretending that one physical allocation can serve both resources.
