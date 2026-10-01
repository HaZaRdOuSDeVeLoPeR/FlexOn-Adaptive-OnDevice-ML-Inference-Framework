# ONNX Runtime Backend Mapping

| FlexOn/LiteRT concept | ONNX Runtime mechanism | Implementation note |
|---|---|---|
| CPU/GPU resource | Execution Provider | CPUExecutionProvider and CUDAExecutionProvider |
| Accelerator fallback operators | EP graph partitioning | CUDA-capable nodes can be assigned to CUDA; unsupported nodes fall back to CPU when CPU EP is registered |
| Executable segment | Independent ONNX segment model + pre-created ORT session | Exact FlexOn scheduling unit is project-owned |
| Preloaded kernels | Pre-created `Ort::Session` objects | Sessions must be created before inference |
| Avoid repeated model construction | Session pool/cache | No session creation in `run()` |
| Shared CPU allocator | `Ort::Env::CreateAndRegisterAllocator` | Can reduce per-session CPU arena duplication |
| Shared prepacked weights | `OrtPrepackedWeightsContainer` | Useful for identical initializers across sessions |
| Zero/minimized device copies | `Ort::IoBinding` + device `OrtValue`s | Boundary transport must be measured explicitly |
| Runtime trace | ORT profiling | Useful for backend diagnosis, not a replacement for FlexOn timing |
| Custom operator | `Ort::CustomOpDomain` / custom kernel | Not required for basic segment scheduling |
| Invoke a specific operator range | No direct public equivalent | Generate a segment graph/model instead of attempting to invoke an arbitrary node range in a monolithic session |

## Critical limitation

ONNX Runtime's EP partitioning is not equivalent to FlexOn's multi-level
segmentation. EP partitioning answers which provider can execute nodes/subgraphs.
FlexOn decides where the scheduling boundaries are and changes those boundaries
at runtime by selecting a precomputed segmentation level.

## Critical timing rule

Do not derive `C_r(segment)` solely by summing per-node ORT kernel events.

For FlexOn, the authoritative offline cost of a segment should be measured by
executing the same generated segment artifact that the online executor will
run, using the same session options and provider configuration. Operator-level
profiles remain useful for segmentation/split analysis and diagnostics.


## Offline resource capability profiling

Before initial segmentation, FlexOn builds a one-operator probe model for each source operator and checks every enabled resource. CUDA probes disable ORT CPU fallback so that an operator which cannot be assigned to CUDA is classified explicitly instead of silently executing on CPU. The resulting status is stored as `supported`, `unsupported`, or `profiling_failed`; unsupported resources receive infinite cost and do not disappear from the artifact.

This capability pass is separate from segment timing. Segment execution measurements remain the authoritative costs used by the segmentation algorithm.
