# FlexOn Architecture

## Phase separation

```text
                         +----------------------+
                         |       config/        |
                         +----------+-----------+
                                    |
                 +------------------+------------------+
                 |                                     |
                 v                                     v
        +------------------+                  +------------------+
        |   OFFLINE PHASE  |                  |   ONLINE PHASE   |
        +------------------+                  +------------------+
        | graph analysis   |                  | artifact loading |
        | operator/profile |                  | resource monitor |
        | initial segments |                  | level selector   |
        | multi-level      |                  | resource select  |
        | artifact build   |                  | segment executor |
        | validation       |                  | recovery         |
        +--------+---------+                  | telemetry        |
                 |                            +--------+---------+
                 v                                     |
        +------------------+                            |
        | offline artifacts|----------------------------+
        +------------------+
```

## Backend boundary

The project does **not** treat ONNX Runtime as the implementation of FlexOn
segmentation.

ONNX Runtime is the execution backend. The offline compiler owns graph slicing
and segment artifact generation. Each segment/resource combination can then be
represented by a prebuilt ORT session.

The runtime must never construct an ORT session on the critical inference path.

## Core data

`core/` contains only data-oriented types shared by offline and online code.
Scheduling algorithms, file I/O, ORT calls and graph transformations do not
belong in `core/`.

## Timing model

Every measured period should expose at least:

```text
T_total =
    scheduler_overhead
  + input/boundary preparation
  + ORT execution
  + output/boundary transfer
  + recovery overhead (if any)
```

Profiling must distinguish:

1. operator kernel time,
2. segment execution time,
3. tensor/device transfer time,
4. ORT/session overhead,
5. scheduler overhead.

This distinction is necessary because a sum of operator kernel times is not
necessarily the same quantity as end-to-end execution of a generated segment.
