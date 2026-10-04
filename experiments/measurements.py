"""Measurement and statistical summaries for FlexOn experiment runs.

This module deliberately contains no subprocess logic and no plotting.  It
operates on the structured records produced by :mod:`experiments.runner`.

The runtime reports an iteration-level ``measured_ms`` value on its
scheduler iteration-measurement line. That value is the authoritative
iteration latency used here. Adaptive runs additionally emit a level-decision
record containing the same measured latency plus control information; that
record is kept separate and is not required for latency measurement. Segment
execution timings are also kept separate because summing them would not
necessarily reproduce the runtime's end-to-end measurement (scheduler
overhead, synchronization, and other work may exist between segment timings).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean, median
from typing import Iterable, Sequence

from .runner import ExecutionRecord, ExperimentRun, RecoveryRecord




@dataclass(frozen=True)
class LatencySummary:
    """Descriptive statistics for a sequence of latency samples in ms."""

    count: int
    mean_ms: float
    median_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> "LatencySummary":
        values = [float(value) for value in samples]
        if not values:
            raise ValueError("at least one latency sample is required")
        if any(value < 0 for value in values):
            raise ValueError("latency samples cannot be negative")

        ordered = sorted(values)
        return cls(
            count=len(values),
            mean_ms=mean(values),
            median_ms=median(values),
            p90_ms=_percentile(ordered, 90.0),
            p95_ms=_percentile(ordered, 95.0),
            p99_ms=_percentile(ordered, 99.0),
            min_ms=ordered[0],
            max_ms=ordered[-1],
            std_ms=_sample_stddev(values),
        )


@dataclass(frozen=True)
class RunMeasurements:
    """Measured quantities extracted from one ``flexon_online`` run."""

    iteration_latencies_ms: tuple[float, ...]
    latency: LatencySummary
    execution_latencies_ms: tuple[float, ...]
    execution_latency: LatencySummary | None
    boundary_copy_ms: tuple[float, ...]
    boundary_copy: LatencySummary | None
    cpu_execution_count: int
    cuda_execution_count: int
    recovery_count: int
    recovery_primary_wins: int
    recovery_alternative_wins: int
    recovery_unresolved: int
    level_transitions: int

    @property
    def total_iteration_latency_ms(self) -> float:
        """Return the sum of measured iteration latencies."""
        return sum(self.iteration_latencies_ms)


def iteration_latencies(
    run: ExperimentRun,
    *,
    warmup_iterations: int = 0,
) -> list[float]:
    """Return authoritative iteration latency samples, excluding warmups.

    Warmup removal is positional across the run, matching the CLI's iteration
    order. The runtime's unconditional ``iteration measured_ms`` record is the
    canonical latency source, so fixed-resource and non-adaptive runs are
    measured exactly like adaptive runs. A missing measurement is treated as
    malformed benchmark output rather than silently dropping the iteration.
    """
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations cannot be negative")

    latencies: list[float] = []
    for iteration in run.iterations:
        if iteration.measured_ms is None:
            raise ValueError(
                "run is missing iteration measured_ms for iteration "
                f"{iteration.iteration}"
            )
        latencies.append(iteration.measured_ms)

    if len(latencies) <= warmup_iterations:
        raise ValueError(
            "run does not contain enough measured iterations after warmup removal"
        )

    return latencies[warmup_iterations:]


def execution_samples(
    run: ExperimentRun,
    *,
    warmup_iterations: int = 0,
) -> list[ExecutionRecord]:
    """Return segment execution records after positional warmup removal."""
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations cannot be negative")
    return [
        record
        for iteration in run.iterations[warmup_iterations:]
        for record in iteration.executions
    ]


def recovery_samples(
    run: ExperimentRun,
    *,
    warmup_iterations: int = 0,
) -> list[RecoveryRecord]:
    """Return recovery records after positional warmup removal."""
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations cannot be negative")
    return [
        record
        for iteration in run.iterations[warmup_iterations:]
        for record in iteration.recoveries
    ]


def measure_run(
    run: ExperimentRun,
    *,
    warmup_iterations: int = 0,
) -> RunMeasurements:
    """Compute the core benchmark metrics for one parsed run."""
    latencies = iteration_latencies(run, warmup_iterations=warmup_iterations)
    executions = execution_samples(run, warmup_iterations=warmup_iterations)
    recoveries = recovery_samples(run, warmup_iterations=warmup_iterations)

    execution_values = [record.elapsed_ms for record in executions]
    boundary_values = [record.boundary_copy_ms for record in executions]

    winners = [record.winner for record in recoveries]
    level_transitions = sum(
        iteration.level_decision is not None
        and iteration.level_decision.level != iteration.level_decision.next_level
        for iteration in run.iterations[warmup_iterations:]
    )

    return RunMeasurements(
        iteration_latencies_ms=tuple(latencies),
        latency=LatencySummary.from_samples(latencies),
        execution_latencies_ms=tuple(execution_values),
        execution_latency=(
            LatencySummary.from_samples(execution_values) if execution_values else None
        ),
        boundary_copy_ms=tuple(boundary_values),
        boundary_copy=(
            LatencySummary.from_samples(boundary_values) if boundary_values else None
        ),
        cpu_execution_count=sum(record.resource == "cpu" for record in executions),
        cuda_execution_count=sum(record.resource == "cuda" for record in executions),
        recovery_count=len(recoveries),
        recovery_primary_wins=sum(winner == "primary" for winner in winners),
        recovery_alternative_wins=sum(winner == "alternative" for winner in winners),
        recovery_unresolved=sum(winner is None for winner in winners),
        level_transitions=level_transitions,
    )


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    """Compute a linearly interpolated percentile from sorted samples."""
    if not sorted_values:
        raise ValueError("at least one value is required")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")

    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return float(
        sorted_values[lower]
        + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def _sample_stddev(values: Iterable[float]) -> float:
    samples = list(values)
    if len(samples) < 2:
        return 0.0
    average = mean(samples)
    return sqrt(sum((value - average) ** 2 for value in samples) / (len(samples) - 1))
