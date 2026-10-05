"""Measurement and statistical summaries for FlexOn experiment runs.

The module is independent of subprocess execution and plotting. It treats the
runtime's ``[scheduler] iteration measured_ms=...`` record as the authoritative
user-facing end-to-end latency. Runtime-path diagnostics are reported
separately so they never contaminate the primary latency metric.
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
    adaptive_decision_count: int
    level_transitions: int
    level_usage: tuple[tuple[int, int], ...]

    @property
    def total_iteration_latency_ms(self) -> float:
        return sum(self.iteration_latencies_ms)

    @property
    def total_segment_executions(self) -> int:
        return self.cpu_execution_count + self.cuda_execution_count

    @property
    def average_segments_per_iteration(self) -> float:
        if not self.iteration_latencies_ms:
            return 0.0
        return self.total_segment_executions / len(self.iteration_latencies_ms)

    @property
    def cpu_execution_pct(self) -> float:
        total = self.total_segment_executions
        return 100.0 * self.cpu_execution_count / total if total else 0.0

    @property
    def cuda_execution_pct(self) -> float:
        total = self.total_segment_executions
        return 100.0 * self.cuda_execution_count / total if total else 0.0

    @property
    def recovery_trigger_rate_pct(self) -> float:
        if not self.iteration_latencies_ms:
            return 0.0
        return 100.0 * self.recovery_count / len(self.iteration_latencies_ms)

    @property
    def levels_used(self) -> tuple[int, ...]:
        return tuple(level for level, _ in self.level_usage)


def iteration_latencies(run: ExperimentRun, *, warmup_iterations: int = 0) -> list[float]:
    """Return authoritative iteration latency samples, excluding warmups."""
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
        raise ValueError("run does not contain enough measured iterations after warmup removal")
    return latencies[warmup_iterations:]


def execution_samples(run: ExperimentRun, *, warmup_iterations: int = 0) -> list[ExecutionRecord]:
    """Return segment execution records after positional warmup removal."""
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations cannot be negative")
    return [
        record
        for iteration in run.iterations[warmup_iterations:]
        for record in iteration.executions
    ]


def recovery_samples(run: ExperimentRun, *, warmup_iterations: int = 0) -> list[RecoveryRecord]:
    """Return recovery records after positional warmup removal."""
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations cannot be negative")
    return [
        record
        for iteration in run.iterations[warmup_iterations:]
        for record in iteration.recoveries
    ]


def measure_run(run: ExperimentRun, *, warmup_iterations: int = 0) -> RunMeasurements:
    """Compute latency and runtime-path diagnostics for one parsed run."""
    latencies = iteration_latencies(run, warmup_iterations=warmup_iterations)
    measured_iterations = run.iterations[warmup_iterations:]
    executions = execution_samples(run, warmup_iterations=warmup_iterations)
    recoveries = recovery_samples(run, warmup_iterations=warmup_iterations)

    execution_values = [record.elapsed_ms for record in executions]
    boundary_values = [record.boundary_copy_ms for record in executions]
    winners = [record.winner for record in recoveries]

    level_transitions = sum(
        iteration.level_decision is not None
        and iteration.level_decision.level != iteration.level_decision.next_level
        for iteration in measured_iterations
    )
    adaptive_decision_count = sum(
        iteration.level_decision is not None for iteration in measured_iterations
    )

    level_counts: dict[int, int] = {}
    for iteration in measured_iterations:
        level_counts[iteration.level] = level_counts.get(iteration.level, 0) + 1

    return RunMeasurements(
        iteration_latencies_ms=tuple(latencies),
        latency=LatencySummary.from_samples(latencies),
        execution_latencies_ms=tuple(execution_values),
        execution_latency=LatencySummary.from_samples(execution_values) if execution_values else None,
        boundary_copy_ms=tuple(boundary_values),
        boundary_copy=LatencySummary.from_samples(boundary_values) if boundary_values else None,
        cpu_execution_count=sum(record.resource == "cpu" for record in executions),
        cuda_execution_count=sum(record.resource == "cuda" for record in executions),
        recovery_count=len(recoveries),
        recovery_primary_wins=sum(winner == "primary" for winner in winners),
        recovery_alternative_wins=sum(winner == "alternative" for winner in winners),
        recovery_unresolved=sum(winner is None for winner in winners),
        adaptive_decision_count=adaptive_decision_count,
        level_transitions=level_transitions,
        level_usage=tuple(sorted(level_counts.items())),
    )


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
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
    return float(sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower]))


def _sample_stddev(values: Iterable[float]) -> float:
    samples = list(values)
    if len(samples) < 2:
        return 0.0
    average = mean(samples)
    return sqrt(sum((value - average) ** 2 for value in samples) / (len(samples) - 1))
