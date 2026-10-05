"""Subprocess runner and stdout parser for ``flexon_online``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import subprocess
from typing import Optional, Sequence

from .config import DEFAULT_ONLINE_BINARY, RunConfig


_FLOAT = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

_ARTIFACT_RE = re.compile(r'^\[online\] artifact loaded: (.+)$')
_MODEL_RE = re.compile(r'^\[online\] model: (.+)$')
_LEVELS_RE = re.compile(r'^\[online\] levels: (\d+)$')
_LEVEL_COUNT_RE = re.compile(r'^\[online\] level (\d+): (\d+) segments prepared$')
_ITERATION_RE = re.compile(r'^\[online\] iteration (\d+)/(\d+) level=(\d+)$')
_SCHEDULER_RE = re.compile(
    rf'^\[scheduler\] segment (\d+) (selected|alternative)=(cpu|cuda) '
    rf'score=({_FLOAT}) degradation=({_FLOAT}) remaining_capacity=({_FLOAT})$'
)
_EXECUTION_RE = re.compile(
    rf'^\s+segment (\d+) \[(cpu|cuda)\] ({_FLOAT}) ms '
    rf'\| boundary_copy=({_FLOAT}) ms '
    rf'\| arena_allocations=(\d+) '
    rf'\| arena_reuses=(\d+) '
    rf'\| arena_bytes=(\d+) '
    rf'\| arena_peak_live_bytes=(\d+)$'
)
_RECOVERY_PRIMARY_RE = re.compile(rf'^\[recovery\] segment (\d+) primary=(cpu|cuda) elapsed_ms=({_FLOAT}) alternative=(cpu|cuda) alternative_score=({_FLOAT}) gamma=({_FLOAT})$')
_RECOVERY_RESULT_RE = re.compile(rf'^\[recovery\] elapsed_ms=({_FLOAT})\s*\|?\s*(primary won|alternative won)$')
_ITERATION_MEASUREMENT_RE = re.compile(
    rf'^\[scheduler\] iteration measured_ms=({_FLOAT})$'
)
_LEVEL_DECISION_RE = re.compile(
    rf'^\[scheduler\] level decision measured_ms=({_FLOAT}) '
    rf'expected_ms=({_FLOAT}) level=(\d+) -> (\d+)$'
)
_SUCCESS_RE = re.compile(r'^\[online\] execution completed successfully$')
_ERROR_RE = re.compile(r'^\[online\] ERROR: (.*)$')


@dataclass
class SchedulerRecord:
    segment: int
    role: str
    resource: str
    score: float
    degradation: float
    remaining_capacity: float


@dataclass
class ExecutionRecord:
    segment: int
    resource: str
    elapsed_ms: float
    boundary_copy_ms: float
    arena_allocations: int
    arena_reuses: int
    arena_bytes: int
    arena_peak_live_bytes: int


@dataclass
class RecoveryRecord:
    segment: int
    primary_resource: str
    primary_elapsed_ms: float
    alternative_resource: str
    alternative_score: float
    gamma: float
    winner_elapsed_ms: Optional[float] = None
    winner: Optional[str] = None


@dataclass
class LevelDecision:
    measured_ms: float
    expected_ms: float
    level: int
    next_level: int


@dataclass
class IterationRecord:
    iteration: int
    total_iterations: int
    level: int
    scheduler: list[SchedulerRecord] = field(default_factory=list)
    executions: list[ExecutionRecord] = field(default_factory=list)
    recoveries: list[RecoveryRecord] = field(default_factory=list)
    # Authoritative end-to-end iteration latency emitted by the runtime.
    # Present for every completed iteration, independent of adaptive mode.
    measured_ms: Optional[float] = None
    level_decision: Optional[LevelDecision] = None


@dataclass
class ExperimentRun:
    artifact: Optional[str] = None
    model: Optional[str] = None
    level_count: Optional[int] = None
    segments_per_level: dict[int, int] = field(default_factory=dict)
    iterations: list[IterationRecord] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    returncode: Optional[int] = None

    @property
    def measured_execution_records(self) -> list[ExecutionRecord]:
        return [record for iteration in self.iterations for record in iteration.executions]

    @property
    def recovery_records(self) -> list[RecoveryRecord]:
        return [record for iteration in self.iterations for record in iteration.recoveries]


def parse_online_output(stdout: str) -> ExperimentRun:
    """Parse stdout emitted by the current ``flexon_online`` CLI.

    Parsing is deliberately isolated from subprocess execution so notebook
    experiments can test parsing against saved logs without rerunning inference.
    Unknown lines are ignored to keep the parser forward-compatible with extra
    diagnostic output.
    """
    result = ExperimentRun(stdout=stdout)
    current: Optional[IterationRecord] = None
    pending_recovery: Optional[RecoveryRecord] = None

    for raw_line in stdout.splitlines():
        line = raw_line.rstrip()

        match = _ARTIFACT_RE.match(line)
        if match:
            result.artifact = match.group(1).strip().strip('"')
            continue

        match = _MODEL_RE.match(line)
        if match:
            result.model = match.group(1).strip()
            continue

        match = _LEVELS_RE.match(line)
        if match:
            result.level_count = int(match.group(1))
            continue

        match = _LEVEL_COUNT_RE.match(line)
        if match:
            result.segments_per_level[int(match.group(1))] = int(match.group(2))
            continue

        match = _ITERATION_RE.match(line)
        if match:
            current = IterationRecord(
                iteration=int(match.group(1)),
                total_iterations=int(match.group(2)),
                level=int(match.group(3)),
            )
            result.iterations.append(current)
            pending_recovery = None
            continue

        match = _SCHEDULER_RE.match(line)
        if match and current is not None:
            current.scheduler.append(
                SchedulerRecord(
                    segment=int(match.group(1)),
                    role=match.group(2),
                    resource=match.group(3),
                    score=float(match.group(4)),
                    degradation=float(match.group(5)),
                    remaining_capacity=float(match.group(6)),
                )
            )
            continue

        match = _RECOVERY_PRIMARY_RE.match(line)
        if match and current is not None:
            pending_recovery = RecoveryRecord(
                segment=int(match.group(1)),
                primary_resource=match.group(2),
                primary_elapsed_ms=float(match.group(3)),
                alternative_resource=match.group(4),
                alternative_score=float(match.group(5)),
                gamma=float(match.group(6)),
            )
            current.recoveries.append(pending_recovery)
            continue

        match = _EXECUTION_RE.match(line)
        if match and current is not None:
            current.executions.append(
                ExecutionRecord(
                    segment=int(match.group(1)),
                    resource=match.group(2),
                    elapsed_ms=float(match.group(3)),
                    boundary_copy_ms=float(match.group(4)),
                    arena_allocations=int(match.group(5)),
                    arena_reuses=int(match.group(6)),
                    arena_bytes=int(match.group(7)),
                    arena_peak_live_bytes=int(match.group(8)),
                )
            )
            continue

        match = _RECOVERY_RESULT_RE.match(line)
        if match and current is not None and pending_recovery is not None:
            pending_recovery.winner_elapsed_ms = float(match.group(1))
            pending_recovery.winner = match.group(2).replace(" won", "")
            pending_recovery = None
            continue

        match = _ITERATION_MEASUREMENT_RE.match(line)
        if match and current is not None:
            current.measured_ms = float(match.group(1))
            continue

        match = _LEVEL_DECISION_RE.match(line)
        if match and current is not None:
            current.level_decision = LevelDecision(
                measured_ms=float(match.group(1)),
                expected_ms=float(match.group(2)),
                level=int(match.group(3)),
                next_level=int(match.group(4)),
            )
            continue

        if _SUCCESS_RE.match(line):
            result.success = True
            continue

        match = _ERROR_RE.match(line)
        if match:
            result.error = match.group(1)

    return result


def run_online(
    config: RunConfig,
    *,
    binary: Path = DEFAULT_ONLINE_BINARY,
    timeout: Optional[float] = None,
    cwd: Optional[Path] = None,
) -> ExperimentRun:
    """Run ``flexon_online`` once and parse its stdout."""
    # Priority isolation is implemented by flexon_online itself. The
    # scheduler thread invokes the narrowly-scoped privileged helper after
    # CUDA/ORT initialization, while inference remains normal priority.
    command = config.command(binary=binary)

    completed = subprocess.run(
        command,
        cwd=cwd or binary.parent.parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )

    result = parse_online_output(completed.stdout)
    result.stderr = completed.stderr
    result.returncode = completed.returncode

    if completed.returncode != 0 and result.error is None:
        result.error = f"flexon_online exited with status {completed.returncode}"

    return result
