"""Configuration for reproducible FlexOn experiments.

Runtime policy lives in ``config/scheduler.yaml``.  This module contains only
experiment-harness configuration: paths, repetition counts, warmup policy,
and command-line settings used to invoke the frozen runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONLINE_BINARY = PROJECT_ROOT / "apps" / "flexon_online"
DEFAULT_SCHEDULER_CONFIG = PROJECT_ROOT / "config" / "scheduler.yaml"


@dataclass(frozen=True)
class RunConfig:
    """One invocation of ``flexon_online``."""

    artifact: Path
    iterations: int = 100
    level: Optional[int] = None
    resource: str = "auto"
    resource_plan: Optional[Tuple[str, ...]] = None
    adaptive_level: bool = False
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None
    recovery: bool = False
    priority_isolation: bool = False
    scheduler_config: Path = DEFAULT_SCHEDULER_CONFIG

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be greater than zero")
        if self.level is not None and self.level < 0:
            raise ValueError("level must be non-negative")
        if self.resource not in {"cpu", "cuda", "auto"}:
            raise ValueError(f"invalid resource: {self.resource}")
        if self.resource_plan is not None:
            if not self.resource_plan:
                raise ValueError("resource_plan cannot be empty")
            if any(resource not in {"cpu", "cuda"} for resource in self.resource_plan):
                raise ValueError("resource_plan entries must be cpu or cuda")
            if self.resource != "auto":
                raise ValueError("resource_plan should be used with resource='auto'")
        if self.adaptive_level and self.resource_plan is not None:
            raise ValueError("adaptive_level cannot be combined with resource_plan")

    def command(self, binary: Path = DEFAULT_ONLINE_BINARY) -> list[str]:
        """Build the exact subprocess command for this run."""
        command = [
            str(binary),
            "--artifact",
            str(self.artifact),
            "--iterations",
            str(self.iterations),
            "--resource",
            self.resource,
            "--scheduler-config",
            str(self.scheduler_config),
        ]

        if self.level is not None:
            command += ["--level", str(self.level)]
        if self.resource_plan is not None:
            command += ["--resource-plan", ",".join(self.resource_plan)]
        if self.adaptive_level:
            command.append("--adaptive-level")
        if self.alpha is not None:
            command += ["--alpha", str(self.alpha)]
        if self.beta is not None:
            command += ["--beta", str(self.beta)]
        if self.gamma is not None:
            command += ["--gamma", str(self.gamma)]
        if self.recovery:
            command.append("--with-recovery")
        if self.priority_isolation:
            command.append("--priority-isolation")

        return command


@dataclass(frozen=True)
class ExperimentConfig:
    """Defaults shared by benchmark notebooks/scripts."""

    online_binary: Path = DEFAULT_ONLINE_BINARY
    warmup_iterations: int = 10
    measured_iterations: int = 100
    repetitions: int = 5
    timeout_seconds: Optional[float] = None
    artifacts: Tuple[Path, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations cannot be negative")
        if self.measured_iterations <= 0:
            raise ValueError("measured_iterations must be greater than zero")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be greater than zero")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
