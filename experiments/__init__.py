"""FlexOn experiment harness.

The package has two intentionally separate responsibilities:

* :mod:`experiments.contention` controls external CPU/GPU contention.
* The remaining modules run, parse, measure, analyze, and plot FlexOn
  experiments without knowing whether contention is active.
"""

from .config import ExperimentConfig, RunConfig
from .runner import ExperimentRun, run_online, parse_online_output
from .measurements import LatencySummary, RunMeasurements, measure_run
from .analysis import (
    ExperimentRecord,
    latency_samples_dataframe,
    run_summary_dataframe,
    summarize_latency,
    compare_latency,
    runtime_diagnostics_dataframe,
)
from .plotting import (
    plot_latency_comparison,
    plot_tail_latency,
    plot_latency_distribution,
    plot_latency_percentiles,
    plot_resource_execution_breakdown,
    plot_level_changes,
)

__all__ = [
    "ExperimentConfig",
    "RunConfig",
    "ExperimentRun",
    "run_online",
    "parse_online_output",
    "LatencySummary",
    "RunMeasurements",
    "measure_run",
    "ExperimentRecord",
    "latency_samples_dataframe",
    "run_summary_dataframe",
    "summarize_latency",
    "compare_latency",
    "runtime_diagnostics_dataframe",
    "plot_latency_comparison",
    "plot_tail_latency",
    "plot_latency_distribution",
    "plot_latency_percentiles",
    "plot_resource_execution_breakdown",
    "plot_level_changes"
]
