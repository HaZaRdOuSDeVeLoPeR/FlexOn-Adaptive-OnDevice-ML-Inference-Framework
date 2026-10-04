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
)
from .contention import ContentionSession, ContentionSpec, run_with_contention
from .plotting import (
    plot_latency_comparison,
    plot_tail_latency,
    plot_latency_distribution,
    plot_latency_percentiles,
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
    "ContentionSession",
    "ContentionSpec",
    "run_with_contention",
    "plot_latency_comparison",
    "plot_tail_latency",
    "plot_latency_distribution",
    "plot_latency_percentiles",
]
