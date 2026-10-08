"""Research-oriented plots for FlexOn experiment results."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

_REQUIRED_SUMMARY = {"experiment", "mean_ms", "median_ms", "p95_ms", "p99_ms"}


def plot_latency_comparison(
    summary: pd.DataFrame,
    *,
    metric: str = "mean_ms",
    experiment_column: str = "experiment",
    title: str | None = None,
):
    """Plot one latency metric for each experimental condition."""
    _require_columns(summary, {experiment_column, metric})
    if summary.empty:
        raise ValueError("cannot plot an empty summary")
    frame = summary[[experiment_column, metric]].copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="raise")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(frame[experiment_column].astype(str), frame[metric])
    axis.set_xlabel(experiment_column)
    axis.set_ylabel(_latency_label(metric))
    axis.set_title(title or f"FlexOn {metric} comparison")
    axis.tick_params(axis="x", rotation=30)

    figure.tight_layout()
    plt.close(figure)
    
    return figure


def plot_tail_latency(
    summary: pd.DataFrame,
    *,
    experiment_column: str = "experiment",
    title: str = "FlexOn latency: median and tail",
):
    """Plot median, P95 and P99 latency for each condition."""
    _require_columns(summary, _REQUIRED_SUMMARY)
    if summary.empty:
        raise ValueError("cannot plot an empty summary")
    frame = summary[[experiment_column, "median_ms", "p95_ms", "p99_ms"]].copy()
    figure, axis = plt.subplots(figsize=(9, 5))
    x = range(len(frame))
    width = 0.25
    axis.bar([i - width for i in x], frame["median_ms"], width=width, label="Median")
    axis.bar(list(x), frame["p95_ms"], width=width, label="P95")
    axis.bar([i + width for i in x], frame["p99_ms"], width=width, label="P99")
    axis.set_xticks(list(x), frame[experiment_column].astype(str))
    axis.set_xlabel(experiment_column)
    axis.set_ylabel("Latency (ms)")
    axis.set_title(title)
    axis.legend()
    axis.tick_params(axis="x", rotation=30)

    figure.tight_layout()
    plt.close(figure)

    return figure


def plot_latency_distribution(
    samples: pd.DataFrame,
    *,
    value_column: str = "latency_ms",
    group_column: str = "experiment",
    exclude_groups: list[str] = [],
    title: str = "FlexOn latency distribution",
):
    """Plot measured latency samples as a boxplot.

    ``exclude_groups`` is useful for a focused FlexOn-only view when a much
    slower baseline would compress the remaining distributions visually.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    _require_columns(samples, {group_column, value_column})

    if samples.empty:
        raise ValueError("cannot plot an empty sample dataframe")

    excluded = set(exclude_groups)
    frame = samples[~samples[group_column].isin(excluded)].copy()

    if frame.empty:
        raise ValueError("no samples remain after excluding groups")

    groups = list(
        frame.groupby(
            group_column,
            sort=False,
            dropna=False,
        )
    )

    figure, axis = plt.subplots(figsize=(9, 5))

    data = [
        group[value_column].astype(float).tolist()
        for _, group in groups
    ]
    labels = [
        str(name)
        for name, _ in groups
    ]

    axis.boxplot(
        data,
        tick_labels=labels,
        showfliers=True,
    )

    axis.set_xlabel(group_column)
    axis.set_ylabel("Latency (ms)")
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=30)

    figure.tight_layout()
    plt.close(figure)

    return figure


def plot_latency_percentiles(
    summary: pd.DataFrame,
    *,
    experiment_column: str = "experiment",
    percentiles: Sequence[str] = ("p50", "p90", "p95", "p99"),
    exclude_experiments: list[str] = [],
    title: str = "FlexOn latency percentiles",
):
    """Plot selected percentile statistics across conditions.

    ``p50`` maps to median. Excluding the CPU baseline is useful for a focused
    view because its much larger latency can compress low-latency FlexOn cases.
    """
    column_map = {"p50": "median_ms", "p90": "p90_ms", "p95": "p95_ms", "p99": "p99_ms"}
    unknown = [p for p in percentiles if p not in column_map]
    if unknown:
        raise ValueError(f"unsupported percentiles: {unknown}")
    required = {experiment_column, *(column_map[p] for p in percentiles)}
    _require_columns(summary, required)
    if summary.empty:
        raise ValueError("cannot plot an empty summary")

    excluded = set(exclude_experiments)
    frame = summary[~summary[experiment_column].isin(excluded)].copy()
    if frame.empty:
        raise ValueError("no summary rows remain after excluding experiments")

    figure, axis = plt.subplots(figsize=(9, 5))
    labels = frame[experiment_column].astype(str).tolist()
    x = list(range(len(frame)))
    for percentile in percentiles:
        axis.plot(x, frame[column_map[percentile]], marker="o", label=percentile.upper())
    axis.set_xticks(x, labels)
    axis.set_xlabel(experiment_column)
    axis.set_ylabel("Latency (ms)")
    axis.set_title(title)
    axis.legend()
    axis.tick_params(axis="x", rotation=30)

    figure.tight_layout()
    plt.close(figure)

    return figure


def plot_resource_execution_breakdown(
    diagnostics: pd.DataFrame,
    *,
    experiment_column: str = "experiment",
    title: str = "Segment executions by resource",
):
    """Plot CPU/CUDA segment execution counts from runtime diagnostics."""
    _require_columns(diagnostics, {experiment_column, "cpu_segment_executions", "cuda_segment_executions"})
    if diagnostics.empty:
        raise ValueError("cannot plot an empty diagnostics dataframe")

    # Diagnostics are repetition-level. Aggregate by experiment so the figure
    # describes the typical run while avoiding a noisy 30-bar plot.
    frame = (
        diagnostics.groupby(experiment_column, sort=False)[
            ["cpu_segment_executions", "cuda_segment_executions"]
        ]
        .mean()
        .reset_index()
    )

    figure, axis = plt.subplots(figsize=(9, 5))
    x = range(len(frame))
    axis.bar(list(x), frame["cpu_segment_executions"], label="CPU segments")
    axis.bar(
        list(x),
        frame["cuda_segment_executions"],
        bottom=frame["cpu_segment_executions"],
        label="CUDA segments",
    )
    axis.set_xticks(list(x), frame[experiment_column].astype(str))
    axis.set_xlabel(experiment_column)
    axis.set_ylabel("Segment executions / measured run")
    axis.set_title(title)
    axis.legend()
    axis.tick_params(axis="x", rotation=30)

    figure.tight_layout()
    plt.close(figure)
    
    return figure


def plot_level_changes(
    diagnostics: pd.DataFrame,
    *,
    experiment_column: str = "experiment",
    title: str = "Adaptive level changes",
):
    """Plot mean number of level changes per repetition."""
    _require_columns(diagnostics, {experiment_column, "level_changes"})
    if diagnostics.empty:
        raise ValueError("cannot plot an empty diagnostics dataframe")
    frame = diagnostics.groupby(experiment_column, sort=False)["level_changes"].mean().reset_index()
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(frame[experiment_column].astype(str), frame["level_changes"])
    axis.set_xlabel(experiment_column)
    axis.set_ylabel("Level changes / measured run")
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=30)
    
    figure.tight_layout()
    plt.close(figure)

    return figure


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"dataframe is missing required columns: {sorted(missing)}")


def _latency_label(metric: str) -> str:
    labels = {
        "mean_ms": "Mean latency (ms)",
        "median_ms": "Median latency (ms)",
        "p90_ms": "P90 latency (ms)",
        "p95_ms": "P95 latency (ms)",
        "p99_ms": "P99 latency (ms)",
    }
    return labels.get(metric, f"{metric} (ms)")
