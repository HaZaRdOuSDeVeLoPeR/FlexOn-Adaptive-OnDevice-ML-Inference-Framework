"""Standardized research plots for FlexOn experiment results.

The plotting layer intentionally contains no experiment execution or statistical
calculation. It consumes DataFrames produced by :mod:`experiments.analysis`
and returns Matplotlib ``Figure`` objects so callers can display or save them.
"""

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
    """Plot one latency metric for each experimental condition.

    Parameters
    ----------
    summary:
        DataFrame from ``run_summary_dataframe`` or ``summarize_latency``.
    metric:
        Numeric latency column, e.g. ``mean_ms`` or ``p95_ms``.
    experiment_column:
        Column containing condition names.
    title:
        Optional figure title.
    """
    _require_columns(summary, {experiment_column, metric})
    if summary.empty:
        raise ValueError("cannot plot an empty summary")

    frame = summary[[experiment_column, metric]].copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="raise")

    figure, axis = plt.subplots()
    axis.bar(frame[experiment_column].astype(str), frame[metric])
    axis.set_xlabel(experiment_column)
    axis.set_ylabel(_latency_label(metric))
    axis.set_title(title or f"FlexOn {metric} comparison")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    return figure


def plot_tail_latency(
    summary: pd.DataFrame,
    *,
    experiment_column: str = "experiment",
    title: str = "FlexOn latency distribution summary",
):
    """Plot median, P95 and P99 latency for each condition."""
    _require_columns(summary, _REQUIRED_SUMMARY)
    if summary.empty:
        raise ValueError("cannot plot an empty summary")

    frame = summary[[experiment_column, "median_ms", "p95_ms", "p99_ms"]].copy()
    figure, axis = plt.subplots()
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
    return figure


def plot_latency_distribution(
    samples: pd.DataFrame,
    *,
    value_column: str = "latency_ms",
    group_column: str = "experiment",
    title: str = "FlexOn latency distribution",
):
    """Plot measured latency samples grouped by experimental condition."""
    _require_columns(samples, {group_column, value_column})
    if samples.empty:
        raise ValueError("cannot plot an empty sample dataframe")

    groups = list(samples.groupby(group_column, sort=False, dropna=False))
    figure, axis = plt.subplots()
    data = [group[value_column].astype(float).tolist() for _, group in groups]
    labels = [str(name) for name, _ in groups]
    axis.boxplot(data, tick_labels=labels, showfliers=True)
    axis.set_xlabel(group_column)
    axis.set_ylabel("Latency (ms)")
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    return figure


def plot_latency_percentiles(
    summary: pd.DataFrame,
    *,
    experiment_column: str = "experiment",
    percentiles: Sequence[str] = ("p50", "p90", "p95", "p99"),
    title: str = "FlexOn latency percentiles",
):
    """Plot selected percentile columns as lines across conditions.

    ``p50`` is mapped to the existing ``median_ms`` column. Other values map
    directly to ``pXX_ms`` columns.
    """
    column_map = {"p50": "median_ms", "p90": "p90_ms", "p95": "p95_ms", "p99": "p99_ms"}
    unknown = [p for p in percentiles if p not in column_map]
    if unknown:
        raise ValueError(f"unsupported percentiles: {unknown}")

    required = {experiment_column, *(column_map[p] for p in percentiles)}
    _require_columns(summary, required)
    if summary.empty:
        raise ValueError("cannot plot an empty summary")

    figure, axis = plt.subplots()
    labels = summary[experiment_column].astype(str).tolist()
    x = list(range(len(summary)))
    for percentile in percentiles:
        axis.plot(x, summary[column_map[percentile]], marker="o", label=percentile.upper())
    axis.set_xticks(x, labels)
    axis.set_xlabel(experiment_column)
    axis.set_ylabel("Latency (ms)")
    axis.set_title(title)
    axis.legend()
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
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
