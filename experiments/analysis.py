"""Experiment-level analysis helpers for FlexOn benchmark runs."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence
import pandas as pd
from .measurements import LatencySummary, RunMeasurements

@dataclass(frozen=True)
class ExperimentRecord:
    """One measured repetition belonging to an experimental condition."""
    experiment: str
    measurements: RunMeasurements
    repetition: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    def __post_init__(self) -> None:
        if not self.experiment.strip(): raise ValueError("experiment name cannot be empty")
        if self.repetition <= 0: raise ValueError("repetition must be greater than zero")

def latency_samples_dataframe(records: Iterable[ExperimentRecord]) -> pd.DataFrame:
    """Flatten iteration latency samples into one row per measured iteration."""
    rows=[]
    for record in records:
        base={"experiment":record.experiment,"repetition":record.repetition}
        base.update(record.metadata)
        for i, latency in enumerate(record.measurements.iteration_latencies_ms,1):
            row=dict(base); row["sample"]=i; row["latency_ms"]=latency; rows.append(row)
    columns=["experiment","repetition","sample","latency_ms"]
    if not rows: return pd.DataFrame(columns=columns)
    extra=[c for c in rows[0] if c not in columns]
    return pd.DataFrame(rows, columns=columns+extra)

def run_summary_dataframe(records: Iterable[ExperimentRecord]) -> pd.DataFrame:
    """Return one row per measured repetition with core latency statistics."""
    rows=[]
    for record in records:
        s=record.measurements.latency
        row={"experiment":record.experiment,"repetition":record.repetition,
             "count":s.count,"mean_ms":s.mean_ms,"median_ms":s.median_ms,
             "p90_ms":s.p90_ms,"p95_ms":s.p95_ms,"p99_ms":s.p99_ms,
             "min_ms":s.min_ms,"max_ms":s.max_ms,"std_ms":s.std_ms}
        row.update(record.metadata); rows.append(row)
    columns=["experiment","repetition","count","mean_ms","median_ms","p90_ms","p95_ms","p99_ms","min_ms","max_ms","std_ms"]
    if not rows: return pd.DataFrame(columns=columns)
    extra=[c for c in rows[0] if c not in columns]
    return pd.DataFrame(rows, columns=columns+extra)

def summarize_latency(records: Iterable[ExperimentRecord], *, by: Sequence[str]=( "experiment",)) -> pd.DataFrame:
    """Pool iteration samples and calculate latency statistics by condition."""
    samples=latency_samples_dataframe(records)
    columns=list(by)+_summary_columns()
    if samples.empty: return pd.DataFrame(columns=columns)
    missing=[c for c in by if c not in samples.columns]
    if missing: raise KeyError(f"grouping columns not found: {missing}")
    rows=[]
    for keys, group in samples.groupby(list(by), sort=False, dropna=False):
        if len(by) == 1:
            keys = (keys[0],) if isinstance(keys, tuple) else (keys,)
        s=LatencySummary.from_samples(group["latency_ms"].tolist())
        row=dict(zip(by,keys)); row.update(_summary_dict(s)); rows.append(row)
    return pd.DataFrame(rows, columns=columns)

def compare_latency(summary: pd.DataFrame, *, baseline: str, experiment_column: str="experiment") -> pd.DataFrame:
    """Compare each summary row against a named baseline using mean latency."""
    missing={experiment_column,"mean_ms"}-set(summary.columns)
    if missing: raise KeyError(f"summary is missing required columns: {sorted(missing)}")
    base=summary[summary[experiment_column]==baseline]
    if len(base)!=1: raise ValueError(f"baseline '{baseline}' must identify exactly one summary row; found {len(base)}")
    base_mean=float(base.iloc[0]["mean_ms"])
    if base_mean<=0: raise ValueError("baseline mean latency must be greater than zero")
    result=summary.copy()
    result["delta_ms"]=result["mean_ms"]-base_mean
    result["speedup"]=base_mean/result["mean_ms"]
    result["latency_reduction_pct"]=(base_mean-result["mean_ms"])/base_mean*100.0
    return result

def _summary_dict(s: LatencySummary)->dict[str,Any]:
    return {"count":s.count,"mean_ms":s.mean_ms,"median_ms":s.median_ms,"p90_ms":s.p90_ms,"p95_ms":s.p95_ms,"p99_ms":s.p99_ms,"min_ms":s.min_ms,"max_ms":s.max_ms,"std_ms":s.std_ms}

def _summary_columns()->list[str]:
    return ["count","mean_ms","median_ms","p90_ms","p95_ms","p99_ms","min_ms","max_ms","std_ms"]
