"""End-to-end analysis shared by the CLI, the exporter and the tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import funnel, metrics, riskmodel


@dataclass
class Analysis:
    appointments: pd.DataFrame
    counts: pd.DataFrame
    ledger: pd.DataFrame
    model: riskmodel.RiskModel
    funnel: funnel.FunnelResult


def load_appointments(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"clinic": str, "appointment_id": str})


def analyse(
    appointments: pd.DataFrame,
    config: metrics.MetricConfig | None = None,
    phi_override: float | None = None,
) -> Analysis:
    config = config or metrics.MetricConfig()
    counts = metrics.clinic_counts(appointments, config)
    ledger = metrics.exclusions_ledger(appointments, config)
    model = riskmodel.fit(appointments, config)
    expected = riskmodel.expected_by_clinic(appointments, model, config)
    result = funnel.build_funnel(counts, expected, phi_override=phi_override)
    return Analysis(appointments, counts, ledger, model, result)
