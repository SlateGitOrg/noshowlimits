"""CSV exports that feed the Tableau workbook.

``clinic_funnel.csv`` has one row per clinic. ``funnel_limits.csv`` is a grid of expected counts
from 5 to the largest clinic's E, with both limit pairs. The grid contains every clinic's exact E
as well as the whole numbers, so a clinic's limits can be looked up in it without interpolation.

``funnel_plot.csv`` is the long-format table behind the workbook's funnel plot: the grid's limits
and the clinic points stacked as (series, method, expected, value) rows, so one datasource draws
both the limit lines and the points and no data blending is needed.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import funnel
from .funnel import METHOD_SUFFIX, METHODS

GRID_START = 5
LIMIT_KEYS = ("lower_95", "upper_95", "lower_998", "upper_998")

CLINIC_FUNNEL_COLUMNS = [
    "clinic",
    "specialty",
    "appointments",
    "eligible",
    "excluded",
    "excl_hospital_cancel",
    "excl_early_patient_cancel",
    "excl_late_patient_cancel",
    "observed",
    "expected",
    "raw_rate",
    "raw_rank",
    "oe_ratio",
    "adjusted_rate",
    "overall_rate",
    "phi",
    "tau2",
    *[f"{key}_{METHOD_SUFFIX[m]}" for m in METHODS for key in LIMIT_KEYS],
    "status_mult",
    "status_add",
]
FUNNEL_LIMITS_COLUMNS = [
    "expected",
    "phi",
    "tau2",
    *[f"{key}_{METHOD_SUFFIX[m]}" for m in METHODS for key in LIMIT_KEYS],
]


FUNNEL_PLOT_COLUMNS = ["series", "method", "expected", "value", "clinic", "status"]
PLOT_SERIES = ("upper_998", "upper_95", "lower_95", "lower_998")


def clinic_funnel_frame(result: funnel.FunnelResult) -> pd.DataFrame:
    return result.table[CLINIC_FUNNEL_COLUMNS].copy()


def funnel_limits_frame(result: funnel.FunnelResult) -> pd.DataFrame:
    """Limit grid: whole-number E from 5 up to the largest clinic E, plus each clinic's own E."""
    e = result.table["expected"].to_numpy(dtype=float)
    grid = np.unique(
        np.concatenate([np.arange(GRID_START, math.floor(e.max()) + 1, dtype=float), e])
    )
    grid = grid[grid >= GRID_START]
    frame = pd.DataFrame({"expected": grid, "phi": result.phi, "tau2": result.tau2})
    for method in METHODS:
        lims = funnel.limits(grid, method, result.phi, result.tau2)
        for key in LIMIT_KEYS:
            frame[f"{key}_{METHOD_SUFFIX[method]}"] = lims[key]
    return frame[FUNNEL_LIMITS_COLUMNS]


def funnel_plot_frame(result: funnel.FunnelResult) -> pd.DataFrame:
    """Long table: one row per (limit series, method, grid E) and one per (clinic, method).

    Limit rows have ``series`` in ``PLOT_SERIES`` and an empty clinic and status. Clinic rows have
    ``series == "clinic"``, ``value`` = O/E and that clinic's status under the row's method."""
    grid = funnel_limits_frame(result)
    parts = []
    for method in METHODS:
        suffix = METHOD_SUFFIX[method]
        for series in PLOT_SERIES:
            parts.append(
                pd.DataFrame(
                    {
                        "series": series,
                        "method": method,
                        "expected": grid["expected"],
                        "value": grid[f"{series}_{suffix}"],
                        "clinic": "",
                        "status": "",
                    }
                )
            )
        table = result.table
        parts.append(
            pd.DataFrame(
                {
                    "series": "clinic",
                    "method": method,
                    "expected": table["expected"],
                    "value": table["oe_ratio"],
                    "clinic": table["clinic"],
                    "status": table[f"status_{suffix}"],
                }
            )
        )
    return pd.concat(parts, ignore_index=True)[FUNNEL_PLOT_COLUMNS]


def write_exports(result: funnel.FunnelResult, out_dir: str | Path) -> tuple[Path, Path, Path]:
    """Write the three CSVs into ``out_dir`` and return their paths. Floats keep full precision."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clinic_path = out / "clinic_funnel.csv"
    limits_path = out / "funnel_limits.csv"
    clinic_funnel_frame(result).to_csv(clinic_path, index=False)
    plot_path = out / "funnel_plot.csv"
    funnel_limits_frame(result).to_csv(limits_path, index=False)
    funnel_plot_frame(result).to_csv(plot_path, index=False)
    return clinic_path, limits_path, plot_path
