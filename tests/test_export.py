from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from noshowlimits import export, funnel

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "tableau" / "data"


@pytest.fixture(scope="module")
def written(analysis, tmp_path_factory):
    out = tmp_path_factory.mktemp("exports")
    export.write_exports(analysis.funnel, out)
    return out


def test_clinic_funnel_csv_has_one_row_per_clinic_and_registered_columns(written):
    frame = pd.read_csv(written / "clinic_funnel.csv")
    assert list(frame.columns) == export.CLINIC_FUNNEL_COLUMNS
    assert len(frame) == 42 and frame["clinic"].is_unique
    assert frame["status_mult"].isin(funnel.STATUSES).all()
    assert (frame["eligible"] + frame["excluded"] == frame["appointments"]).all()
    assert frame["excluded"].to_list() == (
        frame[["excl_hospital_cancel", "excl_early_patient_cancel", "excl_late_patient_cancel"]]
        .sum(axis=1)
        .to_list()
    )


def test_funnel_limits_grid_runs_from_five_to_the_largest_expected(written, analysis):
    grid = pd.read_csv(written / "funnel_limits.csv")
    assert list(grid.columns) == export.FUNNEL_LIMITS_COLUMNS
    e = grid["expected"].to_numpy()
    assert e[0] == 5.0
    assert e[-1] == analysis.funnel.table["expected"].max()
    assert (np.diff(e) > 0).all()
    whole = set(range(5, math.floor(e[-1]) + 1))
    assert whole <= set(e[e == np.floor(e)].astype(int).tolist())


@pytest.mark.parametrize("suffix", ["mult", "add"])
def test_grid_evaluated_at_each_clinics_expected_reproduces_its_limits(written, analysis, suffix):
    grid = pd.read_csv(written / "funnel_limits.csv").set_index("expected")
    clinics = pd.read_csv(written / "clinic_funnel.csv")
    for row in clinics.itertuples():
        for key in export.LIMIT_KEYS:
            column = f"{key}_{suffix}"
            assert grid.loc[row.expected, column] == pytest.approx(getattr(row, column), abs=1e-9)


@pytest.mark.parametrize("method", funnel.METHODS)
def test_grid_rows_match_the_closed_form(written, method):
    grid = pd.read_csv(written / "funnel_limits.csv")
    suffix = funnel.METHOD_SUFFIX[method]
    lims = funnel.limits(grid["expected"].to_numpy(), method, grid["phi"][0], grid["tau2"][0])
    for key in export.LIMIT_KEYS:
        np.testing.assert_allclose(grid[f"{key}_{suffix}"], lims[key], atol=1e-12)
        column = grid[f"{key}_{suffix}"]
        assert column.is_monotonic_increasing == key.startswith("lower")
        assert column.is_monotonic_decreasing == key.startswith("upper")


def test_funnel_plot_long_table_carries_limit_lines_and_clinic_points(written, analysis):
    plot = pd.read_csv(written / "funnel_plot.csv", keep_default_na=False)
    grid = pd.read_csv(written / "funnel_limits.csv")
    clinics = pd.read_csv(written / "clinic_funnel.csv")
    assert list(plot.columns) == export.FUNNEL_PLOT_COLUMNS
    assert set(plot["series"]) == {*export.PLOT_SERIES, "clinic"}
    assert set(plot["method"]) == set(funnel.METHODS)
    assert len(plot) == 2 * (4 * len(grid) + len(clinics))
    for method, suffix in funnel.METHOD_SUFFIX.items():
        part = plot[plot["method"] == method]
        for series in export.PLOT_SERIES:
            lines = part[part["series"] == series]
            np.testing.assert_array_equal(lines["expected"], grid["expected"])
            np.testing.assert_array_equal(lines["value"], grid[f"{series}_{suffix}"])
            assert (lines["clinic"] == "").all() and (lines["status"] == "").all()
        points = part[part["series"] == "clinic"].reset_index(drop=True)
        assert points["clinic"].tolist() == clinics["clinic"].tolist()
        np.testing.assert_array_equal(points["value"], clinics["oe_ratio"])
        np.testing.assert_array_equal(points["expected"], clinics["expected"])
        assert points["status"].tolist() == clinics[f"status_{suffix}"].tolist()


def test_committed_exports_match_a_fresh_run(written):
    for name in ("clinic_funnel.csv", "funnel_limits.csv", "funnel_plot.csv"):
        fresh = pd.read_csv(written / name).fillna("")
        committed = pd.read_csv(COMMITTED / name).fillna("")
        assert list(fresh.columns) == list(committed.columns)
        assert len(fresh) == len(committed)
        for column in fresh.columns:
            if not pd.api.types.is_numeric_dtype(fresh[column]):
                assert fresh[column].tolist() == committed[column].tolist(), column
            else:
                np.testing.assert_allclose(fresh[column], committed[column], rtol=1e-6, atol=1e-9)
