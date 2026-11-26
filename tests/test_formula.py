"""Run the workbook's own Tableau formulas over the exported rows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from noshowlimits.tableau import formula as fm
from noshowlimits.tableau import registry as reg

COMMITTED = Path(__file__).resolve().parents[1] / "tableau" / "data"


def _evaluate(source: reg.DataSource, caption: str, row: dict, method: str):
    formulas = {c.caption: c.formula for c in source.calcs}

    def resolve(name: str):
        if name in formulas:
            return fm.evaluate(formulas[name], resolve, {reg.PARAMETER_NAME: method})
        return row[name]

    return resolve(caption)


def test_arithmetic_comparison_and_iif():
    value = fm.evaluate(
        "IIF([a] / [b] > 1.5, MAX(0, 1 - 2), SQRT(16))", lambda n: {"a": 6, "b": 3}[n]
    )
    assert value == 0
    assert fm.evaluate('IIF([m] = "x", 1, 2)', lambda n: "x") == 1
    assert fm.evaluate('IIF([Parameters].[P] <> "x", 1, 2)', lambda n: 0, {"P": "x"}) == 2
    assert fm.evaluate("1 + 2 * 3 - 4 / 2", lambda n: 0) == 5


def test_references_split_fields_from_parameters():
    text = 'IIF([Parameters].[Parameter 1] = "additive", [a] + [b c], [a])'
    assert fm.references(text) == ["a", "b c", "a"]
    assert fm.parameter_references(text) == ["Parameter 1"]


def test_unsupported_syntax_is_rejected():
    with pytest.raises(fm.FormulaError, match="unsupported word"):
        fm.evaluate("LOOKUP([a], 1)", lambda n: 1)
    with pytest.raises(fm.FormulaError, match="cannot parse"):
        fm.evaluate("[a] ? 2", lambda n: 1)


def test_registry_formulas_use_full_precision_normal_quantiles():
    text = " ".join(c.formula for c in reg.CLINIC_FUNNEL.calcs)
    assert "1.959963984540054" in text and "3.090232306167813" in text


@pytest.mark.parametrize("method", reg.PARAMETER_VALUES)
def test_clinic_funnel_calculated_fields_reproduce_the_exported_columns(method):
    frame = pd.read_csv(COMMITTED / "clinic_funnel.csv")
    suffix = "mult" if method == "multiplicative" else "add"
    checked = 0
    for row in frame.to_dict("records"):
        for calc in reg.CLINIC_FUNNEL.calcs:
            column = calc.reference.get(method) or calc.reference["*"]
            got = _evaluate(reg.CLINIC_FUNNEL, calc.caption, row, method)
            if isinstance(got, str):
                assert got == row[column], (row["clinic"], calc.caption)
            else:
                assert got == pytest.approx(row[column], abs=1e-9), (row["clinic"], calc.caption)
            checked += 1
        assert row[f"status_{suffix}"] in {
            "above_99.8",
            "above_95",
            "within",
            "below_95",
            "below_99.8",
        }
    assert checked == 42 * len(reg.CLINIC_FUNNEL.calcs)


@pytest.mark.parametrize("method", reg.PARAMETER_VALUES)
def test_funnel_plot_calculated_fields_split_series_and_select_the_method(method):
    frame = pd.read_csv(COMMITTED / "funnel_plot.csv", keep_default_na=False)
    for row in frame.iloc[::7].to_dict("records"):
        match = _evaluate(reg.FUNNEL_PLOT, "Method Match", row, method)
        assert match is (row["method"] == method)
        limit = _evaluate(reg.FUNNEL_PLOT, "Limit Value", row, method)
        point = _evaluate(reg.FUNNEL_PLOT, "Clinic O/E", row, method)
        if row["series"] == "clinic":
            assert limit is None and point == row["value"]
        else:
            assert point is None and limit == row["value"]


def test_null_literal_evaluates_to_none():
    assert fm.evaluate("IIF([a] > 1, [a], NULL)", lambda n: 0) is None
