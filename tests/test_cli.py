from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from noshowlimits.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("cli-data")
    result = runner.invoke(app, ["generate", "--n-appointments", "30000", "--out", str(out)])
    assert result.exit_code == 0, result.output
    return out


def test_generate_writes_the_appointment_table(data_dir):
    lines = (data_dir / "appointments.csv").read_text().splitlines()
    assert len(lines) == 30_001
    assert lines[0].startswith("appointment_id,clinic,specialty,date")


def test_fit_prints_coefficients_auc_and_deciles(data_dir):
    result = runner.invoke(app, ["fit", "--data", str(data_dir / "appointments.csv")])
    assert result.exit_code == 0, result.output
    assert "lead_band[>6w]" in result.output and "AUC" in result.output
    assert "appt_type[follow_up]" in result.output
    assert "Calibration by decile" in result.output


@pytest.mark.parametrize("method", ["multiplicative", "additive"])
def test_funnel_lists_the_planted_outliers(data_dir, method):
    args = ["funnel", "--method", method, "--data", str(data_dir / "appointments.csv")]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert f"{method} limits" in result.output
    for clinic in ("C07", "C21", "C40"):
        assert clinic in result.output
    assert "above_99.8" in result.output and "below_99.8" in result.output


def test_funnel_all_lists_every_clinic_and_phi_can_be_forced(data_dir):
    path = str(data_dir / "appointments.csv")
    result = runner.invoke(app, ["funnel", "--all", "--phi", "1.0", "--data", path])
    assert result.exit_code == 0, result.output
    assert "phi = 1.000" in result.output and "42 of 42 clinics listed" in result.output


def test_league_raw_ranks_by_raw_rate(data_dir):
    path = str(data_dir / "appointments.csv")
    raw = runner.invoke(app, ["league", "--raw", "--top", "3", "--data", path])
    assert raw.exit_code == 0, raw.output
    assert "Raw-rate league table" in raw.output
    adjusted = runner.invoke(app, ["league", "--top", "3", "--data", path])
    assert "Risk-adjusted league table" in adjusted.output
    assert adjusted.output != raw.output


def test_export_then_validate_twb(data_dir, tmp_path):
    path = str(data_dir / "appointments.csv")
    out, twb = tmp_path / "tableau" / "data", tmp_path / "tableau" / "NoShowLimits.twb"
    result = runner.invoke(app, ["export", "--data", path, "--out", str(out), "--twb", str(twb)])
    assert result.exit_code == 0, result.output
    for name in ("clinic_funnel.csv", "funnel_limits.csv", "funnel_plot.csv"):
        assert (out / name).exists()
    ok = runner.invoke(app, ["validate-twb", str(twb)])
    assert ok.exit_code == 0, ok.output
    assert "consistent" in ok.output

    header_fix = (out / "clinic_funnel.csv").read_text().replace("oe_ratio", "ratio", 1)
    (out / "clinic_funnel.csv").write_text(header_fix)
    bad = runner.invoke(app, ["validate-twb", str(twb)])
    assert bad.exit_code == 1
    assert "oe_ratio" in bad.output


def test_missing_data_file_is_a_clean_error(tmp_path):
    result = runner.invoke(app, ["fit", "--data", str(tmp_path / "none.csv")])
    assert result.exit_code == 1
    assert "generate" in result.output


def test_late_cancel_switch_changes_the_counts(data_dir):
    path = str(data_dir / "appointments.csv")
    base = runner.invoke(app, ["league", "--raw", "--data", path])
    counted = runner.invoke(app, ["league", "--raw", "--late-cancel", "noshow", "--data", path])
    assert counted.exit_code == 0, counted.output
    assert base.output != counted.output


def test_league_include_adds_a_clinic_outside_the_top_rows(data_dir):
    path = str(data_dir / "appointments.csv")
    base = runner.invoke(app, ["league", "--raw", "--top", "3", "--data", path])
    more = runner.invoke(app, ["league", "--raw", "--top", "3", "--include", "C40", "--data", path])
    assert "C40" not in base.output and "C40" in more.output
    assert "below" in more.output
    bad = runner.invoke(app, ["league", "--include", "C99", "--data", path])
    assert bad.exit_code == 1 and "C99" in bad.output
