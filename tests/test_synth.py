from __future__ import annotations

import pandas as pd
import pytest

from noshowlimits import synth


def test_shape_and_columns(appointments):
    assert list(appointments.columns) == synth.COLUMNS
    assert len(appointments) == 60_000
    assert appointments["clinic"].nunique() == 42
    assert appointments["specialty"].nunique() == 6
    assert appointments["appointment_id"].is_unique
    assert set(appointments["outcome"]) == set(synth.OUTCOMES)


def test_dates_span_eighteen_months(appointments):
    dates = pd.to_datetime(appointments["date"])
    assert dates.min() >= pd.Timestamp("2025-01-01")
    assert dates.max() <= pd.Timestamp("2026-06-30")
    assert (dates.max() - dates.min()).days > 520


def test_same_seed_gives_identical_data_and_other_seed_differs():
    config = synth.SynthConfig(seed=5, n_appointments=25_000)
    first = synth.generate(config)
    second = synth.generate(config)
    other = synth.generate(synth.SynthConfig(seed=6, n_appointments=25_000))
    pd.testing.assert_frame_equal(first, second)
    assert not first["outcome"].equals(other["outcome"])


def test_planted_clinic_sizes(appointments):
    sizes = appointments.groupby("clinic").size()
    assert sizes[["C07", "C21", "C40"]].min() >= 2500
    assert sizes["C12"] == 1500
    assert sizes["C33"] == 120
    assert sizes.drop(["C07", "C21", "C40", "C12", "C33"]).min() >= 400


def test_cancellation_rates_match_the_specification(appointments):
    shares = appointments["outcome"].value_counts(normalize=True)
    assert shares["hospital_cancel"] == pytest.approx(0.04, abs=0.004)
    assert shares["early_patient_cancel"] == pytest.approx(0.06, abs=0.004)
    assert shares["late_patient_cancel"] == pytest.approx(0.03, abs=0.004)


def test_c33_realised_noshow_rate_is_fixed_near_fifteen_percent(appointments):
    c33 = appointments[appointments["clinic"] == "C33"]
    eligible = c33[c33["outcome"].isin(["attended", "no_show"])]
    assert (eligible["outcome"] == "no_show").sum() == round(0.15 * len(eligible))


def test_c12_has_the_planted_case_mix(appointments):
    c12 = appointments[appointments["clinic"] == "C12"]
    assert (c12["lead_band"] == ">6w").mean() == pytest.approx(0.70, abs=0.03)
    assert (c12["age_band"] == "18-29").mean() == pytest.approx(0.50, abs=0.03)


def test_lead_days_agree_with_lead_band(appointments):
    for band, (lo, hi) in synth.LEAD_DAY_RANGES.items():
        days = appointments.loc[appointments["lead_band"] == band, "lead_days"]
        assert days.min() >= lo and days.max() <= hi


def test_too_few_appointments_for_the_planted_clinics_is_rejected():
    with pytest.raises(ValueError, match="too small"):
        synth.generate(synth.SynthConfig(n_appointments=20_000))
