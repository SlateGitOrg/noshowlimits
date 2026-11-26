from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from noshowlimits import funnel, pipeline

# Twelve clinics built so every z-score is a whole number. E is a perfect square, so
# O = E + z * sqrt(E) is an integer.
HAND_E = np.array([4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169], dtype=float)
HAND_Z = np.array([3, -2, 0, 8, 1, -4, 2, 3, -1, 0, 2, 1], dtype=float)
HAND_O = HAND_E + HAND_Z * np.sqrt(HAND_E)

# Worked by hand. Sorted z: -4 -2 -1 0 0 1 1 2 2 3 3 8.
# 10th percentile: position 0.1 * 11 = 1.1 -> -2 + 0.1 * (-1 - -2) = -1.9.
# 90th percentile: position 0.9 * 11 = 9.9 -> 3 + 0.9 * (3 - 3) = 3.
# Winsorised: -1.9 -1.9 -1 0 0 1 1 2 2 3 3 3.
# Squares sum to 3.61 * 2 + 1 + 1 + 1 + 4 + 4 + 27 = 45.22.
HAND_PHI = 45.22 / 12
# tau2 = (12 * phi - 11) / (sum E - sum E^2 / sum E) with sum E = 818 and sum E^2 = 89270.
HAND_TAU2 = (45.22 - 11) / (818 - 89270 / 818)


def test_hand_fixture_observed_counts_are_integers():
    assert HAND_O.tolist() == [10, 3, 16, 65, 42, 21, 80, 108, 90, 121, 168, 182]


def test_z_scores():
    z = funnel.z_scores(HAND_O, HAND_E)
    np.testing.assert_allclose(z, HAND_Z, atol=1e-12)


def test_phi_matches_the_hand_computation_with_winsorisation():
    assert funnel.estimate_phi(HAND_O, HAND_E) == pytest.approx(HAND_PHI, abs=1e-9)


def test_winsorisation_is_what_keeps_phi_down():
    unwinsorised = float(np.mean(HAND_Z**2))
    assert unwinsorised > 2 * HAND_PHI  # the z = 8 and z = -4 clinics would dominate
    assert funnel.winsorise(HAND_Z).min() == pytest.approx(-1.9)
    assert funnel.winsorise(HAND_Z).max() == pytest.approx(3.0)


def test_tau2_matches_the_moment_formula_and_floors_at_zero():
    phi = funnel.estimate_phi(HAND_O, HAND_E)
    assert funnel.estimate_tau2(HAND_E, phi) == pytest.approx(HAND_TAU2, abs=1e-12)
    assert funnel.estimate_tau2(HAND_E, 1.0) == pytest.approx(1 / (818 - 89270 / 818), abs=1e-12)
    assert funnel.estimate_tau2(HAND_E, 0.5) == 0.0


def test_limit_formulas():
    e = np.array([100.0])
    mult = funnel.limits(e, "multiplicative", phi=4.0, tau2=0.01)
    add = funnel.limits(e, "additive", phi=4.0, tau2=0.01)
    assert mult["upper_95"][0] == pytest.approx(1 + 1.959963984540054 * np.sqrt(4 / 100))
    assert mult["lower_998"][0] == pytest.approx(1 - 3.090232306167813 * np.sqrt(4 / 100))
    assert add["upper_95"][0] == pytest.approx(1 + 1.959963984540054 * np.sqrt(1 / 100 + 0.01))
    assert add["lower_95"][0] == pytest.approx(1 - 1.959963984540054 * np.sqrt(1 / 100 + 0.01))
    with pytest.raises(ValueError, match="unknown method"):
        funnel.limits(e, "bootstrap", 1.0, 0.0)  # type: ignore[arg-type]


def test_lower_limit_is_floored_at_zero():
    lims = funnel.limits(np.array([10.0]), "multiplicative", phi=2.2, tau2=0.0)
    assert lims["lower_998"][0] == 0.0 and lims["lower_95"][0] > 0.0


def test_status_boundaries():
    lims = funnel.limits(np.full(5, 100.0), "multiplicative", phi=1.0, tau2=0.0)
    ratio = np.array(
        [
            1.0,
            lims["upper_95"][0] + 1e-9,
            lims["upper_998"][0] + 1e-9,
            lims["lower_95"][0] - 1e-9,
            lims["lower_998"][0] - 1e-9,
        ]
    )
    got = funnel.status(ratio, lims).tolist()
    assert got == ["within", "above_95", "above_99.8", "below_95", "below_99.8"]
    assert set(got) == set(funnel.STATUSES)
    at_limit = funnel.status(np.array([lims["upper_95"][0]]), {k: v[:1] for k, v in lims.items()})
    assert at_limit.tolist() == ["within"]


@pytest.mark.parametrize("method", funnel.METHODS)
def test_limits_narrow_monotonically_as_expected_grows(method):
    e = np.arange(5, 400, dtype=float)
    lims = funnel.limits(e, method, phi=2.2, tau2=0.009)
    assert (np.diff(lims["upper_95"]) < 0).all() and (np.diff(lims["upper_998"]) < 0).all()
    assert (np.diff(lims["lower_95"]) >= 0).all() and (np.diff(lims["lower_998"]) >= 0).all()
    assert (lims["upper_998"] > lims["upper_95"]).all()
    assert (lims["lower_998"] <= lims["lower_95"]).all()
    unclipped = lims["lower_95"] > 0
    assert (np.diff(lims["lower_95"][unclipped]) > 0).all()


def test_additive_limits_have_a_floor_width_set_by_tau2():
    big = funnel.limits(np.array([1e9]), "additive", phi=2.0, tau2=0.01)
    assert big["upper_95"][0] == pytest.approx(1 + 1.959963984540054 * 0.1, abs=1e-6)
    multiplicative = funnel.limits(np.array([1e9]), "multiplicative", phi=2.0, tau2=0.01)
    assert multiplicative["upper_95"][0] == pytest.approx(1.0, abs=1e-3)


def test_overdispersion_is_present_in_the_seeded_data(analysis):
    assert analysis.funnel.phi > 1.5
    assert analysis.funnel.tau2 > 0


@pytest.mark.parametrize("suffix", ["mult", "add"])
def test_planted_genuine_outliers_are_flagged(table, suffix):
    status = table[f"status_{suffix}"]
    assert status["C07"] == "above_99.8"
    assert status["C21"] == "above_99.8"
    assert status["C40"] == "below_99.8"
    assert table.loc["C07", "oe_ratio"] > 1.5 and table.loc["C40", "oe_ratio"] < 0.5


@pytest.mark.parametrize("suffix", ["mult", "add"])
def test_chance_high_and_case_mix_high_are_not_flagged(table, suffix):
    top5 = set(table.sort_values("raw_rate", ascending=False).head(5)["clinic"])
    assert {"C33", "C12"} <= top5
    assert table.loc["C33", f"status_{suffix}"] == "within"
    assert table.loc["C12", f"status_{suffix}"] == "within"
    for clinic in ("C33", "C12"):
        assert table.loc[clinic, f"upper_95_{suffix}"] > table.loc[clinic, "oe_ratio"]


def test_case_mix_explains_c12_and_not_c07(table):
    overall = table["overall_rate"].iloc[0]
    assert table.loc["C12", "raw_rate"] > 1.4 * overall  # high raw rate...
    assert (
        table.loc["C12", "expected"] / table.loc["C12", "eligible"] > 1.4 * overall
    )  # ...expected
    assert abs(table.loc["C12", "oe_ratio"] - 1) < 0.1
    assert table.loc["C07", "oe_ratio"] > 1.5


def test_raw_ranking_puts_planted_genuine_clinics_in_the_top_five_with_the_decoys(table):
    top5 = set(table.sort_values("raw_rank").head(5)["clinic"])
    assert {"C07", "C21", "C12", "C33"} <= top5
    assert table.loc["C40", "raw_rank"] >= 38


@pytest.mark.parametrize("method", funnel.METHODS)
def test_forcing_phi_to_one_flags_strictly_more_clinics(appointments, analysis, method):
    forced = pipeline.analyse(appointments, phi_override=1.0).funnel
    assert forced.phi == 1.0
    assert funnel.outside_95(forced.table, method) > funnel.outside_95(
        analysis.funnel.table, method
    )


def test_forced_phi_one_flags_a_clinic_the_estimated_phi_clears(appointments, analysis):
    forced = pipeline.analyse(appointments, phi_override=1.0).funnel.table.set_index("clinic")
    estimated = analysis.funnel.table.set_index("clinic")
    newly = forced.index[
        (forced["status_mult"] != "within") & (estimated["status_mult"] == "within")
    ]
    assert len(newly) >= 5


def test_adjusted_rate_is_ratio_times_overall_rate(analysis):
    table = analysis.funnel.table
    overall = table["observed"].sum() / table["eligible"].sum()
    assert analysis.funnel.overall_rate == pytest.approx(overall)
    np.testing.assert_allclose(table["adjusted_rate"], table["oe_ratio"] * overall, rtol=1e-12)
    assert table["adjusted_rate"].sum() / len(table) == pytest.approx(
        np.mean(table["oe_ratio"]) * overall
    )


def test_build_funnel_rejects_mismatched_clinics():
    counts = pd.DataFrame(
        {"clinic": ["A", "B"], "eligible": [10, 10], "noshows": [1, 2], "raw_rate": [0.1, 0.2]}
    )
    expected = pd.DataFrame({"clinic": ["A", "A"], "expected": [1.0, 2.0]})
    with pytest.raises(Exception, match="one-to-one|not a one-to-one|unique"):
        funnel.build_funnel(counts, expected)
