from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from noshowlimits import riskmodel
from noshowlimits.metrics import MetricConfig


def test_design_matrix_has_no_clinic_information(appointments):
    x = riskmodel.design_matrix(appointments.head(50))
    assert not any("clinic" in c or "specialty" in c for c in x.columns)
    assert x.columns[0] == "const"
    assert "lead_band[<=2w]" not in x.columns  # reference level
    assert "lead_band[>6w]" in x.columns and "age_band[18-29]" in x.columns
    assert set(np.unique(x.to_numpy())) == {0.0, 1.0}


def test_auc_by_rank_formula_matches_pairwise_counting():
    rng = np.random.default_rng(3)
    y = rng.random(60) < 0.4
    score = np.round(rng.random(60), 1)  # coarse scores force ties
    wins = sum(
        1.0 if s_pos > s_neg else 0.5 if s_pos == s_neg else 0.0
        for s_pos, s_neg in itertools.product(score[y], score[~y])
    )
    assert riskmodel.auc_rank(y, score) == pytest.approx(wins / (y.sum() * (~y).sum()), abs=1e-12)


def test_auc_extremes():
    y = np.array([0, 0, 0, 1, 1])
    assert riskmodel.auc_rank(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9])) == 1.0
    assert riskmodel.auc_rank(y, np.array([0.9, 0.8, 0.7, 0.2, 0.1])) == 0.0
    assert riskmodel.auc_rank(y, np.full(5, 0.5)) == 0.5
    with pytest.raises(ValueError, match="both outcomes"):
        riskmodel.auc_rank(np.zeros(4), np.arange(4))


def test_lead_time_and_age_coefficients_recover_the_truth(analysis):
    coef = analysis.model.coefficients["coef"]
    assert coef["lead_band[>6w]"] == pytest.approx(0.7, abs=0.15)
    assert coef["age_band[18-29]"] == pytest.approx(0.6, abs=0.15)
    assert coef["deprivation_quintile[1]"] == pytest.approx(0.5, abs=0.15)
    assert coef["appt_type[follow_up]"] == pytest.approx(-0.2, abs=0.15)
    # Levels with no true effect stay near zero.
    assert abs(coef["lead_band[2-6w]"]) < 0.15
    assert abs(coef["age_band[65+]"]) < 0.15


def test_expected_total_equals_observed_total(analysis):
    table = analysis.funnel.table
    assert table["expected"].sum() == pytest.approx(table["observed"].sum(), rel=1e-6)


def test_every_calibration_decile_is_within_two_points(analysis):
    cal = analysis.model.calibration
    assert len(cal) == 10
    assert cal["n"].max() - cal["n"].min() <= 1
    assert (cal["abs_diff"] <= 0.02).all()
    assert cal["observed"].is_monotonic_increasing


def test_discrimination_is_modest_as_case_mix_alone_predicts_little(analysis):
    assert 0.58 < analysis.model.auc < 0.70


def test_predict_returns_probabilities_for_new_rows(analysis, appointments):
    probs = analysis.model.predict(appointments.head(200))
    assert probs.shape == (200,)
    assert ((probs > 0) & (probs < 1)).all()


def test_counting_late_cancels_as_noshows_changes_the_fit(appointments):
    base = riskmodel.fit(appointments)
    counted = riskmodel.fit(appointments, MetricConfig(late_cancel="noshow"))
    assert counted.n_obs == base.n_obs + int(
        (appointments["outcome"] == "late_patient_cancel").sum()
    )
    # Late cancels are unrelated to case mix, so the lead-time effect is diluted.
    assert (
        counted.coefficients.loc["lead_band[>6w]", "coef"]
        < base.coefficients.loc["lead_band[>6w]", "coef"]
    )


def test_expected_by_clinic_sums_predicted_probabilities():
    rows = pd.DataFrame(
        {
            "clinic": ["X", "X", "Y"],
            "specialty": "Cardiology",
            "outcome": ["attended", "no_show", "attended"],
            "lead_band": ["<=2w", ">6w", "<=2w"],
            "age_band": ["30-49", "18-29", "30-49"],
            "appt_type": ["new", "new", "follow_up"],
            "deprivation_quintile": [5, 1, 5],
        }
    )
    model = riskmodel.RiskModel(
        coefficients=pd.DataFrame(),
        auc=0.5,
        calibration=pd.DataFrame(),
        params=pd.Series(0.0, index=riskmodel.design_matrix(rows).columns),
        columns=list(riskmodel.design_matrix(rows).columns),
        n_obs=3,
    )
    expected = riskmodel.expected_by_clinic(rows, model).set_index("clinic")["expected"]
    assert expected.to_dict() == {"X": 1.0, "Y": 0.5}  # every probability is 0.5
