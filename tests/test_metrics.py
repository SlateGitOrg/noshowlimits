from __future__ import annotations

import pytest

from noshowlimits import metrics
from noshowlimits.metrics import MetricConfig


def _by_clinic(counts):
    return counts.set_index("clinic")


def test_hand_fixture_with_late_cancellations_excluded(hand_fixture):
    counts = _by_clinic(metrics.clinic_counts(hand_fixture, MetricConfig(late_cancel="exclude")))
    # A: attended 2, no-show 1 -> 1 / 3.  B: attended 1, no-show 1 -> 1 / 2.
    assert counts.loc["A", ["noshows", "eligible", "appointments"]].tolist() == [1, 3, 6]
    assert counts.loc["B", ["noshows", "eligible", "appointments"]].tolist() == [1, 2, 4]
    assert counts["noshows"].sum() == 2 and counts["eligible"].sum() == 5
    assert counts.loc["A", "raw_rate"] == pytest.approx(1 / 3)


def test_hand_fixture_with_late_cancellations_counted_as_noshows(hand_fixture):
    counts = _by_clinic(metrics.clinic_counts(hand_fixture, MetricConfig(late_cancel="noshow")))
    # A: no-show 1 + late 1 = 2 over attended 2 + 2 = 4.  B: 2 over 3.
    assert counts.loc["A", ["noshows", "eligible"]].tolist() == [2, 4]
    assert counts.loc["B", ["noshows", "eligible"]].tolist() == [2, 3]
    assert counts["noshows"].sum() == 4 and counts["eligible"].sum() == 7


def test_default_config_excludes_late_cancellations(hand_fixture):
    default = metrics.clinic_counts(hand_fixture)
    explicit = metrics.clinic_counts(hand_fixture, MetricConfig(late_cancel="exclude"))
    assert default.equals(explicit)


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        (
            "exclude",
            {
                ("A", "hospital_cancel"): 1,
                ("A", "early_patient_cancel"): 1,
                ("A", "late_patient_cancel"): 1,
                ("B", "hospital_cancel"): 1,
                ("B", "late_patient_cancel"): 1,
            },
        ),
        (
            "noshow",
            {
                ("A", "hospital_cancel"): 1,
                ("A", "early_patient_cancel"): 1,
                ("B", "hospital_cancel"): 1,
            },
        ),
    ],
)
def test_exclusions_ledger_accounts_for_every_excluded_appointment(hand_fixture, setting, expected):
    config = MetricConfig(late_cancel=setting)
    ledger = metrics.exclusions_ledger(hand_fixture, config)
    got = {(r.clinic, r.outcome): r.appointments for r in ledger.itertuples()}
    assert got == expected
    counts = metrics.clinic_counts(hand_fixture, config)
    assert ledger["appointments"].sum() == counts["excluded"].sum()
    assert (ledger["reason"].str.len() > 0).all()


def test_ledger_and_counts_agree_on_seeded_data(appointments):
    counts = metrics.clinic_counts(appointments)
    ledger = metrics.exclusions_ledger(appointments)
    assert (counts["eligible"] + counts["excluded"] == counts["appointments"]).all()
    assert ledger["appointments"].sum() == counts["excluded"].sum()
    per_clinic = ledger.groupby("clinic")["appointments"].sum()
    assert per_clinic.reindex(counts["clinic"]).to_numpy().tolist() == counts["excluded"].tolist()
    assert ledger["outcome"].isin(metrics.EXCLUSION_REASONS).all()


def test_unknown_outcome_is_rejected(hand_fixture):
    broken = hand_fixture.copy()
    broken.loc[0, "outcome"] = "left_without_being_seen"
    with pytest.raises(ValueError, match="left_without_being_seen"):
        metrics.classify(broken)


def test_invalid_late_cancel_setting_is_rejected():
    with pytest.raises(ValueError, match="late_cancel"):
        MetricConfig(late_cancel="maybe")
