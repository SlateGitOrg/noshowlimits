"""The no-show definition, per-clinic counts and the exclusions ledger.

Every appointment ends in exactly one outcome:

* ``attended``: in the denominator, not a no-show.
* ``no_show``: in the denominator and the numerator.
* ``hospital_cancel``: excluded (the clinic, not the patient, removed the slot).
* ``early_patient_cancel``: excluded (the patient gave 24 hours or more of notice).
* ``late_patient_cancel``: notice under 24 hours. By config either a no-show (denominator and
  numerator) or excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

LateCancel = Literal["exclude", "noshow"]

EXCLUSION_REASONS = {
    "hospital_cancel": "hospital cancellation",
    "early_patient_cancel": "patient cancelled 24 h or more ahead",
    "late_patient_cancel": "patient cancelled under 24 h ahead (excluded by config)",
}
KNOWN_OUTCOMES = (
    "attended",
    "no_show",
    "late_patient_cancel",
    "early_patient_cancel",
    "hospital_cancel",
)


@dataclass(frozen=True)
class MetricConfig:
    """``late_cancel`` is ``"exclude"`` (default) or ``"noshow"``."""

    late_cancel: LateCancel = "exclude"

    def __post_init__(self) -> None:
        if self.late_cancel not in ("exclude", "noshow"):
            raise ValueError(f"late_cancel must be 'exclude' or 'noshow', got {self.late_cancel!r}")


def classify(appointments: pd.DataFrame, config: MetricConfig | None = None) -> pd.DataFrame:
    """Add boolean ``in_denominator`` and ``is_noshow`` columns and an ``exclusion`` reason."""
    config = config or MetricConfig()
    unknown = sorted(set(appointments["outcome"]) - set(KNOWN_OUTCOMES))
    if unknown:
        raise ValueError(f"unknown outcome values: {unknown}")
    outcome = appointments["outcome"]
    late_counts = config.late_cancel == "noshow"
    is_noshow = (outcome == "no_show") | ((outcome == "late_patient_cancel") & late_counts)
    in_denominator = (outcome == "attended") | is_noshow
    exclusion = outcome.where(~in_denominator)
    out = appointments.copy()
    out["is_noshow"] = is_noshow
    out["in_denominator"] = in_denominator
    out["exclusion"] = exclusion
    return out


def clinic_counts(appointments: pd.DataFrame, config: MetricConfig | None = None) -> pd.DataFrame:
    """Per clinic: booked appointments, eligible (denominator), no-shows, raw rate, exclusions."""
    classified = classify(appointments, config)
    grouped = classified.groupby("clinic", sort=True)
    counts = pd.DataFrame(
        {
            "appointments": grouped.size(),
            "eligible": grouped["in_denominator"].sum(),
            "noshows": grouped["is_noshow"].sum(),
        }
    )
    for outcome in EXCLUSION_REASONS:
        counts[f"excl_{outcome}"] = grouped["exclusion"].apply(lambda s, o=outcome: (s == o).sum())
    counts["excluded"] = counts["appointments"] - counts["eligible"]
    counts["raw_rate"] = counts["noshows"] / counts["eligible"]
    counts["specialty"] = grouped["specialty"].first() if "specialty" in classified else ""
    return counts.reset_index()


def exclusions_ledger(
    appointments: pd.DataFrame, config: MetricConfig | None = None
) -> pd.DataFrame:
    """One row per (clinic, reason) with the number of appointments left out of the rate."""
    classified = classify(appointments, config)
    excluded = classified[~classified["in_denominator"]]
    ledger = excluded.groupby(["clinic", "exclusion"]).size().rename("appointments").reset_index()
    ledger = ledger.rename(columns={"exclusion": "outcome"})
    ledger["reason"] = ledger["outcome"].map(EXCLUSION_REASONS)
    return ledger[["clinic", "outcome", "reason", "appointments"]]
