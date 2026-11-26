"""Seeded synthetic outpatient appointments with planted funnel-plot cases.

The generator draws 42 clinics over 18 months. Each appointment has a case mix (lead time, age
band, deprivation quintile, new or follow-up) and an outcome. The true no-show logit is

    -2.4 + 0.7*[lead time > 6 weeks] + 0.6*[age 18-29] + 0.5*[most deprived quintile]
         - 0.2*[follow-up] + clinic effect

with clinic effects drawn from N(0, 0.10^2) except for the planted clinics in ``PLANTED_EFFECTS``.
Cancellations (hospital, early patient, late patient) are drawn before the no-show decision and are
independent of case mix and clinic.

Planted cases that an analysis has to separate:

* C07 and C21: genuine process problem (+0.8 clinic effect), large volume.
* C40: genuinely good (-1.0 clinic effect), large volume.
* C12: no clinic effect, but 70% long-lead referrals and 50% aged 18-29, so a high raw rate.
* C33: about 120 appointments and no clinic effect; its realised no-shows are fixed at about 15%
  of eligible appointments, a chance high in a small clinic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SPECIALTIES = (
    "Cardiology",
    "Dermatology",
    "Gastroenterology",
    "Neurology",
    "Ophthalmology",
    "Orthopaedics",
)
# Share of appointments that are first attendances, by specialty (same order as SPECIALTIES).
NEW_SHARE = (0.35, 0.45, 0.40, 0.30, 0.25, 0.40)

LEAD_BANDS = ("<=2w", "2-6w", ">6w")
LEAD_DAY_RANGES = {"<=2w": (1, 14), "2-6w": (15, 42), ">6w": (43, 120)}
AGE_BANDS = ("0-17", "18-29", "30-49", "50-64", "65+")
BASE_AGE_PROBS = (0.08, 0.15, 0.27, 0.25, 0.25)
APPT_TYPES = ("new", "follow_up")

OUTCOMES = ("attended", "no_show", "late_patient_cancel", "early_patient_cancel", "hospital_cancel")
HOSPITAL_CANCEL_RATE = 0.04
EARLY_CANCEL_RATE = 0.06
LATE_CANCEL_RATE = 0.03

TRUE_COEFFICIENTS = {
    "intercept": -2.4,
    "long_lead": 0.7,
    "young_adult": 0.6,
    "most_deprived": 0.5,
    "follow_up": -0.2,
}
CLINIC_EFFECT_SD = 0.10
PLANTED_EFFECTS = {"C07": 0.8, "C21": 0.8, "C40": -1.0, "C12": 0.0, "C33": 0.0}
PLANTED_SIZES = {"C07": 2800, "C21": 2800, "C40": 2800, "C12": 1500, "C33": 120}
C12_LONG_LEAD_SHARE = 0.70
C12_YOUNG_ADULT_SHARE = 0.50
C33_FIXED_NOSHOW_RATE = 0.15

N_CLINICS = 42
START_DATE = pd.Timestamp("2025-01-01")
N_DAYS = 546  # 2025-01-01 to 2026-06-30 inclusive

COLUMNS = [
    "appointment_id",
    "clinic",
    "specialty",
    "date",
    "age_band",
    "deprivation_quintile",
    "appt_type",
    "lead_days",
    "lead_band",
    "outcome",
]


@dataclass(frozen=True)
class SynthConfig:
    seed: int = 2026
    n_appointments: int = 60_000


def clinic_ids() -> list[str]:
    return [f"C{i:02d}" for i in range(1, N_CLINICS + 1)]


def clinic_specialty(clinic: str) -> str:
    return SPECIALTIES[(int(clinic[1:]) - 1) % len(SPECIALTIES)]


def clinic_sizes(n_appointments: int, rng: np.random.Generator) -> dict[str, int]:
    """Planted clinics keep their fixed size; the rest share what is left, 400 or more each."""
    planted_total = sum(PLANTED_SIZES.values())
    others = [c for c in clinic_ids() if c not in PLANTED_SIZES]
    remaining = n_appointments - planted_total
    if remaining < 400 * len(others):
        raise ValueError("n_appointments is too small for the planted clinics")
    weights = rng.uniform(0.4, 1.6, size=len(others))
    spare = remaining - 400 * len(others)
    extra = np.floor(weights / weights.sum() * spare).astype(int)
    extra[-1] += spare - extra.sum()
    sizes = {c: 400 + int(e) for c, e in zip(others, extra, strict=True)}
    sizes.update(PLANTED_SIZES)
    return {c: sizes[c] for c in clinic_ids()}


def _clinic_mix(clinic: str, rng: np.random.Generator) -> tuple[float, float]:
    """Share of long-lead appointments and of patients aged 18-29 for one clinic."""
    if clinic == "C12":
        return C12_LONG_LEAD_SHARE, C12_YOUNG_ADULT_SHARE
    long_share = float(np.clip(0.20 + rng.normal(0, 0.04), 0.08, 0.35))
    young_share = float(np.clip(0.15 + rng.normal(0, 0.03), 0.06, 0.28))
    return long_share, young_share


def _draw_clinic(clinic: str, n: int, effect: float, rng: np.random.Generator) -> pd.DataFrame:
    long_share, young_share = _clinic_mix(clinic, rng)
    lead_p = [(1 - long_share) * 0.55, (1 - long_share) * 0.45, long_share]
    lead_band = rng.choice(LEAD_BANDS, size=n, p=lead_p)
    lead_days = np.empty(n, dtype=int)
    for band, (lo, hi) in LEAD_DAY_RANGES.items():
        mask = lead_band == band
        lead_days[mask] = rng.integers(lo, hi + 1, size=int(mask.sum()))

    base = np.array(BASE_AGE_PROBS)
    age_p = base * (1 - young_share) / (1 - base[1])
    age_p[1] = young_share
    age_band = rng.choice(AGE_BANDS, size=n, p=age_p)
    quintile = rng.integers(1, 6, size=n)
    specialty = clinic_specialty(clinic)
    new_share = NEW_SHARE[SPECIALTIES.index(specialty)]
    appt_type = np.where(rng.random(n) < new_share, "new", "follow_up")
    date = START_DATE + pd.to_timedelta(rng.integers(0, N_DAYS, size=n), unit="D")

    t = TRUE_COEFFICIENTS
    logit = (
        t["intercept"]
        + t["long_lead"] * (lead_band == ">6w")
        + t["young_adult"] * (age_band == "18-29")
        + t["most_deprived"] * (quintile == 1)
        + t["follow_up"] * (appt_type == "follow_up")
        + effect
    )
    p_noshow = 1.0 / (1.0 + np.exp(-logit))

    u = rng.random(n)
    outcome = np.full(n, "attended", dtype=object)
    cut1 = HOSPITAL_CANCEL_RATE
    cut2 = cut1 + EARLY_CANCEL_RATE
    cut3 = cut2 + LATE_CANCEL_RATE
    outcome[u < cut1] = "hospital_cancel"
    outcome[(u >= cut1) & (u < cut2)] = "early_patient_cancel"
    outcome[(u >= cut2) & (u < cut3)] = "late_patient_cancel"
    eligible = u >= cut3
    noshow = rng.random(n) < p_noshow
    if clinic == "C33":
        # Fix the realised count so the high raw rate is a chance high, not a model outcome.
        idx = np.flatnonzero(eligible)
        k = int(round(C33_FIXED_NOSHOW_RATE * len(idx)))
        noshow = np.zeros(n, dtype=bool)
        noshow[rng.choice(idx, size=k, replace=False)] = True
    outcome[eligible & noshow] = "no_show"

    frame = pd.DataFrame(
        {
            "clinic": clinic,
            "specialty": specialty,
            "date": date,
            "age_band": age_band,
            "deprivation_quintile": quintile,
            "appt_type": appt_type,
            "lead_days": lead_days,
            "lead_band": lead_band,
            "outcome": outcome,
        }
    )
    return frame.sort_values("date", kind="stable").reset_index(drop=True)


def generate(config: SynthConfig | None = None) -> pd.DataFrame:
    """Return the full appointment table for ``config`` (deterministic for a given seed)."""
    config = config or SynthConfig()
    rng = np.random.default_rng(config.seed)
    sizes = clinic_sizes(config.n_appointments, rng)
    parts = []
    for clinic in clinic_ids():
        effect = PLANTED_EFFECTS.get(clinic)
        if effect is None:
            effect = float(rng.normal(0, CLINIC_EFFECT_SD))
        parts.append(_draw_clinic(clinic, sizes[clinic], effect, rng))
    frame = pd.concat(parts, ignore_index=True)
    frame.insert(0, "appointment_id", [f"A{i:06d}" for i in range(1, len(frame) + 1)])
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame[COLUMNS]
