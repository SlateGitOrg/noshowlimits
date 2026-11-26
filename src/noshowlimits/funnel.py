"""Observed over expected, overdispersion and funnel limits.

For clinic i with observed no-shows O and expected E (from the case-mix model):

    ratio = O / E,    z = (O - E) / sqrt(E)

Spiegelhalter's overdispersion factor is estimated on winsorised z-scores, so the clinics that are
plausibly outliers cannot inflate the factor that is then used to judge them:

    phi = mean(z_w ** 2),   z_w = z clipped to its 10th and 90th percentiles

Two sets of limits on the ratio scale, each for 95% and 99.8% two-sided coverage:

    multiplicative:  1 +/- z_a * sqrt(phi / E)
    additive:        1 +/- z_a * sqrt(1 / E + tau2)
    tau2 = max(0, (I*phi - (I - 1)) / (sum(w) - sum(w^2) / sum(w))),  w = E, I = clinics

Lower limits are floored at 0 because a ratio cannot be negative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

Method = Literal["multiplicative", "additive"]
METHODS: tuple[Method, ...] = ("multiplicative", "additive")
METHOD_SUFFIX = {"multiplicative": "mult", "additive": "add"}

Z_95 = float(norm.ppf(0.975))
Z_998 = float(norm.ppf(0.999))
WINSOR_LOWER = 10
WINSOR_UPPER = 90
STATUSES = ("above_99.8", "above_95", "within", "below_95", "below_99.8")


def z_scores(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    return (observed - expected) / np.sqrt(expected)


def winsorise(z: np.ndarray, lower_pct: float = WINSOR_LOWER, upper_pct: float = WINSOR_UPPER):
    """Clip ``z`` to its own lower and upper percentiles (linear interpolation)."""
    lo, hi = np.percentile(z, [lower_pct, upper_pct])
    return np.clip(z, lo, hi)


def estimate_phi(observed: np.ndarray, expected: np.ndarray) -> float:
    """Overdispersion factor: mean of squared winsorised z-scores."""
    return float(np.mean(winsorise(z_scores(observed, expected)) ** 2))


def estimate_tau2(expected: np.ndarray, phi: float) -> float:
    """Between-clinic variance on the ratio scale by the moment estimator, weights w = E."""
    w = np.asarray(expected, dtype=float)
    n = len(w)
    denominator = w.sum() - (w**2).sum() / w.sum()
    return float(max(0.0, (n * phi - (n - 1)) / denominator))


def limits(
    expected: np.ndarray | float, method: Method, phi: float, tau2: float
) -> dict[str, np.ndarray]:
    """Limits on O/E at 95% and 99.8%: keys ``lower_95``, ``upper_95``, ``lower_998``,
    ``upper_998``."""
    e = np.asarray(expected, dtype=float)
    if method == "multiplicative":
        spread = np.sqrt(phi / e)
    elif method == "additive":
        spread = np.sqrt(1.0 / e + tau2)
    else:
        raise ValueError(f"unknown method {method!r}")
    return {
        "lower_95": np.maximum(0.0, 1.0 - Z_95 * spread),
        "upper_95": 1.0 + Z_95 * spread,
        "lower_998": np.maximum(0.0, 1.0 - Z_998 * spread),
        "upper_998": 1.0 + Z_998 * spread,
    }


def status(ratio: np.ndarray, lims: dict[str, np.ndarray]) -> np.ndarray:
    ratio = np.asarray(ratio, dtype=float)
    out = np.full(ratio.shape, "within", dtype=object)
    out[ratio < lims["lower_95"]] = "below_95"
    out[ratio > lims["upper_95"]] = "above_95"
    out[ratio < lims["lower_998"]] = "below_99.8"
    out[ratio > lims["upper_998"]] = "above_99.8"
    return out


@dataclass(frozen=True)
class FunnelResult:
    table: pd.DataFrame  # one row per clinic
    phi: float
    tau2: float
    overall_rate: float


def build_funnel(
    counts: pd.DataFrame, expected: pd.DataFrame, phi_override: float | None = None
) -> FunnelResult:
    """Join per-clinic counts with expected values and attach ratios, limits and status.

    ``phi_override`` forces the overdispersion factor (``1.0`` gives plain Poisson limits)."""
    table = counts.merge(expected, on="clinic", validate="one_to_one").sort_values("clinic")
    table = table.reset_index(drop=True)
    table = table.rename(columns={"noshows": "observed"})
    o = table["observed"].to_numpy(dtype=float)
    e = table["expected"].to_numpy(dtype=float)
    phi = estimate_phi(o, e) if phi_override is None else float(phi_override)
    tau2 = estimate_tau2(e, phi)
    overall_rate = float(o.sum() / table["eligible"].sum())

    table["z"] = z_scores(o, e)
    table["oe_ratio"] = o / e
    table["adjusted_rate"] = table["oe_ratio"] * overall_rate
    table["overall_rate"] = overall_rate
    table["phi"] = phi
    table["tau2"] = tau2
    table["raw_rank"] = table["raw_rate"].rank(ascending=False, method="first").astype(int)
    for method in METHODS:
        lims = limits(e, method, phi, tau2)
        suffix = METHOD_SUFFIX[method]
        for key, values in lims.items():
            table[f"{key}_{suffix}"] = values
        table[f"status_{suffix}"] = status(table["oe_ratio"].to_numpy(), lims)
    return FunnelResult(table=table, phi=phi, tau2=tau2, overall_rate=overall_rate)


def outside_95(table: pd.DataFrame, method: Method) -> int:
    """Number of clinics whose status under ``method`` is anything but ``within``."""
    return int((table[f"status_{METHOD_SUFFIX[method]}"] != "within").sum())
