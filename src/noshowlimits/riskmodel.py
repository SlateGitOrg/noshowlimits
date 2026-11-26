"""Case-mix model: a binomial GLM on appointment-level covariates.

The model sees lead-time band, age band, new vs follow-up and deprivation quintile, and nothing
else. It never gets clinic identity or clinic practices such as reminder services, so whatever a
clinic does differently stays in its observed-minus-expected gap instead of being absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata

from .metrics import MetricConfig, classify

# (column, reference level, levels in display order). Reference levels are the usual patient.
FACTORS = (
    ("lead_band", "<=2w", ("<=2w", "2-6w", ">6w")),
    ("age_band", "30-49", ("0-17", "18-29", "30-49", "50-64", "65+")),
    ("appt_type", "new", ("new", "follow_up")),
    ("deprivation_quintile", 5, (1, 2, 3, 4, 5)),
)


def design_matrix(appointments: pd.DataFrame) -> pd.DataFrame:
    """Intercept plus one 0/1 column per non-reference level, in a fixed column order."""
    cols: dict[str, np.ndarray] = {"const": np.ones(len(appointments))}
    for column, reference, levels in FACTORS:
        values = appointments[column].to_numpy()
        for level in levels:
            if level == reference:
                continue
            cols[f"{column}[{level}]"] = (values == level).astype(float)
    return pd.DataFrame(cols, index=appointments.index)


def auc_rank(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve by the rank formula (Mann-Whitney U), ties get average ranks."""
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both outcomes present")
    ranks = rankdata(score, method="average")
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibration_by_decile(y: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    """Ten equal-size groups by predicted risk: mean predicted vs observed no-show rate.

    Ordinal ranks split ties, so groups stay equal in size even though predicted risk takes only
    a few dozen distinct values.
    """
    y = np.asarray(y, dtype=float)
    order = rankdata(predicted, method="ordinal")
    decile = ((order - 1) * 10 // len(y)).astype(int) + 1
    frame = pd.DataFrame({"decile": decile, "y": y, "p": predicted})
    out = frame.groupby("decile").agg(
        n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")
    )
    out["abs_diff"] = (out["observed"] - out["predicted"]).abs()
    return out.reset_index()


@dataclass
class RiskModel:
    coefficients: pd.DataFrame  # coef, std_err, odds_ratio per term
    auc: float
    calibration: pd.DataFrame
    params: pd.Series
    columns: list[str]
    n_obs: int

    def predict(self, appointments: pd.DataFrame) -> np.ndarray:
        x = design_matrix(appointments)[self.columns].to_numpy()
        return 1.0 / (1.0 + np.exp(-(x @ self.params.to_numpy())))


def fit(appointments: pd.DataFrame, config: MetricConfig | None = None) -> RiskModel:
    """Fit on appointments in the no-show denominator."""
    eligible = classify(appointments, config)
    eligible = eligible[eligible["in_denominator"]]
    x = design_matrix(eligible)
    y = eligible["is_noshow"].to_numpy().astype(float)
    result = sm.GLM(y, x, family=sm.families.Binomial()).fit()
    params = pd.Series(result.params, index=x.columns)
    table = pd.DataFrame(
        {
            "coef": params,
            "std_err": pd.Series(result.bse, index=x.columns),
            "odds_ratio": np.exp(params),
        }
    )
    predicted = result.predict(x)
    return RiskModel(
        coefficients=table,
        auc=auc_rank(y, predicted),
        calibration=calibration_by_decile(y, predicted),
        params=params,
        columns=list(x.columns),
        n_obs=len(y),
    )


def expected_by_clinic(
    appointments: pd.DataFrame, model: RiskModel, config: MetricConfig | None = None
) -> pd.DataFrame:
    """Expected no-shows E per clinic: the sum of predicted probabilities over its eligible
    appointments."""
    eligible = classify(appointments, config)
    eligible = eligible[eligible["in_denominator"]].copy()
    eligible["predicted"] = model.predict(eligible)
    out = eligible.groupby("clinic", sort=True)["predicted"].sum().rename("expected")
    return out.reset_index()
