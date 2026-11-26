from __future__ import annotations

import pandas as pd
import pytest

from noshowlimits import pipeline, synth


@pytest.fixture(scope="session")
def appointments() -> pd.DataFrame:
    return synth.generate()


@pytest.fixture(scope="session")
def analysis(appointments: pd.DataFrame) -> pipeline.Analysis:
    return pipeline.analyse(appointments)


@pytest.fixture(scope="session")
def table(analysis: pipeline.Analysis) -> pd.DataFrame:
    return analysis.funnel.table.set_index("clinic", drop=False)


@pytest.fixture()
def hand_fixture() -> pd.DataFrame:
    """Ten appointments covering every outcome. Clinic A has six, clinic B has four."""
    rows = [
        ("A", "attended"),
        ("A", "attended"),
        ("A", "no_show"),
        ("A", "late_patient_cancel"),
        ("A", "early_patient_cancel"),
        ("A", "hospital_cancel"),
        ("B", "attended"),
        ("B", "no_show"),
        ("B", "late_patient_cancel"),
        ("B", "hospital_cancel"),
    ]
    return pd.DataFrame(
        {
            "clinic": [r[0] for r in rows],
            "specialty": "Cardiology",
            "outcome": [r[1] for r in rows],
        }
    )
