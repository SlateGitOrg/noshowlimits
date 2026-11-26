"""The field registry: the single description of what the workbook contains.

The builder writes the ``.twb`` from this registry and the validator checks a ``.twb`` (and the
exported CSV headers) against it. Each calculated field also records which exported column it must
reproduce, per limit method, so tests can evaluate the Tableau formula over the exported rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..export import CLINIC_FUNNEL_COLUMNS, FUNNEL_PLOT_COLUMNS
from ..funnel import Z_95, Z_998

WORKBOOK_FILE = "NoShowLimits.twb"
PARAMETER_DATASOURCE = "Parameters"
PARAMETER_NAME = "Parameter 1"
PARAMETER_CAPTION = "Limit Method"
PARAMETER_VALUES = ("multiplicative", "additive")
PARAMETER_REF = f"[{PARAMETER_DATASOURCE}].[{PARAMETER_NAME}]"

_STRING_COLUMNS = {"clinic", "specialty", "status_mult", "status_add", "series", "method", "status"}
_INTEGER_COLUMNS = {
    "appointments",
    "eligible",
    "excluded",
    "excl_hospital_cancel",
    "excl_early_patient_cancel",
    "excl_late_patient_cancel",
    "observed",
    "raw_rank",
}


@dataclass(frozen=True)
class Column:
    name: str
    datatype: str  # string | integer | real

    @property
    def role(self) -> str:
        return "dimension" if self.datatype in ("string", "boolean") else "measure"


@dataclass(frozen=True)
class Calc:
    caption: str
    formula: str
    datatype: str
    # Limit method (or "*" for both) -> exported column the formula must reproduce.
    reference: dict[str, str] = field(default_factory=dict)

    @property
    def role(self) -> str:
        return "dimension" if self.datatype in ("string", "boolean") else "measure"


@dataclass(frozen=True)
class DataSource:
    key: str
    name: str
    caption: str
    csv: str
    columns: tuple[Column, ...]
    calcs: tuple[Calc, ...]

    @property
    def column_names(self) -> set[str]:
        return {c.name for c in self.columns}

    @property
    def calc_names(self) -> set[str]:
        return {c.caption for c in self.calcs}


@dataclass(frozen=True)
class Shelf:
    """A field placed on a shelf: datasource key, field name, derivation and type suffix."""

    ds: str
    field: str
    derivation: str = "none"  # none | sum
    kind: str = "qk"  # qk continuous measure, nk discrete dimension, ok ordinal


@dataclass(frozen=True)
class Pane:
    """One pane of a sheet: its mark class and what colours and details its marks."""

    mark: str
    color: Shelf | None = None
    detail: tuple[Shelf, ...] = ()


@dataclass(frozen=True)
class Worksheet:
    """A sheet over ONE datasource. ``dual_axis`` sheets put two measures on Rows, one per pane,
    on synchronized axes."""

    name: str
    datasource: str
    rows: tuple[Shelf, ...]
    cols: tuple[Shelf, ...]
    panes: tuple[Pane, ...]
    filters: tuple[tuple[Shelf, str], ...] = ()  # (field, member kept)
    sort_field: Shelf | None = None  # sorted descending when set
    dual_axis: bool = False
    aggregate: bool = True

    @property
    def shelves(self) -> tuple[Shelf, ...]:
        extra: list[Shelf] = []
        for pane in self.panes:
            if pane.color:
                extra.append(pane.color)
            extra.extend(pane.detail)
        extra.extend(f for f, _ in self.filters)
        if self.sort_field:
            extra.append(self.sort_field)
        return self.rows + self.cols + tuple(extra)


@dataclass(frozen=True)
class Dashboard:
    name: str
    worksheets: tuple[str, ...]


def _column(name: str) -> Column:
    if name in _STRING_COLUMNS:
        return Column(name, "string")
    if name in _INTEGER_COLUMNS:
        return Column(name, "integer")
    return Column(name, "real")


def _limit_formulas(sign: str, z: float) -> str:
    """Formula for one limit, computed from the clinic's expected count, phi and tau2."""
    add = f"1 {sign} {z!r} * SQRT(1 / [expected] + [tau2])"
    mult = f"1 {sign} {z!r} * SQRT([phi] / [expected])"
    if sign == "-":
        add, mult = f"MAX(0, {add})", f"MAX(0, {mult})"
    return f'IIF({PARAMETER_REF} = "additive", {add}, {mult})'


def _limit_calcs() -> tuple[Calc, ...]:
    specs = (
        ("Upper 95", "+", Z_95, "upper_95"),
        ("Upper 99.8", "+", Z_998, "upper_998"),
        ("Lower 95", "-", Z_95, "lower_95"),
        ("Lower 99.8", "-", Z_998, "lower_998"),
    )
    return tuple(
        Calc(
            caption=caption,
            formula=_limit_formulas(sign, z),
            datatype="real",
            reference={"multiplicative": f"{column}_mult", "additive": f"{column}_add"},
        )
        for caption, sign, z, column in specs
    )


FUNNEL_STATUS_FORMULA = (
    'IIF([O/E Ratio] > [Upper 99.8], "above_99.8", '
    'IIF([O/E Ratio] > [Upper 95], "above_95", '
    'IIF([O/E Ratio] < [Lower 99.8], "below_99.8", '
    'IIF([O/E Ratio] < [Lower 95], "below_95", "within"))))'
)

CLINIC_FUNNEL = DataSource(
    key="clinic_funnel",
    name="federated.clinic_funnel",
    caption="Clinic funnel",
    csv="clinic_funnel.csv",
    columns=tuple(_column(c) for c in CLINIC_FUNNEL_COLUMNS),
    calcs=(
        Calc("O/E Ratio", "[observed] / [expected]", "real", {"*": "oe_ratio"}),
        Calc("Risk-adjusted Rate", "[O/E Ratio] * [overall_rate]", "real", {"*": "adjusted_rate"}),
        *_limit_calcs(),
        Calc(
            "Funnel Status",
            FUNNEL_STATUS_FORMULA,
            "string",
            {"multiplicative": "status_mult", "additive": "status_add"},
        ),
    ),
)

# Long-format table: limit lines and clinic points are rows of one datasource, so the funnel plot
# is a true dual axis over a single source (no blending). Method Match keeps the rows of the
# method chosen in the Limit Method parameter.
FUNNEL_PLOT = DataSource(
    key="funnel_plot",
    name="federated.funnel_plot",
    caption="Funnel plot",
    csv="funnel_plot.csv",
    columns=tuple(_column(c) for c in FUNNEL_PLOT_COLUMNS),
    calcs=(
        Calc("Method Match", f"[method] = {PARAMETER_REF}", "boolean"),
        Calc("Limit Value", 'IIF([series] <> "clinic", [value], NULL)', "real"),
        Calc("Clinic O/E", 'IIF([series] = "clinic", [value], NULL)', "real"),
    ),
)

DATASOURCES = (CLINIC_FUNNEL, FUNNEL_PLOT)
DATASOURCE_BY_KEY = {d.key: d for d in DATASOURCES}

WORKSHEETS = (
    Worksheet(
        name="Funnel Plot",
        datasource="funnel_plot",
        cols=(Shelf("funnel_plot", "expected"),),
        rows=(Shelf("funnel_plot", "Limit Value"), Shelf("funnel_plot", "Clinic O/E")),
        panes=(
            Pane("Line", color=Shelf("funnel_plot", "series", kind="nk")),
            Pane(
                "Circle",
                color=Shelf("funnel_plot", "status", kind="nk"),
                detail=(Shelf("funnel_plot", "clinic", kind="nk"),),
            ),
        ),
        filters=((Shelf("funnel_plot", "Method Match", kind="nk"), "true"),),
        dual_axis=True,
        aggregate=False,
    ),
    Worksheet(
        name="Raw Rate League Table",
        datasource="clinic_funnel",
        cols=(Shelf("clinic_funnel", "raw_rate", "sum"),),
        rows=(Shelf("clinic_funnel", "clinic", kind="nk"),),
        panes=(
            Pane(
                "Bar",
                color=Shelf("clinic_funnel", "Funnel Status", kind="nk"),
                detail=(Shelf("clinic_funnel", "raw_rank"),),
            ),
        ),
        sort_field=Shelf("clinic_funnel", "raw_rate", "sum"),
    ),
    Worksheet(
        name="Exclusions Summary",
        datasource="clinic_funnel",
        cols=(Shelf("clinic_funnel", "excluded", "sum"),),
        rows=(Shelf("clinic_funnel", "clinic", kind="nk"),),
        panes=(
            Pane(
                "Bar",
                detail=(
                    Shelf("clinic_funnel", "excl_hospital_cancel", "sum"),
                    Shelf("clinic_funnel", "excl_early_patient_cancel", "sum"),
                    Shelf("clinic_funnel", "excl_late_patient_cancel", "sum"),
                ),
            ),
        ),
    ),
)
DASHBOARDS = (
    Dashboard(
        name="No-show Limits",
        worksheets=("Funnel Plot", "Raw Rate League Table", "Exclusions Summary"),
    ),
)
