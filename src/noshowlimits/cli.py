"""Command line interface: generate, fit, funnel, league, export, validate-twb."""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import export as export_mod
from . import funnel as funnel_mod
from . import metrics, pipeline, synth
from .tableau import build, validate

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console()

DEFAULT_DATA = Path("data/appointments.csv")
DataOpt = Annotated[Path, typer.Option("--data", help="Appointments CSV from `generate`.")]


class LateCancel(StrEnum):
    exclude = "exclude"
    noshow = "noshow"


class Method(StrEnum):
    multiplicative = "multiplicative"
    additive = "additive"


LateOpt = Annotated[
    LateCancel,
    typer.Option(
        "--late-cancel", help="Treat patient cancellations under 24 h as no-shows or drop them."
    ),
]


def _config(late_cancel: LateCancel) -> metrics.MetricConfig:
    return metrics.MetricConfig(late_cancel=late_cancel.value)


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        console.print(
            f"[red]{escape(str(path))} not found. Run `noshowlimits generate` first.[/red]"
        )
        raise typer.Exit(code=1)
    return pipeline.load_appointments(path)


def _table(title: str, columns: list[tuple[str, str]]) -> Table:
    table = Table(title=title, box=box.SIMPLE_HEAD, title_justify="left")
    for name, justify in columns:
        table.add_column(name, justify=justify)  # type: ignore[arg-type]
    return table


@app.command()
def generate(
    seed: Annotated[int, typer.Option(help="Random seed.")] = synth.SynthConfig.seed,
    n_appointments: Annotated[int, typer.Option(help="Total appointments.")] = 60_000,
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("data"),
) -> None:
    """Write the seeded synthetic appointment table to OUT/appointments.csv."""
    frame = synth.generate(synth.SynthConfig(seed=seed, n_appointments=n_appointments))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "appointments.csv"
    frame.to_csv(path, index=False)
    console.print(
        f"wrote {len(frame):,} appointments for {frame['clinic'].nunique()} clinics to {path}"
    )


@app.command()
def fit(data: DataOpt = DEFAULT_DATA, late_cancel: LateOpt = LateCancel.exclude) -> None:
    """Fit the case-mix model and print coefficients, AUC and decile calibration."""
    model = pipeline.analyse(_load(data), _config(late_cancel)).model
    coefs = _table(
        f"Case-mix model (binomial GLM, n = {model.n_obs:,}, AUC = {model.auc:.3f})",
        [("term", "left"), ("coef", "right"), ("std err", "right"), ("odds ratio", "right")],
    )
    for term, row in model.coefficients.iterrows():
        coefs.add_row(
            escape(str(term)), f"{row.coef:.3f}", f"{row.std_err:.3f}", f"{row.odds_ratio:.3f}"
        )
    console.print(coefs)
    cal = _table(
        "Calibration by decile of predicted risk",
        [
            ("decile", "right"),
            ("n", "right"),
            ("predicted", "right"),
            ("observed", "right"),
            ("|diff|", "right"),
        ],
    )
    for row in model.calibration.itertuples():
        cal.add_row(
            str(row.decile),
            f"{row.n:,}",
            f"{row.predicted:.4f}",
            f"{row.observed:.4f}",
            f"{row.abs_diff:.4f}",
        )
    console.print(cal)


def _summary(result: funnel_mod.FunnelResult) -> str:
    return (
        f"overall no-show rate {result.overall_rate:.4f}   "
        f"phi = {result.phi:.3f}   tau2 = {result.tau2:.5f}"
    )


@app.command("funnel")
def funnel_cmd(
    method: Annotated[Method, typer.Option(help="Limit method.")] = Method.multiplicative,
    data: DataOpt = DEFAULT_DATA,
    late_cancel: LateOpt = LateCancel.exclude,
    phi: Annotated[float | None, typer.Option(help="Force phi (1.0 gives Poisson limits).")] = None,
    show_all: Annotated[
        bool, typer.Option("--all", help="List clinics within the limits too.")
    ] = False,
) -> None:
    """Print clinics outside the funnel limits under METHOD."""
    result = pipeline.analyse(_load(data), _config(late_cancel), phi_override=phi).funnel
    suffix = funnel_mod.METHOD_SUFFIX[method.value]
    table = result.table
    shown = table if show_all else table[table[f"status_{suffix}"] != "within"]
    console.print(_summary(result))
    out = _table(
        f"Funnel plot, {method.value} limits",
        [
            ("clinic", "left"),
            ("O", "right"),
            ("E", "right"),
            ("O/E", "right"),
            ("95% limits", "right"),
            ("99.8% limits", "right"),
            ("status", "left"),
        ],
    )
    cols95 = (f"lower_95_{suffix}", f"upper_95_{suffix}")
    cols998 = (f"lower_998_{suffix}", f"upper_998_{suffix}")
    for _, row in shown.sort_values("oe_ratio", ascending=False).iterrows():
        out.add_row(
            row["clinic"],
            f"{row['observed']:.0f}",
            f"{row['expected']:.1f}",
            f"{row['oe_ratio']:.3f}",
            f"{row[cols95[0]]:.3f} - {row[cols95[1]]:.3f}",
            f"{row[cols998[0]]:.3f} - {row[cols998[1]]:.3f}",
            row[f"status_{suffix}"],
        )
    console.print(out)
    console.print(f"{len(shown)} of {len(table)} clinics listed")


@app.command()
def league(
    raw: Annotated[bool, typer.Option("--raw", help="Rank by raw no-show rate.")] = False,
    top: Annotated[int, typer.Option(help="Rows to show.")] = 10,
    include: Annotated[
        list[str] | None,
        typer.Option("--include", help="Also show this clinic, whatever its rank (repeatable)."),
    ] = None,
    data: DataOpt = DEFAULT_DATA,
    late_cancel: LateOpt = LateCancel.exclude,
) -> None:
    """League table by raw rate (--raw) or by risk-adjusted rate, with funnel status beside it."""
    result = pipeline.analyse(_load(data), _config(late_cancel)).funnel
    key = "raw_rate" if raw else "adjusted_rate"
    ordered = result.table.sort_values(key, ascending=False)
    unknown = sorted(set(include or []) - set(ordered["clinic"]))
    if unknown:
        console.print(f"[red]unknown clinic(s): {escape(', '.join(unknown))}[/red]")
        raise typer.Exit(code=1)
    ranked = pd.concat([ordered.head(top), ordered[ordered["clinic"].isin(include or [])]])
    ranked = ranked.drop_duplicates("clinic")
    title = "Raw-rate league table" if raw else "Risk-adjusted league table"
    console.print(_summary(result))
    out = _table(
        title,
        [
            ("raw rank", "right"),
            ("clinic", "left"),
            ("eligible", "right"),
            ("O", "right"),
            ("E", "right"),
            ("raw rate", "right"),
            ("O/E", "right"),
            ("status (mult)", "left"),
            ("status (add)", "left"),
        ],
    )
    for row in ranked.itertuples():
        out.add_row(
            str(row.raw_rank),
            row.clinic,
            f"{row.eligible:,.0f}",
            f"{row.observed:.0f}",
            f"{row.expected:.1f}",
            f"{row.raw_rate:.3f}",
            f"{row.oe_ratio:.3f}",
            row.status_mult,
            row.status_add,
        )
    console.print(out)


@app.command("export")
def export_cmd(
    data: DataOpt = DEFAULT_DATA,
    out: Annotated[Path, typer.Option(help="Directory for the two CSVs.")] = Path("tableau/data"),
    twb: Annotated[Path, typer.Option(help="Workbook to (re)write.")] = Path(
        "tableau/NoShowLimits.twb"
    ),
    late_cancel: LateOpt = LateCancel.exclude,
) -> None:
    """Write clinic_funnel.csv, funnel_limits.csv, funnel_plot.csv and the Tableau workbook."""
    result = pipeline.analyse(_load(data), _config(late_cancel)).funnel
    paths = export_mod.write_exports(result, out)
    directory = Path(_relative(out, twb.parent)).as_posix()
    build.write_workbook(twb, directory=directory)
    console.print(f"wrote {', '.join(str(p) for p in paths)} and {twb}")


def _relative(target: Path, start: Path) -> str:
    return os.path.relpath(target.resolve(), start.resolve())


@app.command("validate-twb")
def validate_twb(
    twb: Annotated[Path, typer.Argument(help="Workbook to check.")] = Path(
        "tableau/NoShowLimits.twb"
    ),
    data_dir: Annotated[
        Path | None, typer.Option(help="Where the CSVs are (default: as the workbook says).")
    ] = None,
) -> None:
    """Statically check the workbook against the registry and the exported CSV headers."""
    problems = validate.validate_twb(twb, data_dir)
    if problems:
        for problem in problems:
            console.print(f"[red]error:[/red] {escape(problem)}", highlight=False)
        raise typer.Exit(code=1)
    console.print(f"{twb} is consistent with its CSVs and the field registry")
