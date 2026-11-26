"""Static validation of a ``.twb`` against the field registry and the exported CSVs.

Tableau cannot run in CI, so this is the check that the workbook is consistent with the data it
reads: every column a datasource declares must be in its CSV, every field a calculated field or
worksheet uses must exist, and every caption and formula must equal the registry.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from lxml import etree

from . import formula as fm
from . import registry as reg

_SHELF_REF = re.compile(r"\[[^\]]+\]\.\[\w+:[^:\]]+:\w+\]")
_INSTANCE = re.compile(
    r"\[(?P<ds>[^\]]+)\]\.\[(?P<derivation>\w+):(?P<field>[^:\]]+):(?P<kind>\w+)\]"
)


def reg_ref(shelf: reg.Shelf) -> str:
    """The shelf reference the builder writes for ``shelf``."""
    ds = reg.DATASOURCE_BY_KEY[shelf.ds]
    return f"[{ds.name}].[{shelf.derivation}:{shelf.field}:{shelf.kind}]"


def csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def _find_datasource(root: etree._Element, name: str) -> etree._Element | None:
    for node in root.findall("./datasources/datasource"):
        if node.get("name") == name:
            return node
    return None


def _check_parameter(root: etree._Element, errors: list[str]) -> None:
    node = _find_datasource(root, reg.PARAMETER_DATASOURCE)
    if node is None:
        errors.append(f"parameter datasource '{reg.PARAMETER_DATASOURCE}' is missing")
        return
    column = node.find(f"./column[@name='[{reg.PARAMETER_NAME}]']")
    if column is None:
        errors.append(f"parameter '{reg.PARAMETER_NAME}' is missing")
        return
    if column.get("caption") != reg.PARAMETER_CAPTION:
        got, want = column.get("caption"), reg.PARAMETER_CAPTION
        errors.append(f"parameter caption is {got!r}, registry says {want!r}")
    members = [m.get("value") for m in column.findall("./members/member")]
    expected = [f'"{v}"' for v in reg.PARAMETER_VALUES]
    if members != expected:
        errors.append(f"parameter values are {members}, registry says {expected}")


def _check_datasource(
    root: etree._Element,
    source: reg.DataSource,
    data_dir: Path | None,
    twb_dir: Path,
    errors: list[str],
) -> set[str]:
    """Validate one datasource; return the field names usable in formulas and worksheets."""
    label = f"datasource '{source.caption}'"
    node = _find_datasource(root, source.name)
    if node is None:
        errors.append(f"{label} is missing from the workbook")
        return set()

    connection = node.find("./connection/named-connections/named-connection/connection")
    if connection is None or connection.get("class") != "textscan":
        errors.append(f"{label} must read a textscan connection")
        directory = None
    else:
        if connection.get("filename") != source.csv:
            errors.append(
                f"{label} reads {connection.get('filename')!r}, registry says {source.csv!r}"
            )
        directory = connection.get("directory")
    base = data_dir if data_dir is not None else twb_dir / (directory or ".")
    csv_path = base / source.csv
    if not csv_path.exists():
        errors.append(f"{label}: CSV {csv_path} does not exist")
        return set()
    header = csv_header(csv_path)
    header_set = set(header)

    declared = [c.get("name") for c in node.findall("./connection/relation/columns/column")]
    for name in declared:
        if name not in header_set:
            errors.append(f"{label}: column '{name}' is declared but is not in {source.csv}")
    for name in header:
        if name not in declared:
            errors.append(
                f"{label}: column '{name}' of {source.csv} is not declared in the workbook"
            )
    for name in sorted(source.column_names - header_set):
        errors.append(f"{label}: registry column '{name}' is not in {source.csv}")

    usable = set(header) | source.calc_names
    calcs = {}
    for column in node.findall("./column"):
        calculation = column.find("calculation")
        if calculation is None:
            continue
        caption = column.get("caption")
        calcs[caption] = calculation.get("formula")
    registry_calcs = {c.caption: c for c in source.calcs}
    for caption in sorted(set(calcs) - set(registry_calcs)):
        errors.append(f"{label}: calculated field '{caption}' is not in the registry")
    for caption, calc in registry_calcs.items():
        if caption not in calcs:
            errors.append(f"{label}: calculated field '{caption}' is missing from the workbook")
        elif calcs[caption] != calc.formula:
            errors.append(f"{label}: formula of '{caption}' differs from the registry")
    for caption, text in calcs.items():
        try:
            refs = fm.references(text)
            params = fm.parameter_references(text)
        except fm.FormulaError as exc:
            errors.append(f"{label}: calculated field '{caption}' does not parse: {exc}")
            continue
        for ref in refs:
            if ref not in usable:
                errors.append(f"{label}: calculated field '{caption}' uses unknown field '{ref}'")
        for param in params:
            if param != reg.PARAMETER_NAME:
                errors.append(
                    f"{label}: calculated field '{caption}' uses unknown parameter '{param}'"
                )
    return usable


def _check_worksheets(
    root: etree._Element, usable: dict[str, set[str]], errors: list[str]
) -> set[str]:
    by_name = {s.name: s for s in reg.DATASOURCES}
    sheets = root.findall("./worksheets/worksheet")
    names = [s.get("name") for s in sheets]
    expected = [w.name for w in reg.WORKSHEETS]
    for name in expected:
        if name not in names:
            errors.append(f"worksheet '{name}' is missing from the workbook")
    for name in names:
        if name not in expected:
            errors.append(f"worksheet '{name}' is not in the registry")

    def check_field(sheet: str, ds_name: str, field: str) -> None:
        if ds_name == reg.PARAMETER_DATASOURCE:
            return
        if ds_name not in usable:
            errors.append(f"worksheet '{sheet}' refers to unknown datasource '{ds_name}'")
        elif field not in usable[ds_name]:
            errors.append(
                f"worksheet '{sheet}' uses field '{field}' that is not in "
                f"{by_name[ds_name].csv} or its calculated fields"
            )

    registry_sheets = {w.name: w for w in reg.WORKSHEETS}
    for sheet in sheets:
        name = sheet.get("name")
        if name in registry_sheets:
            _check_structure(sheet, registry_sheets[name], errors)
        for deps in sheet.findall(".//datasource-dependencies"):
            ds_name = deps.get("datasource")
            for column in deps.findall("column"):
                field = column.get("name")[1:-1]
                if ds_name == reg.PARAMETER_DATASOURCE:
                    continue
                check_field(name, ds_name, field)
                calc = column.find("calculation")
                registry_calc = (
                    next((c for c in by_name[ds_name].calcs if c.caption == field), None)
                    if ds_name in by_name
                    else None
                )
                if (
                    calc is not None
                    and registry_calc
                    and calc.get("formula") != registry_calc.formula
                ):
                    errors.append(
                        f"worksheet '{name}': formula of '{field}' differs from the registry"
                    )
            for instance in deps.findall("column-instance"):
                check_field(name, ds_name, instance.get("column")[1:-1])
        text_nodes = [sheet.find(".//rows"), sheet.find(".//cols")]
        attrs = [
            node.get(attr)
            for node in sheet.iter()
            for attr in ("column", "using")
            if node.tag in ("color", "lod", "computed-sort") and node.get(attr)
        ]
        references = [n.text or "" for n in text_nodes if n is not None] + attrs
        for text in references:
            for match in _INSTANCE.finditer(text):
                check_field(name, match.group("ds"), match.group("field"))
    return set(names)


def _refs(text: str | None) -> list[str]:
    return [m.group(0) for m in _SHELF_REF.finditer(text or "")]


def _check_structure(node: etree._Element, ws: reg.Worksheet, errors: list[str]) -> None:
    """Structural checks that a sheet is drawn the way the registry says: one datasource, the
    right marks per pane, no stacked measures, and a declared, synchronized dual axis."""
    label = f"worksheet '{ws.name}'"
    source = reg.DATASOURCE_BY_KEY[ws.datasource]
    names = [d.get("name") for d in node.findall("./table/view/datasources/datasource")]
    names = [n for n in names if n != reg.PARAMETER_DATASOURCE]
    if names != [source.name]:
        errors.append(
            f"{label} must draw from the single datasource '{source.name}', found {names}"
        )
    dep_names = [
        d.get("datasource")
        for d in node.findall("./table/view/datasource-dependencies")
        if d.get("datasource") != reg.PARAMETER_DATASOURCE
    ]
    if dep_names != [source.name]:
        errors.append(f"{label} has dependencies on {dep_names}, expected only '{source.name}'")

    marks = [m.get("class") for m in node.findall("./table/panes/pane/mark")]
    want_marks = [p.mark for p in ws.panes]
    if marks != want_marks:
        errors.append(f"{label} has mark types {marks}, expected {want_marks}")

    rows = _refs(node.findtext("./table/rows"))
    cols = _refs(node.findtext("./table/cols"))
    want_rows = [reg_ref(s) for s in ws.rows]
    want_cols = [reg_ref(s) for s in ws.cols]
    if cols != want_cols or rows != want_rows:
        errors.append(
            f"{label} has rows {rows} and columns {cols}, expected {want_rows} and {want_cols}"
        )
    for shelf, refs in (("rows", rows), ("columns", cols)):
        if len(refs) > 1 and not (ws.dual_axis and shelf == "rows" and len(refs) == 2):
            errors.append(
                f"{label} stacks {len(refs)} fields on {shelf}; "
                "stacked measures render as separate panels"
            )

    panes = node.findall("./table/panes/pane")
    if ws.dual_axis:
        text = (node.findtext("./table/rows") or "").strip()
        if len(rows) != 2 or not (text.startswith("(") and " + " in text):
            errors.append(f"{label} must place exactly two measures on rows as a dual axis")
            return
        if len(panes) != 2:
            errors.append(f"{label} declares {len(panes)} panes, a dual axis needs 2")
            return
        for pane, ref in zip(panes, rows, strict=True):
            if pane.get("y-axis-name") != ref:
                errors.append(f"{label}: pane {pane.get('id')} is not bound to axis {ref}")
        sync = [
            e
            for e in node.findall("./table/style/style-rule[@element='axis']/encoding")
            if e.get("synchronized") == "true" and e.get("field") == rows[1]
        ]
        if not sync:
            errors.append(f"{label}: the dual axis is not synchronized")
    else:
        if any(p.get("y-axis-name") for p in panes):
            errors.append(f"{label} declares a dual axis but the registry says it is single")

    for shelf, member in ws.filters:
        ref = reg_ref(shelf)
        found = [
            g
            for f in node.findall("./table/view/filter")
            if f.get("column") == ref
            for g in f.findall("groupfilter")
            if g.get("member") == member
        ]
        if not found:
            errors.append(f"{label} is missing the filter {ref} = {member}")


def _check_dashboards(root: etree._Element, sheet_names: set[str], errors: list[str]) -> None:
    boards = {d.get("name"): d for d in root.findall("./dashboards/dashboard")}
    for board in reg.DASHBOARDS:
        node = boards.get(board.name)
        if node is None:
            errors.append(f"dashboard '{board.name}' is missing from the workbook")
            continue
        zones = [z.get("name") for z in node.iter("zone") if z.get("name")]
        for zone in zones:
            if zone not in sheet_names:
                errors.append(f"dashboard '{board.name}' shows unknown worksheet '{zone}'")
        for sheet in board.worksheets:
            if sheet not in zones:
                errors.append(f"dashboard '{board.name}' does not show worksheet '{sheet}'")


def validate_twb(twb_path: str | Path, data_dir: str | Path | None = None) -> list[str]:
    """Return a list of problems (empty when the workbook is consistent).

    ``data_dir`` overrides where the CSVs are read from; by default the workbook's own
    ``directory`` setting, relative to the workbook, is used.
    """
    twb_path = Path(twb_path)
    errors: list[str] = []
    try:
        root = etree.parse(str(twb_path)).getroot()
    except (OSError, etree.XMLSyntaxError) as exc:
        return [f"cannot read workbook {twb_path}: {exc}"]
    if root.tag != "workbook":
        return [f"{twb_path} is not a Tableau workbook (root element is '{root.tag}')"]

    data = Path(data_dir) if data_dir is not None else None
    _check_parameter(root, errors)
    usable: dict[str, set[str]] = {}
    for source in reg.DATASOURCES:
        usable[source.name] = _check_datasource(root, source, data, twb_path.parent, errors)
    sheet_names = _check_worksheets(root, usable, errors)
    _check_dashboards(root, sheet_names, errors)
    return errors
