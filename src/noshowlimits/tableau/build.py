"""Write ``NoShowLimits.twb`` from the field registry."""

from __future__ import annotations

from pathlib import Path

from lxml import etree

from . import registry as reg

TWB_VERSION = "18.1"
SOURCE_BUILD = "2024.1.0 (20241.24.0314.1020)"
DASHBOARD_SIZE = (1400, 900)

_DERIVATION_NAME = {"none": "None", "sum": "Sum"}
_TYPE_NAME = {"qk": "quantitative", "nk": "nominal", "ok": "ordinal"}
_NAMESPACE = "http://www.tableausoftware.com/xml/user"


def instance_name(shelf: reg.Shelf) -> str:
    return f"{shelf.derivation}:{shelf.field}:{shelf.kind}"


def shelf_ref(shelf: reg.Shelf) -> str:
    ds = reg.DATASOURCE_BY_KEY[shelf.ds]
    return f"[{ds.name}].[{instance_name(shelf)}]"


def _shelf_expression(shelves: tuple[reg.Shelf, ...], dual_axis: bool) -> str:
    """One field as is; the two measures of a dual axis joined with ``+``. Any other number of
    fields on one shelf is rejected, because stacking measures renders separate panels."""
    refs = [shelf_ref(s) for s in shelves]
    if len(refs) == 1:
        return refs[0]
    if dual_axis and len(refs) == 2:
        return "(" + " + ".join(refs) + ")"
    raise ValueError(f"unsupported shelf of {len(refs)} fields")


def _calc_column(parent: etree._Element, calc: reg.Calc) -> None:
    column = etree.SubElement(
        parent,
        "column",
        caption=calc.caption,
        datatype=calc.datatype,
        name=f"[{calc.caption}]",
        role=calc.role,
        type=_field_type(calc.datatype),
    )
    etree.SubElement(column, "calculation", {"class": "tableau", "formula": calc.formula})


def _parameter_datasource(parent: etree._Element) -> None:
    ds = etree.SubElement(
        parent,
        "datasource",
        {"hasconnection": "false", "inline": "true", "name": reg.PARAMETER_DATASOURCE},
        version=TWB_VERSION,
    )
    etree.SubElement(ds, "aliases", enabled="yes")
    default = f'"{reg.PARAMETER_VALUES[0]}"'
    column = etree.SubElement(
        ds,
        "column",
        {
            "caption": reg.PARAMETER_CAPTION,
            "datatype": "string",
            "name": f"[{reg.PARAMETER_NAME}]",
            "param-domain-type": "list",
            "role": "measure",
            "type": "nominal",
            "value": default,
        },
    )
    etree.SubElement(column, "calculation", {"class": "tableau", "formula": default})
    members = etree.SubElement(column, "members")
    for value in reg.PARAMETER_VALUES:
        etree.SubElement(members, "member", value=f'"{value}"')


def _datasource(parent: etree._Element, source: reg.DataSource, directory: str) -> None:
    ds = etree.SubElement(
        parent,
        "datasource",
        {"caption": source.caption, "inline": "true", "name": source.name},
        version=TWB_VERSION,
    )
    connection = etree.SubElement(ds, "connection", {"class": "federated"})
    named = etree.SubElement(connection, "named-connections")
    named_connection = etree.SubElement(
        named, "named-connection", caption=source.key, name=f"textscan.{source.key}"
    )
    etree.SubElement(
        named_connection,
        "connection",
        {
            "class": "textscan",
            "directory": directory,
            "filename": source.csv,
            "password": "",
            "server": "",
        },
    )
    relation = etree.SubElement(
        connection,
        "relation",
        {
            "connection": f"textscan.{source.key}",
            "name": source.csv,
            "table": f"[{source.csv.replace('.', '#')}]",
            "type": "table",
        },
    )
    columns = etree.SubElement(
        relation,
        "columns",
        {"character-set": "UTF-8", "header": "yes", "locale": "en_US", "separator": ","},
    )
    for ordinal, column in enumerate(source.columns):
        etree.SubElement(
            columns, "column", datatype=column.datatype, name=column.name, ordinal=str(ordinal)
        )
    for calc in source.calcs:
        _calc_column(ds, calc)


def _field_type(datatype: str) -> str:
    return "nominal" if datatype in ("string", "boolean") else "quantitative"


def _field_definition(source: reg.DataSource, name: str) -> tuple[str, str, str]:
    """(datatype, role, type) for a field or calculated field of ``source``."""
    for calc in source.calcs:
        if calc.caption == name:
            return calc.datatype, calc.role, _field_type(calc.datatype)
    column = next(c for c in source.columns if c.name == name)
    return column.datatype, column.role, _field_type(column.datatype)


def _dependencies(view: etree._Element, sheet: reg.Worksheet) -> None:
    source = reg.DATASOURCE_BY_KEY[sheet.datasource]
    deps = etree.SubElement(view, "datasource-dependencies", datasource=source.name)
    seen: set[str] = set()
    for shelf in sheet.shelves:
        if shelf.field in seen:
            continue
        seen.add(shelf.field)
        datatype, role, kind = _field_definition(source, shelf.field)
        column = etree.SubElement(
            deps, "column", datatype=datatype, name=f"[{shelf.field}]", role=role, type=kind
        )
        calc = next((c for c in source.calcs if c.caption == shelf.field), None)
        if calc is not None:
            column.set("caption", calc.caption)
            etree.SubElement(column, "calculation", {"class": "tableau", "formula": calc.formula})
    done: set[str] = set()
    for shelf in sheet.shelves:
        name = instance_name(shelf)
        if name in done:
            continue
        done.add(name)
        etree.SubElement(
            deps,
            "column-instance",
            column=f"[{shelf.field}]",
            derivation=_DERIVATION_NAME[shelf.derivation],
            name=f"[{name}]",
            pivot="key",
            type=_TYPE_NAME[shelf.kind],
        )


def _parameter_dependencies(view: etree._Element) -> None:
    deps = etree.SubElement(view, "datasource-dependencies", datasource=reg.PARAMETER_DATASOURCE)
    etree.SubElement(
        deps,
        "column",
        {
            "caption": reg.PARAMETER_CAPTION,
            "datatype": "string",
            "name": f"[{reg.PARAMETER_NAME}]",
            "param-domain-type": "list",
            "role": "measure",
            "type": "nominal",
        },
    )


def _worksheet(parent: etree._Element, sheet: reg.Worksheet) -> None:
    source = reg.DATASOURCE_BY_KEY[sheet.datasource]
    worksheet = etree.SubElement(parent, "worksheet", name=sheet.name)
    table = etree.SubElement(worksheet, "table")
    view = etree.SubElement(table, "view")
    datasources = etree.SubElement(view, "datasources")
    etree.SubElement(datasources, "datasource", caption=source.caption, name=source.name)
    etree.SubElement(
        datasources, "datasource", hasconnection="false", name=reg.PARAMETER_DATASOURCE
    )
    _dependencies(view, sheet)
    _parameter_dependencies(view)
    for shelf, member in sheet.filters:
        flt = etree.SubElement(view, "filter", {"class": "categorical", "column": shelf_ref(shelf)})
        etree.SubElement(
            flt,
            "groupfilter",
            function="member",
            level=f"[{instance_name(shelf)}]",
            member=member,
        )
    if sheet.sort_field is not None:
        etree.SubElement(
            view,
            "computed-sort",
            column=shelf_ref(sheet.rows[0]),
            direction="DESC",
            using=shelf_ref(sheet.sort_field),
        )
    etree.SubElement(view, "aggregation", value="true" if sheet.aggregate else "false")

    style = etree.SubElement(table, "style")
    if sheet.dual_axis:
        # Both measures share one scale: the secondary axis is synchronized with the primary.
        rule = etree.SubElement(style, "style-rule", element="axis")
        etree.SubElement(
            rule,
            "encoding",
            {
                "attr": "space",
                "class": "0",
                "field": shelf_ref(sheet.rows[1]),
                "field-type": "quantitative",
                "synchronized": "true",
                "type": "space",
            },
        )
    panes = etree.SubElement(table, "panes")
    for index, pane in enumerate(sheet.panes):
        attrs = {"selection-relaxation-option": "selection-relaxation-allow"}
        if sheet.dual_axis:
            attrs["id"] = str(index + 1)
            attrs["y-axis-name"] = shelf_ref(sheet.rows[index])
        node = etree.SubElement(panes, "pane", attrs)
        pane_view = etree.SubElement(node, "view")
        etree.SubElement(pane_view, "breakdown", value="auto")
        etree.SubElement(node, "mark", {"class": pane.mark})
        encodings = etree.SubElement(node, "encodings")
        if pane.color is not None:
            etree.SubElement(encodings, "color", column=shelf_ref(pane.color))
        for detail in pane.detail:
            etree.SubElement(encodings, "lod", column=shelf_ref(detail))
    etree.SubElement(table, "rows").text = _shelf_expression(sheet.rows, sheet.dual_axis)
    etree.SubElement(table, "cols").text = _shelf_expression(sheet.cols, False)


def _dashboard(parent: etree._Element, board: reg.Dashboard) -> None:
    dashboard = etree.SubElement(parent, "dashboard", name=board.name)
    etree.SubElement(dashboard, "style")
    width, height = DASHBOARD_SIZE
    etree.SubElement(
        dashboard,
        "size",
        maxheight=str(height),
        maxwidth=str(width),
        minheight=str(height),
        minwidth=str(width),
    )
    zones = etree.SubElement(dashboard, "zones")
    root = etree.SubElement(
        zones,
        "zone",
        {"h": "100000", "id": "1", "type-v2": "layout-basic", "w": "100000"},
        x="0",
        y="0",
    )
    top_h = 60000
    slots = [(0, 0, 100000, top_h)]
    half = 50000
    slots += [(0, top_h, half, 100000 - top_h), (half, top_h, half, 100000 - top_h)]
    for i, (sheet, (x, y, w, h)) in enumerate(zip(board.worksheets, slots, strict=True), start=2):
        etree.SubElement(
            root,
            "zone",
            {"h": str(h), "id": str(i), "name": sheet, "w": str(w)},
            x=str(x),
            y=str(y),
        )


def build_workbook(directory: str = "data") -> bytes:
    """Return the workbook XML. ``directory`` is where the CSVs sit relative to the workbook."""
    root = etree.Element(
        "workbook",
        {
            "original-version": TWB_VERSION,
            "source-build": SOURCE_BUILD,
            "source-platform": "win",
            "version": TWB_VERSION,
        },
        nsmap={"user": _NAMESPACE},
    )
    etree.SubElement(root, "preferences")
    datasources = etree.SubElement(root, "datasources")
    _parameter_datasource(datasources)
    for source in reg.DATASOURCES:
        _datasource(datasources, source, directory)
    worksheets = etree.SubElement(root, "worksheets")
    for sheet in reg.WORKSHEETS:
        _worksheet(worksheets, sheet)
    dashboards = etree.SubElement(root, "dashboards")
    for board in reg.DASHBOARDS:
        _dashboard(dashboards, board)
    windows = etree.SubElement(root, "windows", {"source-height": "30"})
    for board in reg.DASHBOARDS:
        window = etree.SubElement(
            windows, "window", {"class": "dashboard", "maximized": "true", "name": board.name}
        )
        etree.SubElement(window, "viewpoints")
        etree.SubElement(window, "active", id="-1")
    for sheet in reg.WORKSHEETS:
        window = etree.SubElement(windows, "window", {"class": "worksheet", "name": sheet.name})
        etree.SubElement(window, "cards")
    return etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="utf-8")


def write_workbook(path: str | Path, directory: str = "data") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_workbook(directory))
    return path
