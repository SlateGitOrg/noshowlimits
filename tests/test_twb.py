from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from lxml import etree

from noshowlimits.tableau import build, validate
from noshowlimits.tableau import registry as reg

ROOT = Path(__file__).resolve().parents[1]
TWB = ROOT / "tableau" / "NoShowLimits.twb"
DATA = ROOT / "tableau" / "data"


@pytest.fixture()
def workdir(tmp_path):
    """A writable copy of the workbook and its CSVs, laid out as in the repository."""
    (tmp_path / "data").mkdir()
    for name in ("clinic_funnel.csv", "funnel_limits.csv", "funnel_plot.csv"):
        shutil.copy(DATA / name, tmp_path / "data" / name)
    shutil.copy(TWB, tmp_path / "NoShowLimits.twb")
    return tmp_path


def _edit(path: Path, transform) -> None:
    tree = etree.parse(str(path))
    transform(tree.getroot())
    tree.write(str(path), xml_declaration=True, encoding="utf-8")


def test_committed_workbook_is_valid():
    assert validate.validate_twb(TWB) == []


def test_committed_workbook_is_exactly_what_the_registry_builds():
    assert TWB.read_bytes() == build.build_workbook("data")


def test_registry_names_the_required_fields_and_sheets():
    captions = [c.caption for c in reg.CLINIC_FUNNEL.calcs]
    assert captions == [
        "O/E Ratio",
        "Risk-adjusted Rate",
        "Upper 95",
        "Upper 99.8",
        "Lower 95",
        "Lower 99.8",
        "Funnel Status",
    ]
    assert len(reg.DATASOURCES) == 2
    assert [w.name for w in reg.WORKSHEETS] == [
        "Funnel Plot",
        "Raw Rate League Table",
        "Exclusions Summary",
    ]
    assert len(reg.DASHBOARDS) == 1


def _funnel_sheet(root):
    return root.find("./worksheets/worksheet[@name='Funnel Plot']")


def test_workbook_has_two_textscan_datasources(workdir):
    root = etree.parse(str(workdir / "NoShowLimits.twb")).getroot()
    textscan = root.findall(".//named-connection/connection[@class='textscan']")
    assert sorted(c.get("filename") for c in textscan) == ["clinic_funnel.csv", "funnel_plot.csv"]
    assert len(root.findall("./dashboards/dashboard")) == 1


def test_funnel_plot_is_a_synchronized_dual_axis_over_one_datasource(workdir):
    sheet = _funnel_sheet(etree.parse(str(workdir / "NoShowLimits.twb")).getroot())
    sources = [d.get("name") for d in sheet.findall("./table/view/datasources/datasource")]
    assert [s for s in sources if s != "Parameters"] == ["federated.funnel_plot"]
    rows = sheet.findtext("./table/rows")
    refs = rows.strip("()").split(" + ")
    assert len(refs) == 2 and all(r.startswith("[federated.funnel_plot].") for r in refs)
    panes = sheet.findall("./table/panes/pane")
    assert [p.find("mark").get("class") for p in panes] == ["Line", "Circle"]
    assert [p.get("y-axis-name") for p in panes] == refs
    sync = sheet.find("./table/style/style-rule[@element='axis']/encoding")
    assert sync.get("synchronized") == "true" and sync.get("field") == refs[1]
    assert sheet.find("./table/view/filter/groupfilter").get("member") == "true"


def _funnel_problems(workdir, transform):
    _edit(workdir / "NoShowLimits.twb", lambda root: transform(_funnel_sheet(root)))
    return "\n".join(validate.validate_twb(workdir / "NoShowLimits.twb"))


def test_old_stacked_five_measure_form_is_rejected(workdir):
    """The first version stacked five measures on Rows over two datasources and two panes."""

    def stack(sheet):
        old = "[federated.clinic_funnel].[none:O/E Ratio:qk]"
        sheet.find("./table/rows").text = (
            "("
            + " + ".join(
                f"[federated.funnel_plot].[none:{n}:qk]"
                for n in ("Limit Value", "Limit Value", "Limit Value", "Limit Value")
            )
            + " + "
            + old
            + ")"
        )
        for pane in sheet.findall("./table/panes/pane"):
            del pane.attrib["y-axis-name"]
        extra = etree.SubElement(sheet.find("./table/view/datasources"), "datasource")
        extra.set("name", "federated.clinic_funnel")

    problems = _funnel_problems(workdir, stack)
    assert "must draw from the single datasource" in problems
    assert "stacks 5 fields on rows" in problems
    assert "must place exactly two measures on rows as a dual axis" in problems


def test_unsynchronized_dual_axis_is_rejected(workdir):
    def unsync(sheet):
        rule = sheet.find("./table/style/style-rule")
        rule.getparent().remove(rule)

    assert "the dual axis is not synchronized" in _funnel_problems(workdir, unsync)


def test_pane_not_bound_to_its_axis_is_rejected(workdir):
    def unbind(sheet):
        del sheet.findall("./table/panes/pane")[1].attrib["y-axis-name"]

    assert "is not bound to axis" in _funnel_problems(workdir, unbind)


def test_wrong_mark_type_in_a_pane_is_rejected(workdir):
    def to_line(sheet):
        sheet.findall("./table/panes/pane/mark")[1].set("class", "Line")

    assert "has mark types ['Line', 'Line'], expected ['Line', 'Circle']" in _funnel_problems(
        workdir, to_line
    )


def test_single_pane_for_a_dual_axis_is_rejected(workdir):
    def drop_pane(sheet):
        pane = sheet.findall("./table/panes/pane")[1]
        pane.getparent().remove(pane)

    assert "declares 1 panes, a dual axis needs 2" in _funnel_problems(workdir, drop_pane)


def test_missing_method_filter_is_rejected(workdir):
    def drop_filter(sheet):
        node = sheet.find("./table/view/filter")
        node.getparent().remove(node)

    assert "is missing the filter" in _funnel_problems(workdir, drop_filter)


def test_fresh_build_validates_against_freshly_exported_csvs(tmp_path, analysis):
    from noshowlimits import export

    export.write_exports(analysis.funnel, tmp_path / "out")
    build.write_workbook(tmp_path / "wb.twb", directory="out")
    assert validate.validate_twb(tmp_path / "wb.twb") == []


def test_renamed_csv_column_fails_and_the_message_names_it(workdir):
    path = workdir / "data" / "clinic_funnel.csv"
    text = path.read_text()
    header, rest = text.split("\n", 1)
    header = header.replace(",expected,", ",expected_noshows,")
    path.write_text(header + "\n" + rest)
    problems = validate.validate_twb(workdir / "NoShowLimits.twb")
    joined = "\n".join(problems)
    assert "'expected'" in joined and "not in clinic_funnel.csv" in joined
    assert "'expected_noshows'" in joined
    assert "calculated field 'O/E Ratio' uses unknown field 'expected'" in joined


def test_renamed_funnel_plot_column_fails(workdir):
    path = workdir / "data" / "funnel_plot.csv"
    header, rest = path.read_text().split("\n", 1)
    path.write_text(header.replace(",value,", ",val,") + "\n" + rest)
    problems = validate.validate_twb(workdir / "NoShowLimits.twb")
    assert any("column 'value' is declared but is not in funnel_plot.csv" in p for p in problems)
    assert any("calculated field 'Limit Value' uses unknown field 'value'" in p for p in problems)


def test_calculated_field_pointing_at_a_missing_column_fails(workdir):
    def rename_reference(root):
        for node in root.iter("calculation"):
            formula = node.get("formula")
            if formula == "[observed] / [expected]":
                node.set("formula", "[observed] / [expected_count]")

    _edit(workdir / "NoShowLimits.twb", rename_reference)
    problems = validate.validate_twb(workdir / "NoShowLimits.twb")
    assert any("'expected_count'" in p for p in problems)
    assert any("formula of 'O/E Ratio' differs from the registry" in p for p in problems)


def test_caption_that_differs_from_the_registry_fails(workdir):
    def rename_caption(root):
        for node in root.iter("column"):
            if node.get("caption") == "Upper 99.8":
                node.set("caption", "Upper 99.9")

    _edit(workdir / "NoShowLimits.twb", rename_caption)
    problems = validate.validate_twb(workdir / "NoShowLimits.twb")
    assert any("calculated field 'Upper 99.9' is not in the registry" in p for p in problems)
    assert any("calculated field 'Upper 99.8' is missing from the workbook" in p for p in problems)


def test_worksheet_field_that_does_not_exist_fails(workdir):
    def break_rows(root):
        for node in root.iter("rows"):
            node.text = node.text.replace("Clinic O/E", "Clinic O/Ex")

    _edit(workdir / "NoShowLimits.twb", break_rows)
    problems = validate.validate_twb(workdir / "NoShowLimits.twb")
    assert any("worksheet 'Funnel Plot' uses field 'Clinic O/Ex'" in p for p in problems)


def test_missing_worksheet_and_dangling_dashboard_zone_fail(workdir):
    def drop_sheet(root):
        sheet = root.find("./worksheets/worksheet[@name='Exclusions Summary']")
        sheet.getparent().remove(sheet)
        for zone in root.iter("zone"):
            if zone.get("name") == "Raw Rate League Table":
                zone.set("name", "Gone")

    _edit(workdir / "NoShowLimits.twb", drop_sheet)
    problems = "\n".join(validate.validate_twb(workdir / "NoShowLimits.twb"))
    assert "worksheet 'Exclusions Summary' is missing" in problems
    assert "dashboard 'No-show Limits' shows unknown worksheet 'Gone'" in problems


def test_missing_csv_and_non_workbook_inputs_are_reported(workdir, tmp_path):
    (workdir / "data" / "funnel_plot.csv").unlink()
    problems = validate.validate_twb(workdir / "NoShowLimits.twb")
    assert any("funnel_plot.csv" in p and "does not exist" in p for p in problems)
    other = tmp_path / "other.xml"
    other.write_text("<notaworkbook/>")
    assert "not a Tableau workbook" in validate.validate_twb(other)[0]
    assert "cannot read workbook" in validate.validate_twb(tmp_path / "nothing.twb")[0]


def test_data_dir_override_is_used(workdir, tmp_path):
    moved = tmp_path / "elsewhere"
    shutil.copytree(workdir / "data", moved)
    shutil.rmtree(workdir / "data")
    assert validate.validate_twb(workdir / "NoShowLimits.twb", data_dir=moved) == []
