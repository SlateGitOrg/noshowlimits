"""A small evaluator for the Tableau calculation subset the workbook uses.

Supported: field references ``[name]``, parameter references ``[Parameters].[name]``, numbers,
double-quoted strings, ``+ - * /``, comparisons (``= <> < <= > >=``), ``AND``/``OR``/``NOT``,
``NULL`` and the functions ``SQRT``, ``MAX``, ``MIN``, ``ABS`` and ``IIF``. It exists so the tests
can run each calculated field over the exported rows and compare it with the Python reference.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping

_TOKEN = re.compile(
    r"""\s*(?:
        (?P<param>\[Parameters\]\.\[[^\]]+\])
      | (?P<field>\[[^\]]+\])
      | (?P<string>"[^"]*")
      | (?P<number>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
      | (?P<op><>|>=|<=|[-+*/()<>=,])
      | (?P<word>[A-Za-z_]+)
    )""",
    re.VERBOSE,
)
_FUNCTIONS = {
    "SQRT": math.sqrt,
    "MAX": max,
    "MIN": min,
    "ABS": abs,
    "IIF": lambda cond, a, b: a if cond else b,
}
_WORDS = {"AND": "and", "OR": "or", "NOT": "not", "NULL": "None"}


class FormulaError(ValueError):
    pass


def _tokens(formula: str):
    pos = 0
    text = formula.rstrip()
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if not match or match.end() == pos:
            raise FormulaError(f"cannot parse {formula!r} at position {pos}")
        pos = match.end()
        yield match.lastgroup, match.group(match.lastgroup)


def references(formula: str) -> list[str]:
    """Field names (without brackets) referenced by ``formula``; parameters are excluded."""
    return [tok[1:-1] for kind, tok in _tokens(formula) if kind == "field"]


def parameter_references(formula: str) -> list[str]:
    return [tok.split(".", 1)[1][1:-1] for kind, tok in _tokens(formula) if kind == "param"]


def to_python(formula: str) -> str:
    parts = []
    for kind, tok in _tokens(formula):
        if kind == "param":
            parts.append(f"_p({tok.split('.', 1)[1][1:-1]!r})")
        elif kind == "field":
            parts.append(f"_f({tok[1:-1]!r})")
        elif kind == "op":
            parts.append({"=": "==", "<>": "!="}.get(tok, tok))
        elif kind == "word":
            upper = tok.upper()
            if upper in _FUNCTIONS:
                parts.append(f"_fn_{upper}")
            elif upper in _WORDS:
                parts.append(_WORDS[upper])
            else:
                raise FormulaError(f"unsupported word {tok!r} in {formula!r}")
        else:
            parts.append(tok)
    return " ".join(parts)


def evaluate(
    formula: str,
    resolve_field: Callable[[str], object],
    parameters: Mapping[str, object] | None = None,
):
    """Evaluate ``formula``; ``resolve_field`` supplies the value of each ``[name]``."""
    namespace: dict[str, object] = {"__builtins__": {}}
    namespace.update({f"_fn_{name}": fn for name, fn in _FUNCTIONS.items()})
    namespace["_f"] = resolve_field
    namespace["_p"] = lambda name: (parameters or {})[name]
    return eval(to_python(formula), namespace)
