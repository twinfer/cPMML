"""
pmmltext.py -- text helpers shared by the tools that compare cPMML with JPMML
as exact text: tools/cross-validate-text.py (the fixture suite) and
tools/fuzz/compare.py (the differential fuzzer, docs/design/typed-core.md §8).

  as_number(text)       the number a cell spells, Java or Python style, or None
  same_number(a, b)     two cells read as numbers within 1e-12 relative
  classify(jpmml, cpmml)
                        FORMAT, ROUNDING or VALUE for two cells that differ
  read_csv(path)        a CSV file as a list of rows
  local(tag)            an ElementTree tag without its namespace
  model_info(path)      (functionName, target, DataFields, OutputFields)

JPMML_MISSING is the text JPMML's EvaluationExample writes, and reads, for a
missing value. Standard library only.
"""

from __future__ import annotations

import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path

JPMML_MISSING = "N/A"


def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def model_info(pmml_path: Path):
    """(functionName, target, set of DataField names, set of OutputField names)."""
    text = Path(pmml_path).read_bytes()
    if text.startswith(b"\xef\xbb\xbf"):
        text = text[3:]
    root = ET.fromstring(text.strip())
    data_fields = set()
    outputs = set()
    function, target = None, None
    for el in root.iter():
        name = local(el.tag)
        if name == "DataField":
            data_fields.add(el.get("name"))
        elif name == "OutputField":
            outputs.add(el.get("name"))
    for child in root:
        if child.get("functionName"):
            function = child.get("functionName")
            for schema in child:
                if local(schema.tag) != "MiningSchema":
                    continue
                for mf in schema:
                    if mf.get("usageType") in ("target", "predicted"):
                        target = mf.get("name")
                        break
            break
    return function, target, data_fields, outputs


def as_number(text: str):
    spelled = {"Infinity": math.inf, "-Infinity": -math.inf, "NaN": math.nan}
    if text in spelled:
        return spelled[text]
    try:
        return float(text)
    except ValueError:
        return None


def same_number(a: str, b: str) -> bool:
    x, y = as_number(a), as_number(b)
    if x is None or y is None:
        return False
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    if math.isinf(x) or math.isinf(y):
        return x == y
    return abs(x - y) <= 1e-12 * max(abs(x), abs(y), 1e-300)


def classify(jpmml: str, cpmml: str) -> str:
    """FORMAT: the same number written differently ("1" for "1.0", "1.500000"
    for "1.5"); ROUNDING: numbers within 1e-12 relative, the last bits of a
    computation done in a different order; VALUE: anything else, -0.0 against
    0.0 included."""
    x, y = as_number(jpmml), as_number(cpmml)
    if x is not None and y is not None:
        if (x == y and math.copysign(1, x) == math.copysign(1, y)) or (math.isnan(x) and math.isnan(y)):
            return "FORMAT"
        if same_number(jpmml, cpmml) and math.copysign(1, x) == math.copysign(1, y):
            return "ROUNDING"
    return "VALUE"


def read_csv(path: Path):
    with open(path, newline="") as f:
        return list(csv.reader(f))
