#!/usr/bin/env python3
"""
Generate oracle fixtures for HousingGBTClassifier and HousingGBTClassifier_PCA.

Model structure (modelChain):
  Segment 1: MiningModel regression sum-of-trees → decisionFunction(true)
             Targets: rescaleFactor * sum + rescaleConstant
             Output:  transformedDecisionFunction = sigmoid(decisionFunction)
  Segment 2: RegressionModel classification (normalizationMethod="none")
             probability(true) = transformedDecisionFunction
             probability(false) = 1 - probability(true)

Primary output column: probability(false)  (matches cPMML output_name())

Run:
    python test/generate_housing_gbt_fixtures.py
"""

from __future__ import annotations
import csv, math, re, struct, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET


def to_float32(x: float) -> float:
    """Apply float32 quantization (mirrors PMML dataType="float" truncation)."""
    return struct.unpack('f', struct.pack('f', x))[0]

REPO       = Path(__file__).parent.parent
MODEL_DIR  = REPO / "test" / "data" / "model"
DATASET_DIR = REPO / "test" / "data" / "dataset"


def get_ns(root: ET.Element) -> str:
    m = re.match(r"\{(.*?)\}", root.tag)
    return m.group(1) if m else ""


def q(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}" if ns else tag


# ---------------------------------------------------------------------------
# Expression evaluator (handles Constant, FieldRef, Apply)
# ---------------------------------------------------------------------------

def eval_expr(node: ET.Element, row: dict, ns: str) -> float:
    tag = node.tag.split("}")[-1] if "}" in node.tag else node.tag

    if tag == "Constant":
        return float(node.text)

    if tag == "FieldRef":
        v = row.get(node.get("field"))
        return float(v) if v is not None else 0.0

    if tag == "NormDiscrete":
        field = node.get("field")
        value = node.get("value")
        return 1.0 if str(row.get(field, "")) == value else 0.0

    if tag == "Apply":
        fn   = node.get("function")
        args = [eval_expr(c, row, ns) for c in node]
        if fn == "+":    return args[0] + args[1]
        if fn == "-":    return args[0] - args[1]
        if fn == "*":    return args[0] * args[1]
        if fn == "/":    return args[0] / args[1]
        if fn == "exp":  return math.exp(args[0])
        if fn == "sum":  return sum(args)
        raise ValueError(f"unsupported function: {fn}")

    raise ValueError(f"unsupported node: {tag}")


# ---------------------------------------------------------------------------
# Derived field computation
# ---------------------------------------------------------------------------

def compute_derived_fields(derived_field_nodes: list, row: dict, ns: str) -> None:
    for df in derived_field_nodes:
        name = df.get("name")
        datatype = df.get("dataType", "")
        children = list(df)
        if not children:
            continue
        try:
            val = eval_expr(children[0], row, ns)
            if datatype == "float":
                val = to_float32(val)
            row[name] = val
        except Exception:
            row[name] = 0.0


# ---------------------------------------------------------------------------
# Predicate evaluator
# ---------------------------------------------------------------------------

def eval_predicate(node: ET.Element, row: dict, ns: str) -> bool:
    tag = node.tag.split("}")[-1] if "}" in node.tag else node.tag

    if tag == "True":
        return True
    if tag == "False":
        return False

    if tag == "SimplePredicate":
        field = node.get("field")
        op    = node.get("operator")
        value = node.get("value")
        v = row.get(field)
        if v is None:
            return False
        try:
            fv, fval = float(v), float(value)
            if op == "lessOrEqual":    return fv <= fval
            if op == "lessThan":       return fv <  fval
            if op == "greaterOrEqual": return fv >= fval
            if op == "greaterThan":    return fv >  fval
            if op == "equal":          return fv == fval
            if op == "notEqual":       return fv != fval
        except (TypeError, ValueError):
            if op == "equal":    return str(v) == value
            if op == "notEqual": return str(v) != value
        return False

    if tag == "SimpleSetPredicate":
        field  = node.get("field")
        boolop = node.get("booleanOperator")
        arr    = node.find(q(ns, "Array"))
        values = set(arr.text.split()) if arr is not None and arr.text else set()
        v = str(row.get(field, ""))
        return (v in values) if boolop == "isIn" else (v not in values)

    if tag == "CompoundPredicate":
        boolop  = node.get("booleanOperator")
        results = [eval_predicate(c, row, ns) for c in node]
        if boolop == "and":  return all(results)
        if boolop == "or":   return any(results)
        if boolop == "surrogate":
            for r in results:
                if r is not None:
                    return r
    return False


# ---------------------------------------------------------------------------
# Tree scoring
# ---------------------------------------------------------------------------

def score_tree(root_node: ET.Element, row: dict, ns: str) -> float:
    current = root_node
    while True:
        children = current.findall(q(ns, "Node"))
        if not children:
            return float(current.get("score", 0.0))
        matched = False
        for child in children:
            # first child element is the predicate
            preds = [c for c in child
                     if (c.tag.split("}")[-1] if "}" in c.tag else c.tag)
                     in ("True","False","SimplePredicate","SimpleSetPredicate","CompoundPredicate")]
            if preds and eval_predicate(preds[0], row, ns):
                current = child
                matched = True
                break
        if not matched:
            return float(current.get("score", 0.0))


# ---------------------------------------------------------------------------
# Model scoring
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def score_model(zip_path: Path, input_row: dict) -> tuple[float, float]:
    """Return (probability_false, probability_true)."""
    with zipfile.ZipFile(zip_path) as z:
        with z.open(z.namelist()[0]) as f:
            root = ET.parse(f).getroot()

    ns = get_ns(root)

    outer = root.find(q(ns, "MiningModel"))
    outer_seg = outer.find(q(ns, "Segmentation"))
    segments  = outer_seg.findall(q(ns, "Segment"))

    # Segment 1: inner GBT regression (sum of trees)
    inner_model  = segments[0].find(q(ns, "MiningModel"))
    inner_seg    = inner_model.find(q(ns, "Segmentation"))
    tree_segs    = inner_seg.findall(q(ns, "Segment"))

    # LocalTransformations
    lt = inner_model.find(q(ns, "LocalTransformations"))
    derived_fields = lt.findall(q(ns, "DerivedField")) if lt is not None else []

    # Targets rescaling
    targets   = inner_model.find(q(ns, "Targets"))
    tgt_elem  = targets.find(q(ns, "Target")) if targets is not None else None
    rescale_f = float(tgt_elem.get("rescaleFactor",   "1.0")) if tgt_elem is not None else 1.0
    rescale_c = float(tgt_elem.get("rescaleConstant", "0.0")) if tgt_elem is not None else 0.0

    row = dict(input_row)
    compute_derived_fields(derived_fields, row, ns)

    total = 0.0
    for seg in tree_segs:
        tree_model = seg.find(q(ns, "TreeModel"))
        root_node  = tree_model.find(q(ns, "Node"))
        total += score_tree(root_node, row, ns)

    decision = rescale_f * total + rescale_c
    p_true   = sigmoid(decision)
    p_false  = 1.0 - p_true
    return p_false, p_true


# ---------------------------------------------------------------------------
# Fixture generation
# ---------------------------------------------------------------------------

def generate(model_name: str) -> None:
    zip_path = MODEL_DIR  / f"{model_name}.zip"
    csv_path = DATASET_DIR / f"{model_name}.csv"

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows   = list(reader)
        all_cols = list(reader.fieldnames)

    # Input columns = everything except legacy oracle columns
    skip = {"true", "false", "probability(false)", "probability(true)",
            "decisionFunction(true)", "median_house_value"}
    input_cols = [c for c in all_cols if c not in skip]

    print(f"Scoring {model_name} ({len(rows)} rows)...", flush=True)

    results = []
    for i, row in enumerate(rows):
        inp = {k: row[k] for k in input_cols}
        pf, pt = score_model(zip_path, inp)
        results.append((inp, pf, pt))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(input_cols + ["probability(false)", "probability(true)"])
        for inp, pf, pt in results:
            writer.writerow([inp[c] for c in input_cols] + [pf, pt])

    print(f"  -> {csv_path}")


if __name__ == "__main__":
    generate("HousingGBTClassifier")
    generate("HousingGBTClassifier_PCA")
    print("Done.")
