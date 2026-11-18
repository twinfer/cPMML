#!/usr/bin/env python3
"""
gen.py -- seeded, type-directed generator of fuzz cases for the differential
fuzzer (docs/design/typed-core.md §8.3). Standard library only.

A case is one PMML document with one expression under test, placed in one of
five contexts, plus eight input rows:

  ctx1  a TransformationDictionary DerivedField x of a random dataType, read
        by a string and a typed transformedValue OutputField;
  ctx2  the same in the model's LocalTransformations, x referring forward to
        a DerivedField w declared after it;
  ctx3  a TreeModel whose SimplePredicates compare x (and the integer field i)
        with typed and decimal-on-integer document text (K1);
  ctx4  segments: (a) modelChain with x in the TransformationDictionary read
        by segment OutputFields, an outer OutputField reading the chain's
        output (B15); (b) selectFirst with x in the outer LocalTransformations
        read through a child MiningSchema (B14); (c) two sum segments each
        declaring a LocalTransformations field z of the same name (B16);
  ctx5  DataDictionary Values/Intervals/missing markers and MiningFields with
        random missing/invalid/outlier treatments under the expression.

The expression is built by gen(type, depth): a FieldRef or Constant leaf
(typed, or untyped with edge text), an Apply of a function whose result rule
can give the type (the hand-written FUNCTIONS table below; P11 replaces it by
the table exported from cPMML), with optional mapMissingTo, defaultValue and
invalidValueTreatment, or a NormContinuous, NormDiscrete, Discretize or
MapValues element (with and without dataType). About 10% of the productions
are ill-typed on purpose, to exercise load errors.

Every case is reproducible on its own: case N of seed S is drawn from
random.Random("S:N") and is the same whether generated alone (--case N) or in
a batch.

Inputs: eight rows (§8.3): rows 0-3 plain values, rows 4-6 per-type edge
values in the fields the case references, row 7 edge values in every field;
any cell missing (N/A) with probability 0.12; ctx5 adds missing markers.

Output (--out DIR):
  DIR/cases/<id>/model.pmml, DIR/cases/<id>/inputs.csv
  DIR/cases.jsonl   one JSON object per case: id, seed, case, context, the
                    expression as text, tags (used by departures.json), and
                    the input fields with their types

Usage:
  tools/fuzz/gen.py --seed S --cases N --out DIR
  tools/fuzz/gen.py --seed S --case K --out DIR
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

TYPES = ["string", "integer", "float", "double", "boolean"]
NUMERIC = ["integer", "float", "double"]
OPTYPE = {"string": "categorical", "boolean": "categorical", "integer": "continuous",
          "float": "continuous", "double": "continuous"}

# The input fields every case declares, one per type.
FIELDS = {"s": "string", "i": "integer", "f": "float", "d": "double", "b": "boolean"}

# Per-type input values (§8.3): plain values first, then edge values.
PLAIN = {
    "integer": ["3", "4", "-2", "0", "7"],
    "float": ["0.1", "2.5", "-1.25", "3"],
    "double": ["3.5", "2.0", "-1.25", "0.1", "7"],
    "string": ["A", "abc", "3", "B"],
    "boolean": ["true", "false"],
}
EDGE = {
    "integer": ["0", "-1", "2147483648", "9007199254740993", "1.0", ""],
    "float": ["0.1", "3.4e39", "1e-50", "-0", "NaN", "INF"],
    "double": ["0.1", "3.4e39", "1e-50", "-0", "NaN", "INF"],
    "string": ["", " a", "a ", "A", "unseen", "3", "3.0", "true"],
    "boolean": ["true", "false", "1", "0", "TRUE"],
}

# Constant text: typed edge text per type, and the untyped edge text (§8.3).
TYPED_CONSTANT = {
    "integer": ["0", "1", "2", "3", "-1", "10", "007", "2147483648", "1e10", "2.5"],
    "float": ["0.1", "2.5", "-0", "3.4e39", "1e-50", "NaN", "INF", "1"],
    "double": ["0.1", "1.5", "2.0", "-0", "3.5", "1e10", "3.14E-5", "NaN", "INF", "-INF", "0"],
    "string": ["A", "abc", "3", "3.0", "", " a", "true", "ABC"],
    "boolean": ["true", "false", "1", "0", "TRUE"],
}
UNTYPED_CONSTANT = ["7", "007", "-0", "1e10", "3.14E-5", "NaN", "inf", "", "abc", " a", "A", "3", "true", "2.5"]

# The functions under test (hand-written for P0; P11 reads cPMML's table).
# name: (minimum arguments, maximum arguments, parameter classes, result rule)
# A parameter class is a list of types; "*" repeats the last class. The
# result rule maps the argument types to the result type, or None.


def promote(types):
    nums = [t for t in types if t in NUMERIC]
    if len(nums) != len(types) or not nums:
        return None
    for t in ("double", "float", "integer"):
        if t in nums:
            return t


def promote_real(types):
    t = promote(types)
    return None if t is None else ("double" if t == "integer" else t)


def always(result):
    return lambda types: result


def same_as(index):
    return lambda types: types[index] if len(types) > index else None


def to_integer(types):
    return "integer" if promote(types) else None


def double_only(types):
    return "double" if promote(types) else None


N, S, B, A = NUMERIC, ["string"], ["boolean"], TYPES
FUNCTIONS = {
    "+": (2, 2, [N, N], promote), "-": (2, 2, [N, N], promote), "*": (2, 2, [N, N], promote),
    "/": (2, 2, [N, N], promote_real),
    "min": (1, 4, [N, "*"], promote), "max": (1, 4, [N, "*"], promote), "sum": (1, 4, [N, "*"], promote),
    "avg": (1, 4, [N, "*"], promote_real), "product": (1, 4, [N, "*"], promote),
    "median": (1, 4, [N, "*"], promote_real),
    "log10": (1, 1, [N], promote_real), "ln": (1, 1, [N], promote_real), "sqrt": (1, 1, [N], promote_real),
    "exp": (1, 1, [N], promote_real), "abs": (1, 1, [N], promote), "pow": (2, 2, [N, N], promote),
    "threshold": (2, 2, [N, N], always("integer")), "floor": (1, 1, [N], to_integer),
    "ceil": (1, 1, [N], to_integer), "round": (1, 1, [N], to_integer), "modulo": (2, 2, [N, N], promote),
    "rint": (1, 1, [N], double_only), "expm1": (1, 1, [N], double_only), "ln1p": (1, 1, [N], double_only),
    "hypot": (2, 2, [N, N], double_only), "sin": (1, 1, [N], double_only),
    "stdNormalCDF": (1, 1, [N], double_only), "erf": (1, 1, [N], double_only),
    "isMissing": (1, 1, [A], always("boolean")), "isNotMissing": (1, 1, [A], always("boolean")),
    "isValid": (1, 1, [A], always("boolean")), "isNotValid": (1, 1, [A], always("boolean")),
    "equal": (2, 2, [A, A], always("boolean")), "notEqual": (2, 2, [A, A], always("boolean")),
    "lessThan": (2, 2, [A, A], always("boolean")), "lessOrEqual": (2, 2, [A, A], always("boolean")),
    "greaterThan": (2, 2, [A, A], always("boolean")), "greaterOrEqual": (2, 2, [A, A], always("boolean")),
    "and": (2, 3, [B, "*"], always("boolean")), "or": (2, 3, [B, "*"], always("boolean")),
    "not": (1, 1, [B], always("boolean")),
    "isIn": (2, 4, [A, "*"], always("boolean")), "isNotIn": (2, 4, [A, "*"], always("boolean")),
    "if": (2, 3, [B, A, A], lambda ts: ts[1] if len(ts) == 2 or ts[1] == ts[2] else promote(ts[1:])),
    "uppercase": (1, 1, [S], always("string")), "lowercase": (1, 1, [S], always("string")),
    "trimBlanks": (1, 1, [S], always("string")), "substring": (3, 3, [S, ["integer"], ["integer"]], always("string")),
    "concat": (2, 3, [A, "*"], always("string")), "replace": (3, 3, [S, S, S], always("string")),
    "matches": (2, 2, [S, S], always("boolean")), "formatNumber": (2, 2, [N, S], always("string")),
}
# Functions whose leading argument types are tied (comparisons, isIn values).
TIED = {"equal", "notEqual", "lessThan", "lessOrEqual", "greaterThan", "greaterOrEqual", "isIn", "isNotIn", "if"}
FORMAT_PATTERNS = ["%3d", "%.2f", "%s", "%5.1f", "%d"]
REGEX_PATTERNS = ["^A", "a$", "[0-9]+", ".*", "b"]


class Ctx:
    """What an expression may reference, and the tags it collects."""

    def __init__(self, rng: random.Random, fields: dict):
        self.rng = rng
        self.fields = dict(fields)
        self.tags: set = set()
        self.ill = False


def attr(name, value):
    return f" {name}={quoteattr(value)}"


def constant(ctx: Ctx, t: str, untyped_ok=True):
    rng = ctx.rng
    if untyped_ok and rng.random() < 0.3:
        text = rng.choice(UNTYPED_CONSTANT)
        ctx.tags.add("const-untyped")
        ctx.tags.add("const:" + text)
        return f"<Constant>{escape(text)}</Constant>", repr(text)
    text = rng.choice(TYPED_CONSTANT[t])
    ctx.tags.add(f"const:{t}:{text}")
    return f'<Constant dataType="{t}">{escape(text)}</Constant>', f"{t}({text!r})"


def fieldref(ctx: Ctx, t: str):
    names = [n for n, ft in ctx.fields.items() if ft == t]
    if not names:
        return None
    name = ctx.rng.choice(names)
    ctx.tags.add("ref:" + name)
    return f'<FieldRef field="{name}"/>', name


def leaf(ctx: Ctx, t: str):
    if ctx.rng.random() < 0.6:
        got = fieldref(ctx, t)
        if got:
            return got
    return constant(ctx, t)


def missing_text(ctx: Ctx, t: str):
    return ctx.rng.choice(TYPED_CONSTANT[t] + ["M", "-999"])


def apply_attrs(ctx: Ctx, t: str):
    rng = ctx.rng
    out, text = "", []
    if rng.random() < 0.25:
        v = missing_text(ctx, t)
        out += attr("mapMissingTo", v)
        text.append(f"mapMissingTo={v!r}")
        ctx.tags.add("mapMissingTo")
    if rng.random() < 0.2:
        v = missing_text(ctx, t)
        out += attr("defaultValue", v)
        text.append(f"defaultValue={v!r}")
        ctx.tags.add("defaultValue")
    if rng.random() < 0.35:
        v = rng.choice(["returnInvalid", "asMissing", "asIs"])
        out += attr("invalidValueTreatment", v)
        text.append(f"ivt={v}")
        ctx.tags.add("ivt:" + v)
    return out, text


def gen_apply(ctx: Ctx, t: str, depth: int):
    rng = ctx.rng
    names = list(FUNCTIONS)
    ill = rng.random() < 0.1
    for _ in range(40):
        name = rng.choice(names)
        lo, hi, classes, rule = FUNCTIONS[name]
        n = rng.randint(lo, hi)
        if ill:
            types = [rng.choice(TYPES) for _ in range(n)]
        else:
            types = []
            for k in range(n):
                cls = classes[k] if k < len(classes) and classes[k] != "*" else (
                    classes[-2] if classes[-1] == "*" else classes[-1])
                if name in TIED and k >= (2 if name == "if" else 1) and types:
                    tied = types[1] if name == "if" else types[0]
                    if tied in cls and rng.random() < 0.8:
                        types.append(tied)
                        continue
                types.append(rng.choice(cls))
        result = rule(types)
        if ill or result == t:
            break
    else:
        return None
    if ill:
        ctx.ill = True
        ctx.tags.add("ill-typed")
    ctx.tags.add("fn:" + name)
    ctx.tags.add(f"sig:{name}({','.join(types)})")
    args_xml, args_text = [], []
    for k, at in enumerate(types):
        if name in ("isIn", "isNotIn") and k >= 1:
            x, txt = constant(ctx, at)
        elif name == "formatNumber" and k == 1:
            p = rng.choice(FORMAT_PATTERNS)
            x, txt = f'<Constant dataType="string">{escape(p)}</Constant>', repr(p)
        elif name in ("matches", "replace") and k == 1:
            p = rng.choice(REGEX_PATTERNS)
            x, txt = f'<Constant dataType="string">{escape(p)}</Constant>', repr(p)
        else:
            x, txt = gen(ctx, at, depth - 1)
        args_xml.append(x)
        args_text.append(txt)
    attrs, attrs_text = apply_attrs(ctx, t)
    xml = f'<Apply function={quoteattr(name)}{attrs}>' + "".join(args_xml) + "</Apply>"
    text = f"{name}({', '.join(args_text)}{''.join(', ' + a for a in attrs_text)})"
    return xml, text


def numeric_field(ctx: Ctx):
    names = [n for n, ft in ctx.fields.items() if ft in NUMERIC]
    return ctx.rng.choice(names)


def gen_norm_continuous(ctx: Ctx, t: str):
    rng = ctx.rng
    name = numeric_field(ctx)
    ctx.tags.add("ref:" + name)
    pts = sorted(rng.sample([-2, -1, 0, 0.5, 1, 2, 3, 5, 10], rng.randint(2, 3)))
    norms = [round(rng.uniform(-1, 1), 2) for _ in pts]
    a = ""
    if rng.random() < 0.4:
        a += attr("outliers", rng.choice(["asIs", "asMissingValues", "asExtremeValues"]))
    if rng.random() < 0.3:
        a += attr("mapMissingTo", rng.choice(["0", "-1", "0.5"]))
    ctx.tags.add("el:NormContinuous")
    body = "".join(f'<LinearNorm orig="{o}" norm="{m}"/>' for o, m in zip(pts, norms))
    return f'<NormContinuous field="{name}"{a}>{body}</NormContinuous>', f"NormContinuous({name}, {pts}->{norms}{a})"


def gen_norm_discrete(ctx: Ctx, t: str):
    rng = ctx.rng
    name = rng.choice(list(ctx.fields))
    ctx.tags.add("ref:" + name)
    ft = ctx.fields[name]
    value = rng.choice(PLAIN[ft] + TYPED_CONSTANT[ft][:3])
    a = attr("mapMissingTo", "0") if rng.random() < 0.3 else ""
    ctx.tags.add("el:NormDiscrete")
    ctx.tags.add(f"nd:{ft}")
    return (f'<NormDiscrete field="{name}" value={quoteattr(value)}{a}/>',
            f"NormDiscrete({name}, {value!r}{a})")


def gen_discretize(ctx: Ctx, t: str):
    rng = ctx.rng
    name = numeric_field(ctx)
    ctx.tags.add("ref:" + name)
    cuts = sorted(rng.sample([-1, 0, 1, 2, 2.5, 3, 5], 2))
    bins = [("a1", None, cuts[0]), ("1.50", cuts[0], cuts[1]), ("3", cuts[1], None)]
    if t in NUMERIC:
        bins = [(rng.choice(["1", "1.50", "2.0", "-3"]), lo, hi) for _, lo, hi in bins]
    elif t == "boolean":
        bins = [(rng.choice(["true", "false"]), lo, hi) for _, lo, hi in bins]
    a = ""
    if rng.random() < 0.5:
        a += attr("dataType", t)
        ctx.tags.add("disc-typed")
    else:
        ctx.tags.add("disc-untyped")
    if rng.random() < 0.3:
        a += attr("mapMissingTo", bins[0][0])
    if rng.random() < 0.3:
        a += attr("defaultValue", bins[-1][0])
    body = ""
    for value, lo, hi in bins:
        interval = '<Interval closure="closedOpen"'
        if lo is not None:
            interval += f' leftMargin="{lo}"'
        if hi is not None:
            interval += f' rightMargin="{hi}"'
        body += f'<DiscretizeBin binValue={quoteattr(value)}>{interval}/></DiscretizeBin>'
    ctx.tags.add("el:Discretize")
    return f'<Discretize field="{name}"{a}>{body}</Discretize>', f"Discretize({name}, {bins}{a})"


def gen_map_values(ctx: Ctx, t: str):
    rng = ctx.rng
    name = rng.choice(list(ctx.fields))
    ctx.tags.add("ref:" + name)
    ft = ctx.fields[name]
    keys = rng.sample(PLAIN[ft] + EDGE[ft], min(3, len(set(PLAIN[ft] + EDGE[ft]))))
    keys = list(dict.fromkeys(keys))
    outs = [rng.choice(TYPED_CONSTANT[t]) for _ in keys]
    a = ""
    if rng.random() < 0.5:
        a += attr("dataType", t)
        ctx.tags.add("mv-typed")
    else:
        ctx.tags.add("mv-untyped")
    if rng.random() < 0.3:
        a += attr("mapMissingTo", rng.choice(TYPED_CONSTANT[t]))
    if rng.random() < 0.3:
        a += attr("defaultValue", rng.choice(TYPED_CONSTANT[t]))
    rows = "".join(f"<row><k>{escape(k)}</k><v>{escape(v)}</v></row>" for k, v in zip(keys, outs))
    ctx.tags.add("el:MapValues")
    ctx.tags.add(f"mv-key:{ft}")
    return (f'<MapValues outputColumn="v"{a}><FieldColumnPair field="{name}" column="k"/>'
            f"<InlineTable>{rows}</InlineTable></MapValues>",
            f"MapValues({name}, {dict(zip(keys, outs))}{a})")


def gen(ctx: Ctx, t: str, depth: int):
    rng = ctx.rng
    if depth <= 0 or rng.random() < 0.25:
        return leaf(ctx, t)
    roll = rng.random()
    if roll < 0.12 and t == "double":
        return gen_norm_continuous(ctx, t)
    if roll < 0.2 and t == "double":
        return gen_norm_discrete(ctx, t)
    if roll < 0.28:
        return gen_discretize(ctx, t)
    if roll < 0.36:
        return gen_map_values(ctx, t)
    got = gen_apply(ctx, t, depth)
    return got if got else leaf(ctx, t)


# ---- documents --------------------------------------------------------------

def data_dictionary(extra="", values=None, target='<DataField name="y" optype="continuous" dataType="double"/>'):
    values = values or {}
    out = "<DataDictionary>"
    for name, t in FIELDS.items():
        out += f'<DataField name="{name}" optype="{OPTYPE[t]}" dataType="{t}">{values.get(name, "")}</DataField>'
    return out + target + extra + "</DataDictionary>"


def mining_schema(fields=FIELDS, target="y", attrs=None, extra=()):
    attrs = attrs or {}
    out = "<MiningSchema>"
    for name in list(fields) + list(extra):
        out += f'<MiningField name="{name}"{attrs.get(name, "")}/>'
    if target:
        out += f'<MiningField name="{target}" usageType="target"/>'
    return out + "</MiningSchema>"


def outputs(field: str, t: str, rng: random.Random, prefix="o"):
    typed = rng.choice([t, t, rng.choice(TYPES)])
    return (f'<OutputField name="{prefix}_s" optype="categorical" dataType="string" feature="transformedValue">'
            f'<FieldRef field="{field}"/></OutputField>'
            f'<OutputField name="{prefix}_t" optype="{OPTYPE[typed]}" dataType="{typed}" feature="transformedValue">'
            f'<FieldRef field="{field}"/></OutputField>')


def derived(name, t, body):
    return f'<DerivedField name="{name}" optype="{OPTYPE[t]}" dataType="{t}">{body}</DerivedField>'


def document(dd, body, td=""):
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/>'
            + dd + (f"<TransformationDictionary>{td}</TransformationDictionary>" if td else "") + body + "</PMML>\n")


def regression(schema, out="", lt="", table='<RegressionTable intercept="0"/>', extra_attrs=""):
    return (f'<RegressionModel functionName="regression"{extra_attrs}>{schema}'
            + (f"<Output>{out}</Output>" if out else "")
            + (f"<LocalTransformations>{lt}</LocalTransformations>" if lt else "")
            + f"{table}</RegressionModel>")


def ctx1(rng, ctx):
    t = rng.choice(TYPES)
    xml, text = gen(ctx, t, rng.randint(1, 3))
    return document(data_dictionary(), regression(mining_schema(), outputs("x", t, rng)), derived("x", t, xml)), t, text


def ctx2(rng, ctx):
    t = rng.choice(TYPES)
    wt = rng.choice(TYPES)
    src = rng.choice([n for n, ft in FIELDS.items() if ft == wt])
    ctx.fields["w"] = wt
    # x must read w, which is declared after it: draw until it does, else
    # x is w itself.
    for _ in range(10):
        trial = Ctx(rng, ctx.fields)
        xml, text = gen(trial, t, rng.randint(1, 3))
        if "ref:w" in trial.tags:
            break
    else:
        trial = Ctx(rng, ctx.fields)
        trial.tags.add("ref:w")
        xml, text = '<FieldRef field="w"/>', "w"
    ctx.tags |= trial.tags
    ctx.ill = trial.ill
    lt =derived("x", t, xml) + derived("w", wt, f'<FieldRef field="{src}"/>')
    ctx.tags.add("forward-ref")
    ctx.tags.add("ref:" + src)
    return document(data_dictionary(), regression(mining_schema(), outputs("x", t, rng), lt)), t, text + f"; w={src}"


def ctx3(rng, ctx):
    t = rng.choice(["integer", "double", "float", "string"])
    xml, text = gen(ctx, t, rng.randint(1, 2))
    if t == "string":
        vals = ["A", "abc", "3", "3.0"]
    elif t == "integer":
        vals = ["2", "2.5", "33.0", "0", "-1"]
    else:
        vals = ["2.5", "0.1", "2", "1e10"]
    op = rng.choice(["equal", "notEqual", "lessThan", "lessOrEqual", "greaterThan", "greaterOrEqual"])
    v = rng.choice(vals)
    iv = rng.choice(["2", "2.5", "3.0", "1e1"])
    iop = rng.choice(["lessOrEqual", "equal", "greaterThan"])
    ctx.tags.add("ref:i")
    ctx.tags.update({f"tree-op:{op}", f"tree-value:{t}:{v}", f"tree-i:{iop}:{iv}"})
    target = ('<DataField name="yc" optype="categorical" dataType="string">'
              '<Value value="a"/><Value value="b"/><Value value="c"/></DataField>')
    tree = ('<TreeModel functionName="classification" missingValueStrategy="none" noTrueChildStrategy="returnLastPrediction">'
            + mining_schema(target="yc")
            + "<Output>" + outputs("x", t, rng) + "</Output>"
            + '<Node score="c"><True/>'
            + f'<Node score="a"><SimplePredicate field="x" operator="{op}" value={quoteattr(v)}/></Node>'
            + f'<Node score="b"><SimplePredicate field="i" operator="{iop}" value="{iv}"/></Node>'
            + "</Node></TreeModel>")
    doc = document(data_dictionary(target=target), tree, derived("x", t, xml))
    return doc, t, text + f"; x {op} {v!r}; i {iop} {iv}"


def ctx4(rng, ctx):
    variant = rng.choice(["chain", "select", "sum"])
    ctx.tags.add("seg:" + variant)
    if variant == "sum":
        t = rng.choice(NUMERIC)
        xml, text = gen(ctx, t, rng.randint(1, 3))
        segs = ""
        for k in (1, 2):
            lt = derived("z", "double", f'<Apply function="+"><FieldRef field="x"/><Constant dataType="double">{k}</Constant></Apply>')
            segs += (f'<Segment id="{k}"><True/>'
                     + regression(mining_schema(), lt=lt,
                                  table='<RegressionTable intercept="0"><NumericPredictor name="z" coefficient="1"/></RegressionTable>')
                     + "</Segment>")
        body = ('<MiningModel functionName="regression">' + mining_schema()
                + "<Output>" + outputs("x", t, rng) + "</Output>"
                + f'<Segmentation multipleModelMethod="sum">{segs}</Segmentation></MiningModel>')
        return document(data_dictionary(), body, derived("x", t, xml)), t, text + "; sum of x+1, x+2"
    t = rng.choice(TYPES)
    xml, text = gen(ctx, t, rng.randint(1, 3))
    if variant == "chain":
        seg1 = regression(mining_schema(),
                          out='<OutputField name="c1" optype="categorical" dataType="string" feature="transformedValue"><FieldRef field="x"/></OutputField>')
        seg2 = regression(mining_schema(extra=["c1"]),
                          out='<OutputField name="c2" optype="categorical" dataType="string" feature="transformedValue"><FieldRef field="c1"/></OutputField>')
        body = ('<MiningModel functionName="regression">' + mining_schema()
                + '<Output><OutputField name="o_chain" optype="categorical" dataType="string" feature="transformedValue">'
                  '<FieldRef field="c1"/></OutputField></Output>'
                + '<Segmentation multipleModelMethod="modelChain">'
                + f'<Segment id="1"><True/>{seg1}</Segment><Segment id="2"><True/>{seg2}</Segment>'
                + "</Segmentation></MiningModel>")
        return document(data_dictionary(), body, derived("x", t, xml)), t, text + "; modelChain c1=x, c2=c1"
    seg_out = '<OutputField name="c1" optype="categorical" dataType="string" feature="transformedValue"><FieldRef field="x"/></OutputField>'
    seg1 = regression(mining_schema(extra=["x"]), out=seg_out)
    body = ('<MiningModel functionName="regression">' + mining_schema()
            + "<Output>" + outputs("x", t, rng) + "</Output>"
            + "<LocalTransformations>" + derived("x", t, xml) + "</LocalTransformations>"
            + f'<Segmentation multipleModelMethod="selectFirst"><Segment id="1"><True/>{seg1}</Segment></Segmentation>'
            + "</MiningModel>")
    return document(data_dictionary(), body), t, text + "; selectFirst, x in outer LocalTransformations"


def ctx5(rng, ctx):
    values, attrs = {}, {}
    for name, t in FIELDS.items():
        r = rng.random()
        if t == "string" and r < 0.5:
            values[name] = '<Value value="A"/><Value value="abc"/><Value value="3"/><Value value="?" property="missing"/>'
            ctx.tags.add(f"dd-values:{name}")
        elif t in NUMERIC and r < 0.3:
            lo, hi = rng.choice([(0, 10), (-1, 3), (2, 5)])
            closure = rng.choice(["closedClosed", "openOpen", "closedOpen"])
            values[name] = f'<Interval closure="{closure}" leftMargin="{lo}" rightMargin="{hi}"/>'
            ctx.tags.add(f"dd-interval:{name}")
        if t in NUMERIC and rng.random() < 0.3:
            values[name] = values.get(name, "") + '<Value value="-999" property="missing"/>'
            ctx.tags.add(f"dd-marker:{name}")
        a = ""
        if rng.random() < 0.3:
            a += attr("missingValueReplacement", rng.choice(PLAIN[t]))
            ctx.tags.add(f"mf-mvr:{name}")
        if rng.random() < 0.4:
            ivt = rng.choice(["returnInvalid", "asMissing", "asIs", "asValue"])
            a += attr("invalidValueTreatment", ivt)
            if ivt == "asValue":
                a += attr("invalidValueReplacement", rng.choice(PLAIN[t]))
            ctx.tags.add(f"mf-ivt:{name}:{ivt}")
        if t in NUMERIC and rng.random() < 0.3:
            treatment = rng.choice(["asMissingValues", "asExtremeValues"])
            a += attr("outliers", treatment)
            if rng.random() < 0.7:
                a += attr("lowValue", rng.choice(["-1", "0", "0.5"]))
            if rng.random() < 0.7:
                a += attr("highValue", rng.choice(["3", "4.5", "100"]))
            ctx.tags.add(f"mf-outliers:{name}:{treatment}")
        attrs[name] = a
    t = rng.choice(TYPES)
    xml, text = gen(ctx, t, rng.randint(0, 2))
    doc = document(data_dictionary(values=values), regression(mining_schema(attrs=attrs), outputs("x", t, rng)),
                   derived("x", t, xml))
    return doc, t, text


CONTEXTS = [ctx1, ctx2, ctx3, ctx4, ctx5]


def inputs_csv(rng: random.Random, marker_rows: bool, referenced: set) -> list:
    """Rows 0-3 plain values; rows 4-6 edge values in the fields the case
    references (plain elsewhere, so an edge value in a field the expression
    never reads does not mask the expression); row 7 edge values in every
    field. Any cell is missing with probability 0.12."""
    rows = []
    for r in range(8):
        row = []
        for name, t in FIELDS.items():
            if rng.random() < 0.12:
                row.append(None)
            elif r < 4 or (r < 7 and name not in referenced):
                row.append(rng.choice(PLAIN[t]))
            elif marker_rows and t in NUMERIC and rng.random() < 0.3:
                row.append(rng.choice(["-999", "-999.0", "?"]))
            elif marker_rows and t == "string" and rng.random() < 0.3:
                row.append("?")
            else:
                row.append(rng.choice(EDGE[t]))
        rows.append(row)
    return rows


def csv_cell(value):
    if value is None:
        return "N/A"
    if value == "" or any(c in value for c in ',"\n') or value == "N/A":
        return '"' + value.replace('"', '""') + '"'
    return value


def make_case(seed: int, k: int):
    rng = random.Random(f"{seed}:{k}")
    ci = rng.randrange(len(CONTEXTS))
    ctx = Ctx(rng, FIELDS)
    doc, t, text = CONTEXTS[ci](rng, ctx)
    ctx.tags.add(f"ctx:{ci + 1}")
    ctx.tags.add(f"x:{t}")
    referenced = {t[4:] for t in ctx.tags if t.startswith("ref:")}
    rows = inputs_csv(rng, ci == 4, referenced)
    meta = {"id": f"s{seed}_c{k:06d}", "seed": seed, "case": k, "context": ci + 1, "x_type": t,
            "expr": text, "tags": sorted(ctx.tags), "fields": FIELDS}
    return meta, doc, rows


def write_case(out: Path, meta, doc, rows):
    d = out / "cases" / meta["id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.pmml").write_text(doc)
    with open(d / "inputs.csv", "w", newline="") as f:
        f.write(",".join(FIELDS) + "\n")
        for row in rows:
            f.write(",".join(csv_cell(v) for v in row) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--cases", type=int, help="generate cases 0..N-1")
    group.add_argument("--case", type=int, help="generate case K only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ks = [args.case] if args.case is not None else range(args.cases)
    with open(out / "cases.jsonl", "w") as index:
        for k in ks:
            meta, doc, rows = make_case(args.seed, k)
            write_case(out, meta, doc, rows)
            index.write(json.dumps(meta) + "\n")
    print(f"gen.py: {len(ks)} cases in {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
