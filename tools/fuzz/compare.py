#!/usr/bin/env python3
"""
compare.py -- classify every cell of a fuzz run (docs/design/typed-core.md
§8.4) and match the disagreements against tools/fuzz/departures.json.

Per case (outputs written by tools/fuzz/run.py):
  LOAD_BOTH          neither loads the document (pass); also when cPMML
                     refuses it at load and JPMML loads it but fails every
                     row (both refuse it, JPMML only later)
  LOAD_CPMML_ONLY    cPMML refuses a document JPMML scores
  LOAD_JPMML_ONLY    JPMML refuses a document cPMML scores
  CRASH              fuzz_runner died on the case (always a finding)
Per cell of every column JPMML writes (a column cPMML lacks is MISSING_COLUMN):
  identical text     pass
  ERROR_BOTH         both fail the row (pass)
  ERROR_JPMML_ONLY, ERROR_CPMML_ONLY
  FORMAT, ROUNDING, VALUE
                     as in tools/cross-validate-text.py (tools/pmmltext.py);
                     ROUNDING is counted but is not a finding
A disagreement that a rule of departures.json matches is reported as
expected under that departure's id; every other one is a finding. Findings
are grouped -- errors by (class, JPMML's message, cPMML's message) with the
data elided, value differences by (class, context, x's type, column,
outermost element of the expression) -- and printed with one example each;
all of them are written to DIR/findings.jsonl.

Usage:
  tools/fuzz/compare.py --corpus DIR [--departures tools/fuzz/departures.json]
      [--examples N] [--json OUT]

Exit status: 0 when there is no finding, 1 otherwise.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pmmltext import JPMML_MISSING, classify, read_csv  # noqa: E402

# The string OutputField that carries x in each context (tools/fuzz/gen.py).
X_COLUMNS = ("o_s", "c1", "o_chain")
PASS = {"PASS", "ERROR_BOTH", "LOAD_BOTH", "ROUNDING"}


def is_error(cell: str) -> bool:
    return cell.startswith("ERROR: ")


def error_class(cell: str) -> str:
    """An error cell with its data elided, so one defect groups as one
    finding: quoted text becomes _ unless it names a Function or Field, a
    record dump {...} becomes {}, numbers become #."""
    body = cell.split(": ", 1)[1] if cell.startswith(("ERROR: ", "LOAD-ERROR: ", "CRASH: ")) else cell
    body = re.sub(r"\{[^}]*\}", "{}", body)
    body = re.sub(r'(Function |Field )?"([^"]*)"', lambda m: m.group(0) if m.group(1) else "_", body)
    body = re.sub(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", "#", body)
    return body[:100]


def head(expr: str) -> str:
    m = re.match(r"[^(;]*", expr)
    return (m.group(0) or expr).strip() or expr


def load_rules(path: Path):
    doc = json.loads(path.read_text())
    rules = []
    # "proposed" entries are departures measured by the fuzzer that §8.5 does
    # not list yet; they match like the others and are reported by their id.
    for dep in doc["departures"] + doc.get("proposed", []):
        for rule in dep.get("match", []):
            rules.append((dep["id"], rule))
    return doc, rules


def tag_match(patterns, tags) -> bool:
    return all(any(fnmatch.fnmatchcase(t, p) for t in tags) for p in patterns)


def rule_matches(rule, cell) -> bool:
    if "kinds" in rule and cell["kind"] not in rule["kinds"]:
        return False
    if "tags_all" in rule and not tag_match(rule["tags_all"], cell["tags"]):
        return False
    if "tags_any" in rule and not any(tag_match([p], cell["tags"]) for p in rule["tags_any"]):
        return False
    if "tags_none" in rule and any(tag_match([p], cell["tags"]) for p in rule["tags_none"]):
        return False
    if "cpmml_x" in rule and not re.search(rule["cpmml_x"], cell.get("xcell", "")):
        return False
    for side in ("jpmml", "cpmml", "column"):
        if side in rule and not re.search(rule[side], cell[side]):
            return False
    if "row_any" in rule and not any(re.fullmatch(rule["row_any"], v) for v in cell["inrow"].values()):
        return False
    if "unreferenced_field" in rule:
        m = re.search(rule["unreferenced_field"], cell["cpmml"])
        if not m or f"ref:{m.group(1)}" in cell["tags"]:
            return False
    for field, pattern in rule.get("row", {}).items():
        if not re.fullmatch(pattern, cell["inrow"].get(field, JPMML_MISSING)):
            return False
    return True


def compare_case(meta, corpus: Path):
    """Yields one dict per cell (or one per case, for load/crash)."""
    cid = meta["id"]
    jpath, cpath = corpus / "jpmml" / f"{cid}.csv", corpus / "cpmml" / f"{cid}.csv"
    base = {"case": cid, "context": meta["context"], "expr": meta["expr"], "tags": meta["tags"]}
    if not jpath.exists() or not cpath.exists():
        yield dict(base, kind="NO_OUTPUT", row=-1, inrow={}, column="", jpmml=str(jpath.exists()), cpmml=str(cpath.exists()))
        return
    j, c = read_csv(jpath), read_csv(cpath)
    inputs = read_csv(corpus / "cases" / cid / "inputs.csv")
    names = inputs[0]
    rows = [dict(zip(names, r)) for r in inputs[1:]]
    jload, cload = j[0] == ["LOAD-ERROR"], c[0] == ["LOAD-ERROR"]
    if c[0] == ["CRASH"]:
        yield dict(base, kind="CRASH", row=-1, inrow={}, column="", jpmml=j[1][0] if jload else "(scores)", cpmml=c[1][0])
        return
    if jload or cload:
        if jload and cload:
            kind = "LOAD_BOTH"
        elif cload and len(j) > 1 and all(is_error(r[0]) for r in j[1:]):
            kind = "LOAD_BOTH"
        else:
            kind = "LOAD_CPMML_ONLY" if cload else "LOAD_JPMML_ONLY"
        jtext = j[1][0] if jload else ("(every row: " + j[1][0] + ")" if kind == "LOAD_BOTH" else "(scores)")
        yield dict(base, kind=kind, row=-1, inrow={}, column="", jpmml=jtext, cpmml=c[1][0] if cload else "(scores)")
        return
    jhead, chead = j[0], c[0]
    for col in jhead:
        if col not in chead:
            yield dict(base, kind="MISSING_COLUMN", row=-1, inrow={}, column=col, jpmml="(column)", cpmml="(no such column)")
            continue
        ji, ci = jhead.index(col), chead.index(col)
        # cPMML's string text of x in the row: a rule about a record JPMML
        # fails as a whole checks what cPMML gave for x (cpmml_x), since the
        # other columns (the target) do not depend on x.
        xcol = next((chead.index(n) for n in X_COLUMNS if n in chead), None)
        for r in range(1, len(j)):
            jv = j[r][ji]
            cv = c[r][ci] if r < len(c) else "(no row)"
            if jv == cv:
                kind = "PASS"
            elif is_error(jv) and is_error(cv):
                kind = "ERROR_BOTH"
            elif is_error(jv):
                kind = "ERROR_JPMML_ONLY"
            elif is_error(cv):
                kind = "ERROR_CPMML_ONLY"
            else:
                kind = classify(jv, cv)
            xcell = c[r][xcol] if xcol is not None and r < len(c) else ""
            yield dict(base, kind=kind, row=r, inrow=rows[r - 1] if r - 1 < len(rows) else {}, column=col, jpmml=jv,
                       cpmml=cv, xcell=xcell)


def group_key(cell):
    """Errors and load failures group by the two messages; a value
    difference by the context, the declared type of x and the outermost
    element of the expression."""
    j, c = cell["jpmml"], cell["cpmml"]
    j_err = is_error(j) or j.startswith("LOAD-ERROR")
    c_err = is_error(c) or c.startswith(("LOAD-ERROR", "CRASH"))
    if j_err or c_err:
        return (cell["kind"], error_class(j) if j_err else "value", error_class(c) if c_err else "value")
    x_type = next((t[2:] for t in cell["tags"] if t.startswith("x:")), "?")
    return (cell["kind"], f"ctx{cell['context']} x:{x_type} col:{cell['column']}", head(cell["expr"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--departures", default=str(HERE / "departures.json"))
    ap.add_argument("--examples", type=int, default=1, help="examples printed per finding group")
    ap.add_argument("--json", default=None, help="also write the summary as JSON here")
    args = ap.parse_args()
    corpus = Path(args.corpus)
    _, rules = load_rules(Path(args.departures))

    kinds, expected = Counter(), Counter()
    cases_with_finding = set()
    groups = defaultdict(list)
    ncases = 0
    with open(corpus / "findings.jsonl", "w") as out:
        for line in open(corpus / "cases.jsonl"):
            meta = json.loads(line)
            ncases += 1
            for cell in compare_case(meta, corpus):
                kinds[cell["kind"]] += 1
                if cell["kind"] in PASS:
                    continue
                dep = next((d for d, rule in rules if rule_matches(rule, cell)), None)
                if dep:
                    expected[dep] += 1
                    continue
                cases_with_finding.add(cell["case"])
                groups[group_key(cell)].append(cell)
                out.write(json.dumps(cell) + "\n")

    print(f"cases: {ncases}  cases with a finding: {len(cases_with_finding)}")
    print("cells/cases by class: " + "  ".join(f"{k}: {v}" for k, v in sorted(kinds.items())))
    print("expected under a departure: " + ("  ".join(f"{k}: {v}" for k, v in sorted(expected.items())) or "none"))
    print(f"\nfinding groups: {len(groups)}")
    for key, cells in sorted(groups.items(), key=lambda kv: (-len({c['case'] for c in kv[1]}), kv[0])):
        ncase = len({c["case"] for c in cells})
        print(f"\n[{key[0]}] {' | '.join(key[1:])}  ({len(cells)} cells, {ncase} cases)")
        for cell in cells[: args.examples]:
            print(f"    {cell['case']} row {cell['row']} col {cell['column']}: JPMML {cell['jpmml']!r}  cPMML {cell['cpmml']!r}")
            print(f"      expr: {cell['expr'][:200]}")
            if cell["inrow"]:
                print(f"      row: {cell['inrow']}")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "cases": ncases, "cases_with_finding": len(cases_with_finding), "kinds": kinds, "expected": expected,
            "groups": [{"key": list(k), "cells": len(v), "cases": len({c['case'] for c in v}), "example": v[0]}
                       for k, v in groups.items()]}, indent=1, default=dict))
    return 1 if groups else 0


if __name__ == "__main__":
    sys.exit(main())
