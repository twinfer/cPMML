#!/usr/bin/env python3
"""
cross-validate-text.py -- re-score every fixture with JPMML 1.7.7 and with
cPMML, and compare the predicted value and every OutputField as exact text.

tools/cross-validate.sh checks cPMML against JPMML on JPMML's own test
models; this one runs over cPMML's fixture suite (test/data/model/*.zip with
test/data/dataset/*.csv) and compares text, not numbers within a tolerance:
"1.5" is not "1.500000", "1.0" is not "1".

For each fixture:
  1. the PMML is extracted from the zip;
  2. the input CSV is the fixture's columns that are DataFields and not the
     model's target, an empty cell written N/A (JPMML's missing value);
  3. JPMML's EvaluationExample scores it, and
     pmml_tester.exe <zip> <inputs> --dump <out> scores it with cPMML;
  4. every column JPMML writes is compared cell by cell with cPMML's.

A mismatch is reported as
  FORMAT    the two texts differ but read as exactly the same number: a
            formatting difference, which this tool exists to find;
  ROUNDING  numbers within 1e-12 relative: the same computation rounded
            differently (summation order, libm), not a formatting difference;
  VALUE     anything else: a different value, or one side has none.
Time-series and transactional association fixtures are skipped (they are not
scored row by row), as are fixtures JPMML cannot load or evaluate (reported as
JPMM-ERROR with its exception).

Usage:
  tools/cross-validate-text.py --cpmml-exe BUILD/test/pmml_tester.exe
      [--jar tools/pmml-evaluator-example-executable-1.7.7.jar]
      [--fixtures test/data] [--work DIR] [--only NAME ...] [-j N]

Exit status: 0 when no fixture has a mismatch (JPMML errors and skips do not
count), 1 otherwise. Requires python3 (standard library only) and java.
The text helpers (classify, same_number, the CSV reader) live in
tools/pmmltext.py, shared with tools/fuzz/compare.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from pmmltext import JPMML_MISSING, classify, model_info, read_csv  # noqa: E402


def check(name: str, args) -> dict:
    zip_path = Path(args.fixtures) / "model" / f"{name}.zip"
    csv_path = Path(args.fixtures) / "dataset" / f"{name}.csv"
    work = Path(args.work) / name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    with zipfile.ZipFile(zip_path) as z:
        member = z.namelist()[0]
        pmml = work / "model.pmml"
        pmml.write_bytes(z.read(member))

    function, target, data_fields, outputs = model_info(pmml)
    rows = read_csv(csv_path)
    header = rows[0]
    if function == "timeSeries" or "transaction" in header:
        return {"name": name, "status": "SKIP", "detail": "time series or transactional"}

    inputs = [c for c in header if c in data_fields and c != target and c not in outputs]
    in_csv = work / "inputs.csv"
    with open(in_csv, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(inputs)
        for row in rows[1:]:
            cells = dict(zip(header, row))
            w.writerow([cells.get(c, "") or JPMML_MISSING for c in inputs])

    j_out = work / "jpmml.csv"
    jp = subprocess.run(
        ["java", "-cp", args.jar, "org.jpmml.evaluator.example.EvaluationExample",
         "--model", str(pmml), "--input", str(in_csv), "--output", str(j_out)],
        capture_output=True, text=True)
    if jp.returncode != 0 or not j_out.exists():
        lines = [l for l in (jp.stderr + jp.stdout).splitlines() if "Exception" in l]
        return {"name": name, "status": "JPMML-ERROR", "detail": lines[0] if lines else "exit %d" % jp.returncode}

    c_out = work / "cpmml.csv"
    cp = subprocess.run([args.cpmml_exe, str(zip_path), str(in_csv), "--dump", str(c_out)],
                        capture_output=True, text=True)
    if cp.returncode != 0 or not c_out.exists():
        return {"name": name, "status": "CPMML-ERROR", "detail": (cp.stderr or cp.stdout).strip()[:200]}

    jrows, crows = read_csv(j_out), read_csv(c_out)
    jhead, chead = jrows[0], crows[0]
    mismatches = []
    for col in jhead:
        if col not in chead:
            mismatches.append(("VALUE", col, -1, "(column)", "(no such column)"))
            continue
        ji, ci = jhead.index(col), chead.index(col)
        for r in range(1, max(len(jrows), len(crows))):
            jv = jrows[r][ji] if r < len(jrows) else "(no row)"
            cv = crows[r][ci] if r < len(crows) else "(no row)"
            if jv == cv:
                continue
            kind = classify(jv, cv)
            mismatches.append((kind, col, r, jv, cv))
    status = "PASS" if not mismatches else "MISMATCH"
    return {"name": name, "status": status, "mismatches": mismatches}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpmml-exe", required=True)
    ap.add_argument("--jar", default=str(REPO / "tools" / "pmml-evaluator-example-executable-1.7.7.jar"))
    ap.add_argument("--fixtures", default=str(REPO / "test" / "data"))
    ap.add_argument("--work", default=None, help="keep inputs and both outputs here (default: a temp dir)")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("-j", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    args.cpmml_exe = str(Path(args.cpmml_exe).resolve())
    temp = None
    if args.work is None:
        temp = tempfile.mkdtemp(prefix="cpmml-text-")
        args.work = temp

    names = sorted(p.stem for p in (Path(args.fixtures) / "model").glob("*.zip")
                   if (Path(args.fixtures) / "dataset" / f"{p.stem}.csv").exists())
    if args.only:
        names = [n for n in names if n in args.only]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.j) as pool:
        results = list(pool.map(lambda n: check(n, args), names))

    counts: dict = {}
    for res in results:
        counts[res["status"]] = counts.get(res["status"], 0) + 1
        if res["status"] == "PASS":
            continue
        if res["status"] != "MISMATCH":
            print(f"{res['status']:12} {res['name']}: {res['detail']}")
            continue
        kinds = sorted({m[0] for m in res["mismatches"]})
        print(f"MISMATCH     {res['name']}: {len(res['mismatches'])} cells ({', '.join(kinds)})")
        for kind, col, row, jv, cv in res["mismatches"][:5]:
            print(f"    {kind:6} row {row} {col}: JPMML {jv!r}  cPMML {cv!r}")

    print()
    print("  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    if temp:
        shutil.rmtree(temp, ignore_errors=True)
    return 1 if counts.get("MISMATCH") else 0


if __name__ == "__main__":
    sys.exit(main())
