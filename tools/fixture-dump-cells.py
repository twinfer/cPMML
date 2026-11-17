#!/usr/bin/env python3
r"""
fixture-dump-cells.py -- the two halves of tools/fixture-dump-diff.sh.

  dump --exe PMML_TESTER --fixtures DIR --out DIR [--only NAME ...] [-j N]
      Scores every fixture (DIR/model/<name>.zip with DIR/dataset/<name>.csv)
      with `pmml_tester.exe <zip> <inputs> --dump <csv>` and writes, per
      fixture, OUT/<name>.cells: one line per output cell,
          <row>\t<column>\t<text>
      in row order, then the dump's column order. The inputs are those of
      tools/cross-validate-text.py: the fixture's columns that are DataFields
      and neither the target nor an OutputField, an empty cell written N/A.
      A time-series fixture is its own input (pmml_tester dumps every step).
      A tester that cannot load the model, or writes no dump, gives the single
      line "*\t*\tLOAD-ERROR: <first line of its stderr>".

  diff BASE_DIR HEAD_DIR
      Compares two such directories cell by cell and prints every cell whose
      text differs, every cell only one side has, and every fixture only one
      side has. Exit status 0 when nothing differs, 1 otherwise, 2 on an
      error (dump too).

Standard library only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import importlib.util
import os
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True  # no tools/__pycache__ from the import below
_spec = importlib.util.spec_from_file_location("cross_validate_text", HERE / "cross-validate-text.py")
cvt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cvt)

MISSING = "N/A"


def escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n")


def fixtures(root: Path, only):
    names = sorted(p.stem for p in (root / "model").glob("*.zip") if (root / "dataset" / f"{p.stem}.csv").exists())
    if only:
        unknown = sorted(set(only) - set(names))
        if unknown:
            print(f"fixture-dump-cells.py: no such fixture: {' '.join(unknown)}", file=sys.stderr)
            raise SystemExit(2)
        names = [n for n in names if n in set(only)]
    return names


def dump_one(name: str, exe: str, root: Path, out: Path) -> str:
    work = out / "work" / name
    work.mkdir(parents=True, exist_ok=True)
    zip_path = root / "model" / f"{name}.zip"
    csv_path = root / "dataset" / f"{name}.csv"
    with zipfile.ZipFile(zip_path) as z:
        pmml = work / "model.pmml"
        pmml.write_bytes(z.read(z.namelist()[0]))
    function, target, data_fields, outputs = cvt.model_info(pmml)
    rows = cvt.read_csv(csv_path)
    header = rows[0]
    if function == "timeSeries":
        in_csv = csv_path
    else:
        inputs = [c for c in header if c in data_fields and c != target and c not in outputs]
        in_csv = work / "inputs.csv"
        with open(in_csv, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(inputs)
            for row in rows[1:]:
                cells = dict(zip(header, row))
                w.writerow([cells.get(c, "") or MISSING for c in inputs])

    dump = work / "dump.csv"
    dump.unlink(missing_ok=True)
    proc = subprocess.run([exe, str(zip_path.resolve()), str(Path(in_csv).resolve()), "--dump", str(dump)],
                          capture_output=True, text=True)
    lines = []
    if proc.returncode != 0 or not dump.exists():
        err = (proc.stderr or proc.stdout).strip().splitlines()
        lines.append(f"*\t*\tLOAD-ERROR: {escape(err[0] if err else 'exit %d' % proc.returncode)}")
    else:
        drows = cvt.read_csv(dump)
        # csv reads an empty line as no cell: the header of a model with no
        # target field and no top-level OutputField is the one column "".
        dhead = (drows[0] if drows else []) or [""]
        for r, row in enumerate(drows[1:], start=1):
            row = row or [""]
            for c, col in enumerate(dhead):
                lines.append(f"{r}\t{escape(col)}\t{escape(row[c] if c < len(row) else '(no cell)')}")
    (out / f"{name}.cells").write_text("".join(line + "\n" for line in lines))
    return name


def cmd_dump(args) -> int:
    root = Path(args.fixtures).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.cells"):
        old.unlink()
    names = fixtures(root, args.only)
    exe = str(Path(args.exe).resolve())
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.j) as pool:
        for future in concurrent.futures.as_completed([pool.submit(dump_one, n, exe, root, out) for n in names]):
            future.result()
    print(f"dumped {len(names)} fixtures with {exe} into {out}")
    return 0


def read_cells(path: Path):
    cells = {}
    for line in path.read_text().splitlines():
        row, col, text = line.split("\t", 2)
        cells[(row, col)] = text
    return cells


def cmd_diff(args) -> int:
    base, head = Path(args.base), Path(args.head)
    bnames = {p.stem for p in base.glob("*.cells")}
    hnames = {p.stem for p in head.glob("*.cells")}
    changed_fixtures = changed_cells = total_cells = 0
    for name in sorted(bnames | hnames):
        if name not in hnames:
            print(f"{name}: only in base")
            changed_fixtures += 1
            continue
        if name not in bnames:
            print(f"{name}: only in head")
            changed_fixtures += 1
            continue
        b, h = read_cells(base / f"{name}.cells"), read_cells(head / f"{name}.cells")
        keys = list(b) + [k for k in h if k not in b]
        total_cells += len(keys)
        diffs = [k for k in keys if b.get(k) != h.get(k)]
        if not diffs:
            continue
        changed_fixtures += 1
        changed_cells += len(diffs)
        for row, col in diffs:
            bv = b.get((row, col), "(no cell)")
            hv = h.get((row, col), "(no cell)")
            print(f"{name}\trow {row}\t{col}\t{bv}\t->\t{hv}")
    print(f"{len(bnames | hnames)} fixtures, {total_cells} cells compared: "
          f"{changed_cells} cells changed in {changed_fixtures} fixtures")
    return 1 if changed_fixtures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--exe", required=True)
    d.add_argument("--fixtures", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--only", nargs="*", default=None)
    d.add_argument("-j", type=int, default=os.cpu_count() or 4)
    c = sub.add_parser("diff")
    c.add_argument("base")
    c.add_argument("head")
    args = ap.parse_args()
    try:
        return cmd_dump(args) if args.cmd == "dump" else cmd_diff(args)
    except (OSError, ValueError, zipfile.BadZipFile) as e:
        print(f"fixture-dump-cells.py: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
