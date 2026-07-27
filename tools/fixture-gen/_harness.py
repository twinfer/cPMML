#!/usr/bin/env python3
"""
Shared machinery for the cPMML fixture generators.

A fixture is a pair of committed files that ctest discovers automatically
(see test/CMakeLists.txt):

    test/data/model/<name>.zip       containing <name>.pmml
    test/data/dataset/<name>.csv     inputs plus the expected output columns

The generators in this directory produce those files from an *independent*
reference implementation, so that the expected columns say what the model
should score rather than what cPMML currently scores.  Two rules keep that
property intact:

  1. The oracle reads the PMML document, never a Python-side copy of the
     model parameters.  It walks the same XML cPMML walks, so a generator
     bug cannot quietly cancel out an evaluator bug.

  2. Every generator supports --check, which regenerates in memory and diffs
     against the committed files instead of overwriting them.  That turns
     "these fixtures still agree with the oracle" into something runnable.

Python is deliberately not a build or test dependency: `cmake && make &&
ctest` consumes the committed CSVs and needs nothing but a compiler.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import io
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence
from xml.sax.saxutils import escape, quoteattr

REPO = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = REPO / "test" / "data" / "model"
DATASET_DIR = REPO / "test" / "data" / "dataset"

# Fixed member timestamp so regenerating a fixture produces a byte-identical
# zip rather than a spurious diff on every run.
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)

# The committed corpus is CRLF (Python's csv default); stay consistent with it.
_CSV_NEWLINE = "\r\n"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def provenance(script: str, oracle: str, note: str = "") -> str:
    """A PMML <Header> recording how the fixture was produced.

    Both evaluators ignore Header content, so this costs nothing at scoring
    time and makes an oracle-derived fixture distinguishable at a glance from
    a hand-computed one.
    """
    annotation = f"oracle={oracle}" + (f"; {note}" if note else "")
    return (
        '  <Header copyright="cPMML test fixture">\n'
        f"    <Application name={quoteattr(script)}/>\n"
        f"    <Annotation>{escape(annotation)}</Annotation>\n"
        "  </Header>\n"
    )


def format_number(value: float) -> str:
    """Shortest representation that round-trips back to the same double."""
    return repr(float(value))


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@dataclass
class Fixture:
    """One model/dataset pair.

    pmml=None means "keep the committed model, refresh only the expected
    columns" -- the right mode for a hand-authored PMML whose expectations
    were never checked against a reference.
    """

    name: str
    columns: Sequence[str]
    rows: Sequence[dict]
    pmml: str | None = None

    @property
    def model_path(self) -> Path:
        return MODEL_DIR / f"{self.name}.zip"

    @property
    def dataset_path(self) -> Path:
        return DATASET_DIR / f"{self.name}.csv"

    def render_csv(self) -> str:
        buf = io.StringIO(newline="")
        writer = csv.DictWriter(buf, fieldnames=list(self.columns), lineterminator=_CSV_NEWLINE)
        writer.writeheader()
        for row in self.rows:
            writer.writerow({c: row.get(c, "") for c in self.columns})
        return buf.getvalue()


def read_model(name: str) -> str:
    """The PMML document inside a committed fixture zip."""
    path = MODEL_DIR / f"{name}.zip"
    if not path.exists():
        raise FileNotFoundError(f"no committed model at {path}")
    with zipfile.ZipFile(path) as z:
        members = [m for m in z.namelist() if m.endswith(".pmml")]
        if len(members) != 1:
            raise ValueError(f"{path} should hold exactly one .pmml, found {members}")
        return z.read(members[0]).decode("utf-8")


def _write_model(fixture: Fixture) -> None:
    info = zipfile.ZipInfo(f"{fixture.name}.pmml", date_time=_ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(fixture.model_path, "w") as z:
        z.writestr(info, fixture.pmml)


# ---------------------------------------------------------------------------
# write / check
# ---------------------------------------------------------------------------


def _diff(label: str, expected: str, actual: str) -> list[str]:
    if expected == actual:
        return []
    return list(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile=f"{label} (committed)",
            tofile=f"{label} (regenerated)",
        )
    )


def _check_one(fixture: Fixture) -> list[str]:
    diffs: list[str] = []

    if fixture.pmml is not None:
        try:
            diffs += _diff(f"{fixture.name}.pmml", read_model(fixture.name), fixture.pmml)
        except FileNotFoundError as exc:
            diffs.append(f"{exc}\n")

    if fixture.dataset_path.exists():
        with open(fixture.dataset_path, newline="") as handle:
            committed = handle.read()
        diffs += _diff(f"{fixture.name}.csv", committed, fixture.render_csv())
    else:
        diffs.append(f"no committed dataset at {fixture.dataset_path}\n")

    return diffs


def emit(fixtures: Iterable[Fixture], *, check: bool) -> int:
    """Write the fixtures, or diff them against what is committed.

    Returns a process exit code: 0 when everything agrees, 1 on any mismatch.
    """
    failures = 0

    for fixture in fixtures:
        if check:
            diffs = _check_one(fixture)
            if diffs:
                failures += 1
                print(f"MISMATCH {fixture.name}", file=sys.stderr)
                sys.stderr.writelines(diffs)
                print(file=sys.stderr)
            else:
                print(f"ok       {fixture.name}")
            continue

        if fixture.pmml is not None:
            _write_model(fixture)
        with open(fixture.dataset_path, "w", newline="") as handle:
            handle.write(fixture.render_csv())
        kind = "model+dataset" if fixture.pmml is not None else "dataset only"
        print(f"wrote    {fixture.name}  ({kind})")

    if check and failures:
        print(
            f"\n{failures} fixture(s) disagree with the oracle.\n"
            "Either the evaluator changed behaviour, or the oracle version moved.\n"
            "Re-run without --check to accept the regenerated values.",
            file=sys.stderr,
        )
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(build: Callable[[], list[Fixture]], description: str) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "names",
        nargs="*",
        help="fixture names to process (default: all produced by this generator)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="diff against the committed fixtures instead of overwriting them; "
        "exits non-zero on any difference",
    )
    parser.add_argument("--list", action="store_true", help="print the fixture names and exit")
    args = parser.parse_args()

    fixtures = build()

    if args.list:
        for fixture in fixtures:
            print(fixture.name)
        return 0

    if args.names:
        known = {f.name for f in fixtures}
        unknown = sorted(set(args.names) - known)
        if unknown:
            parser.error(f"unknown fixture(s): {', '.join(unknown)}\nknown: {', '.join(sorted(known))}")
        fixtures = [f for f in fixtures if f.name in set(args.names)]

    return emit(fixtures, check=args.check)
