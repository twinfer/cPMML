#!/usr/bin/env python3
"""
run.py -- score a corpus written by tools/fuzz/gen.py with both evaluators:
JPMML 1.7.7 through tools/fuzz/JpmmlOracle.java (one JVM for every case) and
cPMML through test/fuzz_runner.cc (one process for every case).

  DIR/jpmml/<id>.csv   JpmmlOracle's output per case
  DIR/cpmml/<id>.csv   fuzz_runner's output per case

If fuzz_runner dies on a case (a crash is a finding, not a reason to stop),
that case's output is a one-column file "CRASH" and the run resumes with the
next case. A fuzz_runner that stops before any case or exits 2 (its
usage or manifest error) stops run.py instead.

Usage:
  tools/fuzz/run.py --corpus DIR --runner BUILD/test/fuzz_runner.exe
      [--jar tools/pmml-evaluator-example-executable-1.7.7.jar] [--skip-jpmml]

Then tools/fuzz/compare.py --corpus DIR. Requires python3 (standard library),
java and javac.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def manifest(corpus: Path, side: str, ids) -> Path:
    (corpus / side).mkdir(exist_ok=True)
    path = corpus / f"manifest.{side}.tsv"
    with open(path, "w") as f:
        for i in ids:
            d = corpus / "cases" / i
            f.write(f"{i}\t{d / 'model.pmml'}\t{d / 'inputs.csv'}\t{corpus / side / (i + '.csv')}\n")
    return path


def run_jpmml(corpus: Path, ids, jar: str) -> None:
    classes = corpus / "oracle-classes"
    classes.mkdir(exist_ok=True)
    subprocess.run(["javac", "-nowarn", "-cp", jar, "-d", str(classes), str(HERE / "JpmmlOracle.java")], check=True)
    m = manifest(corpus, "jpmml", ids)
    subprocess.run(["java", "-cp", f"{classes}:{jar}", "JpmmlOracle", str(m)], check=True)


def run_cpmml(corpus: Path, ids, runner: str) -> None:
    remaining = list(ids)
    while remaining:
        m = manifest(corpus, "cpmml", remaining)
        for i in remaining:
            (corpus / "cpmml" / f"{i}.csv").unlink(missing_ok=True)
        proc = subprocess.run([runner, str(m)])
        if proc.returncode == 0:
            return
        # fuzz_runner opens a case's output before loading it, so the last
        # case with an output file is the one it died on.
        done = 0
        while done < len(remaining) and (corpus / "cpmml" / f"{remaining[done]}.csv").exists():
            done += 1
        if not done or proc.returncode == 2:
            # Exit 2 is fuzz_runner refusing its arguments or a manifest
            # line, and with no output file no case started: neither is a
            # crash of a case, so stop rather than blame one.
            sys.exit(f"run.py: fuzz_runner exited {proc.returncode} on manifest {m} (not a case crash)")
        crashed = remaining[done - 1]
        print(f"run.py: fuzz_runner exited {proc.returncode} on {crashed}", file=sys.stderr)
        with open(corpus / "cpmml" / f"{crashed}.csv", "w") as f:
            f.write(f"CRASH\nCRASH: exit {proc.returncode}\n")
        remaining = remaining[remaining.index(crashed) + 1:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--jar", default=str(REPO / "tools" / "pmml-evaluator-example-executable-1.7.7.jar"))
    ap.add_argument("--skip-jpmml", action="store_true", help="reuse DIR/jpmml (rescore cPMML only)")
    args = ap.parse_args()
    corpus = Path(args.corpus).resolve()
    ids = [json.loads(line)["id"] for line in open(corpus / "cases.jsonl")]
    if not args.skip_jpmml:
        run_jpmml(corpus, ids, args.jar)
    run_cpmml(corpus, ids, str(Path(args.runner).resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
