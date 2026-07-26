#!/usr/bin/env python3
"""
Generate ground-truth test fixtures for cPMML models using JPMML as the oracle.

Extracts the PMML from each test zip, scores sample inputs via the JPMML
evaluator jar, and writes oracle CSV fixtures.

Usage:
    python test/generate_fixtures_jpmml.py [model_name ...]

    With no arguments, regenerates fixtures for ALL models under test/data/model/.
    With arguments, regenerates only the named models (e.g. HousingGBTClassifier_PCA).

Requires: Java 11+, the JPMML evaluator jar at tools/pmml-evaluator-example-executable-1.7.7.jar
"""

from __future__ import annotations
import csv, json, os, subprocess, sys, tempfile, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).parent.parent
JAR  = REPO / "tools" / "pmml-evaluator-example-executable-1.7.7.jar"
MODEL_DIR   = REPO / "test" / "data" / "model"
DATASET_DIR = REPO / "test" / "data" / "dataset"


def score_with_jpmml(pmml_path: Path, input_rows: list[dict]) -> list[dict]:
    """Score input_rows against the PMML model using the JPMML evaluator jar.

    Returns a list of dicts, one per input row, containing all JPMML output fields.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Write input CSV
        in_csv = tmp / "input.csv"
        if not input_rows:
            return []
        cols = list(input_rows[0].keys())
        with open(in_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(input_rows)

        # Run JPMML evaluator
        out_csv = tmp / "output.csv"
        cmd = [
            "java", "-cp", str(JAR),
            "org.jpmml.evaluator.example.EvaluationExample",
            "--model", str(pmml_path),
            "--input", str(in_csv),
            "--output", str(out_csv),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"  JPMML error: {result.stderr.strip()[:200]}", file=sys.stderr)
            return []

        # Read output CSV
        with open(out_csv, newline="") as f:
            return list(csv.DictReader(f))


def protected_target_columns(pmml_path: Path) -> set[str]:
    """Target columns that must keep their ACTUAL values from the input CSV.

    When a model has feature="residual" OutputFields, the target column is a
    genuine model input (residual = actual - predicted).  JPMML echoes the
    target field name in its results with the *predicted* value, which must
    not overwrite the actual value the residuals were computed from.
    """
    try:
        root = ET.parse(pmml_path).getroot()
    except ET.ParseError:
        return set()

    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    has_residual = any(
        local(el.tag) == "OutputField" and el.get("feature") == "residual"
        for el in root.iter()
    )
    if not has_residual:
        return set()

    return {
        el.get("name")
        for el in root.iter()
        if local(el.tag) == "MiningField" and el.get("usageType") == "target" and el.get("name")
    }


def generate(model_name: str) -> bool:
    """Regenerate the fixture CSV for one model using JPMML as oracle.

    Returns True on success, False on failure.
    """
    zip_path = MODEL_DIR / f"{model_name}.zip"
    csv_path = DATASET_DIR / f"{model_name}.csv"

    if not zip_path.exists():
        print(f"  SKIP: {zip_path} not found")
        return False

    # Extract PMML from zip
    with zipfile.ZipFile(zip_path) as z:
        pmml_name = z.namelist()[0]
        with tempfile.TemporaryDirectory() as tmp:
            pmml_path = Path(tmp) / pmml_name
            z.extract(pmml_name, tmp)

            # Read existing CSV to get input rows
            if not csv_path.exists():
                print(f"  SKIP: {csv_path} not found")
                return False

            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                all_cols = list(reader.fieldnames)
                existing_rows = list(reader)

            # Score with JPMML
            print(f"  Scoring {model_name} ({len(existing_rows)} rows) via JPMML...", flush=True)
            protected = protected_target_columns(pmml_path)
            oracle_rows = score_with_jpmml(pmml_path, existing_rows)

            if not oracle_rows:
                print(f"  FAIL: JPMML returned no results")
                return False

    # Merge: keep all JPMML output columns, overlay on existing columns
    oracle_cols = list(oracle_rows[0].keys())
    # Build merged column list: original columns + any new oracle columns
    merged_cols = list(all_cols)
    for oc in oracle_cols:
        if oc not in merged_cols:
            merged_cols.append(oc)

    # Write updated CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=merged_cols, extrasaction="ignore")
        w.writeheader()
        for i, row in enumerate(existing_rows):
            if i < len(oracle_rows):
                # Overlay oracle values, keeping actual target inputs intact
                merged = dict(row)
                merged.update({k: v for k, v in oracle_rows[i].items() if k not in protected})
                w.writerow(merged)
            else:
                w.writerow(row)

    print(f"  -> {csv_path} ({len(oracle_rows)} rows)")
    return True


def main():
    if not JAR.exists():
        print(f"ERROR: JPMML jar not found at {JAR}")
        print("  Run: tools/cross-validate.sh  (it downloads the jar)")
        sys.exit(1)

    if len(sys.argv) > 1:
        models = sys.argv[1:]
    else:
        # All models
        models = sorted(p.stem for p in MODEL_DIR.glob("*.zip"))

    ok = 0
    fail = 0
    for model_name in models:
        print(f"[{model_name}]")
        if generate(model_name):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} regenerated, {fail} failed")


if __name__ == "__main__":
    main()
