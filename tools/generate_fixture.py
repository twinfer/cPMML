#!/usr/bin/env python3
"""
tools/generate_fixture.py — cPMML test fixture generator / conformance tester

Uses jpmml-evaluator as a reference oracle to create and verify test fixtures
for the C++ model_tester.exe harness:

    test/data/model/<name>.zip      — zipped PMML model
    test/data/dataset/<name>.csv   — inputs + "prediction" column

jpmml-evaluator is AGPL-3.0.

Subcommands
-----------
generate <pmml> <input_csv> [options]
    Score <input_csv> with jpmml-evaluator to produce the fixture pair.
    The first PMML OutputField (feature="predictedValue") is aliased as the
    "prediction" column that model_tester.exe checks.

    --name NAME     fixture stem (default: PMML file stem)
    --force         overwrite an existing fixture

generate --all [--force]
    Regenerate every existing fixture in test/data/ using jpmml-evaluator.
    Input rows are extracted from the existing CSV (active-field columns only).

cross-validate [options]
    Download jpmml-evaluator's own PMML test files at runtime, score them
    with both jpmml-evaluator and the cPMML model_tester.exe binary, and
    report any discrepancies.  Nothing is written to test/data/.

    --category CAT [CAT ...]   restrict to specific model categories
    --cpmml-exe PATH           path to model_tester.exe (default: auto-detect)

Examples
--------
    # Create a new fixture for a model you just wrote:
    python tools/generate_fixture.py generate path/to/MyModel.pmml path/to/inputs.csv

    # Regenerate all existing fixtures (after changing a model):
    python tools/generate_fixture.py generate --all --force

    # Run jpmml conformance tests against cPMML (tree + regression only):
    python tools/generate_fixture.py cross-validate --category tree regression
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

JPMML_REPO = "jpmml/jpmml-evaluator"
JAR_CACHE_DIR = Path.home() / ".cache" / "cpmml-jpmml"

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_MODEL_DIR = REPO_ROOT / "test" / "data" / "model"
TEST_DATASET_DIR = REPO_ROOT / "test" / "data" / "dataset"

# jpmml-evaluator test PMML paths (relative to repo root).
# These are fetched at runtime — never committed here (AGPL-3.0).
JPMML_TEST_CATEGORIES: dict[str, list[str]] = {
    "tree": [
        "pmml-evaluator/src/test/resources/pmml/tree/ClassificationOutputTest.pmml",
        "pmml-evaluator/src/test/resources/pmml/tree/DefaultChildTest.pmml",
        "pmml-evaluator/src/test/resources/pmml/tree/MissingValueStrategyTest.pmml",
        "pmml-evaluator/src/test/resources/pmml/tree/NoTrueChildStrategyTest.pmml",
        "pmml-evaluator/src/test/resources/pmml/tree/ScalarVerificationTest.pmml",
    ],
    "regression": [
        "pmml-evaluator/src/test/resources/pmml/regression/RegressionOutputTest.pmml",
        "pmml-evaluator/src/test/resources/pmml/regression/TransformationDictionaryTest.pmml",
        "pmml-evaluator/src/test/resources/pmml/regression/DefaultValueTest.pmml",
        "pmml-evaluator/src/test/resources/pmml/regression/EmptyTargetCategoryTest.pmml",
        "pmml-evaluator/src/test/resources/pmml/regression/PriorProbabilitiesTest.pmml",
        "pmml-evaluator/src/test/resources/pmml/regression/CategoricalResidualTest.pmml",
        "pmml-evaluator/src/test/resources/pmml/regression/ContinuousResidualTest.pmml",
    ],
    "clustering": [
        "pmml-evaluator/src/test/resources/pmml/clustering/RankingTest.pmml",
    ],
    "naive_bayes": [
        "pmml-evaluator/src/test/resources/pmml/naive_bayes/BayesInputTest.pmml",
        "pmml-evaluator/src/test/resources/pmml/naive_bayes/TargetValueCountsTest.pmml",
    ],
    "nearest_neighbor": [
        "pmml-evaluator/src/test/resources/pmml/nearest_neighbor/ClusteringNeighborhoodTest.pmml",
        "pmml-evaluator/src/test/resources/pmml/nearest_neighbor/MixedNeighborhoodTest.pmml",
        "pmml-evaluator/src/test/resources/pmml/nearest_neighbor/TieBreakTest.pmml",
    ],
    "scorecard": [
        "pmml-evaluator/src/test/resources/pmml/scorecard/AttributeReasonCodeTest.pmml",
        "pmml-evaluator/src/test/resources/pmml/scorecard/CharacteristicReasonCodeTest.pmml",
        "pmml-evaluator/src/test/resources/pmml/scorecard/ComplexPartialScoreTest.pmml",
    ],
    "rule_set": [
        "pmml-evaluator/src/test/resources/pmml/rule_set/CompoundRuleTest.pmml",
        "pmml-evaluator/src/test/resources/pmml/rule_set/SimpleRuleTest.pmml",
    ],
    "support_vector_machine": [
        "pmml-evaluator/src/test/resources/pmml/support_vector_machine/AlternateBinaryTargetCategoryTest.pmml",
        "pmml-evaluator/src/test/resources/pmml/support_vector_machine/VectorInstanceTest.pmml",
    ],
    "mining": [
        "pmml-evaluator/src/test/resources/pmml/mining/GradientBoosterTest.pmml",
        "pmml-evaluator/src/test/resources/pmml/mining/ModelChainSimpleTest.pmml",
        "pmml-evaluator/src/test/resources/pmml/mining/SelectAllTest.pmml",
    ],
    "general_regression": [
        "pmml-evaluator/src/test/resources/pmml/general_regression/ContrastMatrixTest.pmml",
        "pmml-evaluator/src/test/resources/pmml/general_regression/EmptyPPMatrixTest.pmml",
    ],
}

# ---------------------------------------------------------------------------
# JAR management
# ---------------------------------------------------------------------------

def _github_api(path: str) -> dict:
    url = f"https://api.github.com/repos/{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def _get_latest_version() -> str:
    data = _github_api(f"{JPMML_REPO}/releases/latest")
    return data["tag_name"]


def get_jar(version: Optional[str] = None) -> Path:
    """Return path to jpmml-evaluator executable JAR, downloading if needed."""
    if version is None:
        print("Checking jpmml-evaluator latest release...")
        version = _get_latest_version()

    jar_name = f"pmml-evaluator-example-executable-{version}.jar"
    jar_path = JAR_CACHE_DIR / jar_name

    if jar_path.exists():
        return jar_path

    JAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url = (f"https://github.com/{JPMML_REPO}/releases/download/"
           f"{version}/{jar_name}")
    print(f"Downloading jpmml-evaluator {version}...")
    urllib.request.urlretrieve(url, jar_path)
    print(f"  Cached: {jar_path}")
    return jar_path


# ---------------------------------------------------------------------------
# PMML parsing helpers
# ---------------------------------------------------------------------------

def _ns(root: ET.Element) -> str:
    """Return '{namespace}' prefix from the PMML root element tag."""
    tag = root.tag
    return tag[: tag.index("}") + 1] if tag.startswith("{") else ""


def _find_all(root: ET.Element, local_tag: str, ns: str) -> list[ET.Element]:
    return root.findall(f".//{ns}{local_tag}")


def find_primary_output(pmml_path: Path) -> Optional[str]:
    """
    Return the name of the primary output field.

    Search order:
      1. First <OutputField feature="predictedValue"> (or feature absent)
      2. First <OutputField> regardless of feature
      3. First <MiningField usageType="predicted"|"target">
    """
    root = ET.parse(pmml_path).getroot()
    ns = _ns(root)

    output_fields = _find_all(root, "OutputField", ns)
    for of in output_fields:
        if of.get("feature", "predictedValue") == "predictedValue":
            return of.get("name")
    if output_fields:
        return output_fields[0].get("name")

    for mf in _find_all(root, "MiningField", ns):
        if mf.get("usageType") in ("predicted", "target"):
            return mf.get("name")
    return None


def get_input_field_names(pmml_path: Path) -> list[str]:
    """
    Return the ordered list of input MiningField names (active + supplementary).
    These are the columns the evaluator accepts.
    """
    root = ET.parse(pmml_path).getroot()
    ns = _ns(root)
    return [
        mf.get("name", "")
        for mf in _find_all(root, "MiningField", ns)
        if mf.get("usageType", "active") in ("active", "supplementary")
        and mf.get("name") is not None
    ]


def get_output_field_names(pmml_path: Path) -> set[str]:
    """Return the set of PMML OutputField names and the target MiningField name."""
    root = ET.parse(pmml_path).getroot()
    ns = _ns(root)
    names: set[str] = set()
    for of in _find_all(root, "OutputField", ns):
        if of.get("name") is not None:
            names.add(of.get("name", ""))
    for mf in _find_all(root, "MiningField", ns):
        if mf.get("usageType") in ("predicted", "target") and mf.get("name") is not None:
            names.add(mf.get("name", ""))
    return names


def _decode_jpmml_col(col: str) -> str:
    """Decode jpmml hex-encoded column names, e.g. petal_x0020_length → petal length."""
    return re.sub(r"_x([0-9A-Fa-f]{4})_",
                  lambda m: chr(int(m.group(1), 16)), col)


def extract_model_verification(
    pmml_path: Path,
) -> Optional[tuple[list[dict], list[dict]]]:
    """
    Extract embedded <ModelVerification> rows if present.

    Returns (input_rows, output_rows) where each element is a list of
    {field_name: value} dicts, or None if the element is absent.

    Column names in the InlineTable may be hex-encoded (jpmml convention:
    space → _x0020_).  VerificationField/@column is the InlineTable column;
    VerificationField/@field is the PMML field name.
    """
    root = ET.parse(pmml_path).getroot()
    ns = _ns(root)

    mv = root.find(f".//{ns}ModelVerification")
    if mv is None:
        return None

    # Build XML-column → PMML-field mapping
    col_to_field: dict[str, str] = {}
    for vf in mv.findall(f"{ns}VerificationFields/{ns}VerificationField"):
        field = vf.get("field", "")
        col = vf.get("column", field)
        col_to_field[col] = field

    output_names = get_output_field_names(pmml_path)

    inline = mv.find(f".//{ns}InlineTable")
    if inline is None:
        return None

    input_rows: list[dict] = []
    output_rows: list[dict] = []

    for row_el in inline.findall(f"{ns}row"):
        inp: dict = {}
        out: dict = {}
        for child in row_el:
            # Strip namespace to get the raw column name
            raw_col = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            decoded = _decode_jpmml_col(raw_col)
            field = col_to_field.get(raw_col, col_to_field.get(decoded, decoded))
            val = child.text or ""
            if field in output_names:
                out[field] = val
            else:
                inp[field] = val
        input_rows.append(inp)
        output_rows.append(out)

    return input_rows, output_rows


def generate_synthetic_inputs(pmml_path: Path, n: int = 5) -> list[dict]:
    """
    Generate n synthetic input rows from the DataDictionary.
    Used when no <ModelVerification> is available.

    Strategy per dataType:
      - continuous  → midpoint of <Interval>, or 0.0
      - categorical → cycles through declared <Value> elements
      - boolean     → alternates true/false
      - integer     → 0
      - string      → first <Value>, or empty string
    """
    root = ET.parse(pmml_path).getroot()
    ns = _ns(root)
    input_names = set(get_input_field_names(pmml_path))

    rows: list[dict] = [{} for _ in range(n)]

    for df in _find_all(root, "DataField", ns):
        name = df.get("name", "")
        if name not in input_names:
            continue

        dtype = df.get("dataType", "string")
        values = [v.get("value", "") for v in df.findall(f"{ns}Value")]
        interval = df.find(f"{ns}Interval")

        if values:
            pool = (values * (n // len(values) + 1))[:n]
        elif dtype in ("double", "float") and interval is not None:
            lo = float(interval.get("leftMargin") or 0)
            hi = float(interval.get("rightMargin") or 1)
            mid = round((lo + hi) / 2, 6)
            pool = [str(mid)] * n
        elif dtype in ("double", "float"):
            pool = ["0.0"] * n
        elif dtype == "integer":
            pool = ["0"] * n
        elif dtype == "boolean":
            pool = (["true", "false"] * (n // 2 + 1))[:n]
        else:
            pool = [""] * n

        for i, row in enumerate(rows):
            row[name] = pool[i]

    return rows


# ---------------------------------------------------------------------------
# jpmml-evaluator runner
# ---------------------------------------------------------------------------

def run_jpmml(jar: Path, pmml_path: Path, input_csv: Path, output_csv: Path) -> None:
    """
    Invoke jpmml-evaluator CLI:
        java -jar <jar> --model <pmml> --input <input_csv> --output <output_csv>

    Raises RuntimeError on non-zero exit.
    """
    cmd = [
        "java", "-jar", str(jar),
        "--model", str(pmml_path),
        "--input", str(input_csv),
        "--output", str(output_csv),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "jpmml-evaluator exited non-zero")


# ---------------------------------------------------------------------------
# CSV utilities
# ---------------------------------------------------------------------------

def read_csv_file(path: Path) -> tuple[list[str], list[dict]]:
    """Return (fieldnames, rows) from a CSV file."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    fieldnames = list(reader.fieldnames or [])
    return fieldnames, rows


def write_csv_file(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_fixture_csv(
    input_rows: list[dict],
    jpmml_rows: list[dict],
    input_fields: list[str],
    primary_output_col: str,
) -> tuple[list[str], list[dict]]:
    """
    Merge jpmml input and output rows into cPMML fixture format.

    Column order:  input_fields | prediction | <remaining jpmml output cols>

    The primary_output_col from jpmml is aliased as "prediction" — that is
    the column model_tester.exe checks against Prediction::as_string().
    """
    if not jpmml_rows:
        raise ValueError("jpmml produced no output rows")

    jpmml_cols = list(jpmml_rows[0].keys())
    other_cols = [c for c in jpmml_cols if c != primary_output_col]
    fieldnames = input_fields + ["prediction"] + other_cols

    merged: list[dict] = []
    for inp, out in zip(input_rows, jpmml_rows):
        row = {f: inp.get(f, "") for f in input_fields}
        row["prediction"] = out.get(primary_output_col, "")
        for c in other_cols:
            row[c] = out.get(c, "")
        merged.append(row)

    return fieldnames, merged


# ---------------------------------------------------------------------------
# Fixture writing
# ---------------------------------------------------------------------------

def write_fixture(
    pmml_path: Path,
    fieldnames: list[str],
    rows: list[dict],
    name: str,
    force: bool = False,
) -> bool:
    """
    Write test/data/model/<name>.zip and test/data/dataset/<name>.csv.
    Returns True if written, False if skipped (already exists and not forced).
    """
    zip_path = TEST_MODEL_DIR / f"{name}.zip"
    csv_path = TEST_DATASET_DIR / f"{name}.csv"

    if zip_path.exists() and not force:
        print(f"  skipped (exists): {name}  — pass --force to overwrite")
        return False

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pmml_path, arcname=pmml_path.name)

    write_csv_file(csv_path, fieldnames, rows)
    print(f"  {zip_path.name}  ({len(rows)} rows)")
    print(f"  {csv_path.name}  ({len(fieldnames)} columns: {', '.join(fieldnames[:6])}{'...' if len(fieldnames) > 6 else ''})")
    return True


# ---------------------------------------------------------------------------
# generate subcommand
# ---------------------------------------------------------------------------

def cmd_generate(pmml_path: Path, input_csv_path: Path,
                 name: Optional[str] = None, force: bool = False) -> None:
    """Generate a fixture pair for a single PMML model."""
    name = name or pmml_path.stem
    jar = get_jar()

    primary_col = find_primary_output(pmml_path)
    if primary_col is None:
        sys.exit(f"ERROR: cannot determine primary output field in {pmml_path}")

    input_fields_pmml = get_input_field_names(pmml_path)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_out = Path(tmp.name)

    try:
        run_jpmml(jar, pmml_path, input_csv_path, tmp_out)

        _, input_rows = read_csv_file(input_csv_path)
        _, jpmml_rows = read_csv_file(tmp_out)

        # Use active input fields in PMML order; fall back to all CSV columns
        available = set(input_rows[0].keys()) if input_rows else set()
        input_fields = [f for f in input_fields_pmml if f in available] or list(available)

        fieldnames, rows = build_fixture_csv(input_rows, jpmml_rows, input_fields, primary_col)
        print(f"\nFixture: {name}")
        write_fixture(pmml_path, fieldnames, rows, name, force)

    except RuntimeError as e:
        sys.exit(f"ERROR: jpmml-evaluator failed:\n{e}")
    finally:
        tmp_out.unlink(missing_ok=True)


def cmd_generate_all(force: bool = False) -> None:
    """Regenerate every existing fixture using jpmml-evaluator as oracle."""
    jar = get_jar()
    zips = sorted(TEST_MODEL_DIR.glob("*.zip"))
    if not zips:
        sys.exit(f"No fixtures found in {TEST_MODEL_DIR}")

    ok = failed = skipped = 0

    for zip_path in zips:
        name = zip_path.stem
        csv_path = TEST_DATASET_DIR / f"{name}.csv"
        print(f"\n{name}")

        if not csv_path.exists():
            print("  skipped: no dataset CSV")
            skipped += 1
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Extract PMML from zip
            with zipfile.ZipFile(zip_path) as zf:
                pmml_names = [n for n in zf.namelist()
                               if n.lower().endswith((".pmml", ".xml"))]
                if not pmml_names:
                    print("  skipped: no PMML in zip")
                    skipped += 1
                    continue
                zf.extract(pmml_names[0], tmp)
                pmml_path = tmp / pmml_names[0]

            primary_col = find_primary_output(pmml_path)
            if primary_col is None:
                print("  skipped: cannot determine primary output")
                skipped += 1
                continue

            input_fields_pmml = get_input_field_names(pmml_path)

            _, existing_rows = read_csv_file(csv_path)
            if not existing_rows:
                print("  skipped: empty CSV")
                skipped += 1
                continue

            available = set(existing_rows[0].keys())
            input_fields = [f for f in input_fields_pmml if f in available]
            if not input_fields:
                print("  skipped: no input columns identified")
                skipped += 1
                continue

            # Write input-only CSV for jpmml
            input_csv = tmp / "input.csv"
            write_csv_file(
                input_csv, input_fields,
                [{f: r.get(f, "") for f in input_fields} for r in existing_rows],
            )

            # Run jpmml-evaluator
            out_csv = tmp / "output.csv"
            try:
                run_jpmml(jar, pmml_path, input_csv, out_csv)
            except RuntimeError as e:
                print(f"  FAILED: {e}")
                failed += 1
                continue

            _, jpmml_rows = read_csv_file(out_csv)
            fieldnames, rows = build_fixture_csv(
                existing_rows, jpmml_rows, input_fields, primary_col
            )
            if write_fixture(pmml_path, fieldnames, rows, name, force=True):
                ok += 1
            else:
                skipped += 1

    print(f"\nDone: {ok} updated, {failed} failed, {skipped} skipped")


# ---------------------------------------------------------------------------
# cross-validate subcommand
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path) -> None:
    urllib.request.urlretrieve(url, dest)  # noqa: S310


def _raw_url(repo: str, path: str, ref: str = "master") -> str:
    return f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"


def _find_cpmml_exe() -> Optional[Path]:
    candidates = list((REPO_ROOT / "build").rglob("model_tester.exe"))
    return candidates[0] if candidates else None


def cmd_cross_validate(
    categories: Optional[list[str]],
    cpmml_exe: Optional[Path],
) -> None:
    """
    Download jpmml-evaluator test PMMLs at runtime (AGPL-3.0; not committed),
    score them with both jpmml-evaluator and cPMML, report differences.
    """
    jar = get_jar()

    exe = cpmml_exe or _find_cpmml_exe()
    if exe is None:
        sys.exit(
            "ERROR: model_tester.exe not found.\n"
            "  Build cPMML first (cd build && cmake --build .) or pass --cpmml-exe."
        )

    selected = {k: v for k, v in JPMML_TEST_CATEGORIES.items()
                if categories is None or k in categories}

    passed = failed = skipped = 0
    failures: list[str] = []

    for category, pmml_paths in selected.items():
        print(f"\n{'=' * 60}")
        print(f"  {category}")
        print(f"{'=' * 60}")

        for repo_path in pmml_paths:
            pmml_name = Path(repo_path).stem
            print(f"  {pmml_name:<45} ", end="", flush=True)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                pmml_file = tmp / f"{pmml_name}.pmml"

                # Download PMML from jpmml-evaluator repo (AGPL-3.0)
                url = _raw_url(JPMML_REPO, repo_path)
                try:
                    _download(url, pmml_file)
                except Exception as e:
                    print(f"SKIP  (download: {e})")
                    skipped += 1
                    continue

                primary_col = find_primary_output(pmml_file)
                if primary_col is None:
                    print("SKIP  (no output field)")
                    skipped += 1
                    continue

                input_names = get_input_field_names(pmml_file)

                # Prefer ModelVerification data; fall back to synthetic inputs
                mv = extract_model_verification(pmml_file)
                if mv is not None:
                    input_rows, _ = mv
                    # Keep only input-field keys
                    input_set = set(input_names)
                    input_rows = [{k: v for k, v in r.items() if k in input_set}
                                  for r in input_rows]
                else:
                    input_rows = generate_synthetic_inputs(pmml_file)

                if not input_rows:
                    print("SKIP  (no input data)")
                    skipped += 1
                    continue

                # Reconcile field order: keep PMML order, filter to available
                available = set(input_rows[0].keys())
                input_fields = [f for f in input_names if f in available] or list(available)

                if not input_fields:
                    print("SKIP  (empty input fields)")
                    skipped += 1
                    continue

                # Write input CSV
                input_csv = tmp / "input.csv"
                write_csv_file(
                    input_csv, input_fields,
                    [{f: r.get(f, "") for f in input_fields} for r in input_rows],
                )

                # jpmml-evaluator → expected output
                jpmml_out = tmp / "jpmml_output.csv"
                try:
                    run_jpmml(jar, pmml_file, input_csv, jpmml_out)
                except RuntimeError as e:
                    print(f"SKIP  (jpmml: {e})")
                    skipped += 1
                    continue

                _, jpmml_rows = read_csv_file(jpmml_out)

                # Build fixture for cPMML
                fieldnames, fixture_rows = build_fixture_csv(
                    input_rows, jpmml_rows, input_fields, primary_col
                )
                fixture_csv = tmp / "fixture.csv"
                write_csv_file(fixture_csv, fieldnames, fixture_rows)

                # Zip the PMML for model_tester.exe
                fixture_zip = tmp / f"{pmml_name}.zip"
                with zipfile.ZipFile(fixture_zip, "w") as zf:
                    zf.write(pmml_file, arcname=pmml_file.name)

                # Run cPMML
                result = subprocess.run(
                    [str(exe), str(fixture_zip), str(fixture_csv)],
                    capture_output=True, text=True,
                )

                if result.returncode == 0:
                    print("PASS")
                    passed += 1
                else:
                    label = f"{category}/{pmml_name}"
                    detail = result.stderr.strip()
                    print("FAIL")
                    print(f"    {detail}")
                    failures.append(f"  {label}: {detail}")
                    failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f)
    print()

    sys.exit(1 if failed else 0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="generate_fixture.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # -- generate --
    gen = sub.add_parser(
        "generate",
        help="Create or update a fixture using jpmml-evaluator as oracle",
    )
    gen.add_argument("pmml", nargs="?", help="Path to PMML model file")
    gen.add_argument("input_csv", nargs="?", help="Path to input CSV (features only)")
    gen.add_argument("--name", help="Fixture stem name (default: PMML file stem)")
    gen.add_argument("--all", dest="all_", action="store_true",
                     help="Regenerate all existing fixtures")
    gen.add_argument("--force", action="store_true",
                     help="Overwrite existing fixtures")

    # -- cross-validate --
    cv = sub.add_parser(
        "cross-validate",
        help="Verify cPMML against the jpmml-evaluator test suite",
    )
    cv.add_argument(
        "--category", nargs="+",
        choices=list(JPMML_TEST_CATEGORIES),
        metavar="CAT",
        help=f"Categories to test (default: all). Choices: {', '.join(JPMML_TEST_CATEGORIES)}",
    )
    cv.add_argument(
        "--cpmml-exe", type=Path, metavar="PATH",
        help="Path to model_tester.exe (default: auto-detect in build/)",
    )
    args = parser.parse_args()

    if args.cmd == "generate":
        if args.all_:
            cmd_generate_all(force=args.force)
        elif args.pmml and args.input_csv:
            cmd_generate(
                pmml_path=Path(args.pmml),
                input_csv_path=Path(args.input_csv),
                name=args.name,
                force=args.force,
            )
        else:
            gen.print_help()
            sys.exit(1)

    elif args.cmd == "cross-validate":
        cmd_cross_validate(
            categories=args.category,
            cpmml_exe=args.cpmml_exe,
        )


if __name__ == "__main__":
    main()
