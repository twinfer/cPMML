#!/usr/bin/env bash
# cross-validate.sh — cross-validate cPMML against jpmml-evaluator
#
# Clones jpmml-evaluator at tag 1.7.7, uses the jpmml API to score each test
# PMML with generated inputs, then checks cPMML produces matching output.
#
# Usage:
#   tools/cross-validate.sh [--cpmml-exe PATH] [--output-dir DIR]
#
# Requires: java (11+), git (2.25+)
#
# This compares numbers within a tolerance on JPMML's own test models. To
# compare cPMML's fixture suite with JPMML as exact text (the predicted value
# and every OutputField), use tools/cross-validate-text.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

JAR_URL="https://github.com/jpmml/jpmml-evaluator/releases/download/1.7.7/pmml-evaluator-example-executable-1.7.7.jar"
JAR="$SCRIPT_DIR/pmml-evaluator-example-executable-1.7.7.jar"
CV_SRC="$SCRIPT_DIR/fixture-generator/CrossValidator.java"
CV_CLASS_DIR="$SCRIPT_DIR/fixture-generator/classes"

JPMML_REPO_URI="https://github.com/jpmml/jpmml-evaluator.git"
JPMML_TAG="1.7.7"
JPMML_PMML_PATH="pmml-evaluator/src/test/resources/pmml"

CPMML_EXE=""
USER_OUTPUT_DIR=""

usage() {
    echo "Usage: cross-validate.sh [--cpmml-exe PATH] [--output-dir DIR]"
    echo "  --cpmml-exe PATH   path to pmml_tester.exe (auto-detected from build/ if omitted)"
    echo "  --output-dir DIR   keep generated fixtures here for inspection (default: temp, deleted on exit)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpmml-exe)  CPMML_EXE="$2"; shift 2 ;;
        --output-dir) USER_OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# --------------------------------------------------------------------------
# 1. Download pmml-evaluator-example jar (oracle + compile classpath)
# --------------------------------------------------------------------------
if [[ ! -f "$JAR" ]]; then
    echo "Downloading pmml-evaluator-example jar..."
    curl -fsSL -o "$JAR" "$JAR_URL"
    echo ""
fi

# --------------------------------------------------------------------------
# 2. Compile CrossValidator.java (skip if class is newer than source)
# --------------------------------------------------------------------------
CV_CLASS="$CV_CLASS_DIR/org/cpmml/CrossValidator.class"
if [[ ! -f "$CV_CLASS" || "$CV_SRC" -nt "$CV_CLASS" ]]; then
    echo "Compiling CrossValidator.java..."
    mkdir -p "$CV_CLASS_DIR"
    javac -cp "$JAR" -d "$CV_CLASS_DIR" "$CV_SRC"
    echo ""
fi

# --------------------------------------------------------------------------
# 3. Find cPMML executable
# --------------------------------------------------------------------------
if [[ -z "$CPMML_EXE" ]]; then
    # Prefer Release build, then RelWithDebInfo, then any match
    for variant in Release RelWithDebInfo Debug ""; do
        CPMML_EXE=$(find "$REPO_ROOT/build" -path "*${variant}*" -name "pmml_tester.exe" 2>/dev/null | head -1 || true)
        [[ -n "$CPMML_EXE" ]] && break
    done
    if [[ -z "$CPMML_EXE" ]]; then
        echo "ERROR: pmml_tester.exe not found. Build cPMML or pass --cpmml-exe."
        exit 1
    fi
    echo "Using cPMML executable: $CPMML_EXE"
    echo ""
fi

# --------------------------------------------------------------------------
# 4. Sparse-clone jpmml-evaluator test PMMls
# --------------------------------------------------------------------------
TMP_REPO=$(mktemp -d)

if [[ -n "$USER_OUTPUT_DIR" ]]; then
    FIXTURES="$USER_OUTPUT_DIR"
    mkdir -p "$FIXTURES"
    trap 'rm -rf "$TMP_REPO"' EXIT
    echo "Fixtures will be written to: $FIXTURES"
    echo ""
else
    FIXTURES=$(mktemp -d)
    trap 'rm -rf "$TMP_REPO" "$FIXTURES"' EXIT
fi

echo "Fetching jpmml-evaluator test PMMls (tag $JPMML_TAG)..."
if ! git clone --depth=1 --filter=blob:none --no-checkout --branch "$JPMML_TAG" \
    "$JPMML_REPO_URI" "$TMP_REPO" 2>&1 | tail -1; then
    echo "ERROR: Failed to clone jpmml-evaluator at tag $JPMML_TAG"
    exit 1
fi
git -C "$TMP_REPO" config core.sparseCheckout true
mkdir -p "$TMP_REPO/.git/info"
printf '%s/\n' "$JPMML_PMML_PATH" > "$TMP_REPO/.git/info/sparse-checkout"
git -C "$TMP_REPO" checkout HEAD 2>&1 | tail -1
echo ""

JPMML_PMML_DIR="$TMP_REPO/$JPMML_PMML_PATH"

if [[ ! -d "$JPMML_PMML_DIR" ]] || [[ -z "$(ls "$JPMML_PMML_DIR" 2>/dev/null)" ]]; then
    echo "ERROR: No PMML files found at $JPMML_PMML_DIR"
    echo "  The sparse checkout path '$JPMML_PMML_PATH' may have changed upstream."
    exit 1
fi

# --------------------------------------------------------------------------
# 5. Run CrossValidator: load each PMML via jpmml, generate inputs, score
#    → writes <stem>.zip + <stem>.csv into FIXTURES dir
# --------------------------------------------------------------------------
java -cp "$JAR:$CV_CLASS_DIR" org.cpmml.CrossValidator \
    --jpmml-pmml-dir "$JPMML_PMML_DIR" \
    --output-dir     "$FIXTURES"
echo ""

FIXTURE_COUNT=$(find "$FIXTURES" -name "*.zip" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$FIXTURE_COUNT" -eq 0 ]]; then
    echo "ERROR: CrossValidator generated 0 fixtures — nothing to test."
    exit 1
fi

# --------------------------------------------------------------------------
# 6. Run cPMML against each generated fixture pair
# --------------------------------------------------------------------------
PASSED=0
FAILED=0
SKIPPED=0
FAILURES=()

SEP=$(printf '%0.s=' {1..60})
echo "$SEP"

for ZIP in "$FIXTURES"/*.zip; do
    [[ -e "$ZIP" ]] || { echo "No fixtures generated."; break; }
    STEM=$(basename "$ZIP" .zip)
    CSV="$FIXTURES/$STEM.csv"

    printf "  %-50s " "$STEM"

    if [[ ! -f "$CSV" ]]; then
        echo "SKIP (no CSV)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    RC=0
    OUTPUT=$("$CPMML_EXE" "$ZIP" "$CSV" 2>&1) || RC=$?
    if [[ $RC -eq 0 ]]; then
        echo "PASS"
        PASSED=$((PASSED + 1))
    else
        if [[ $RC -gt 128 && $RC -lt 255 ]]; then
            SIG=$((RC - 128))
            echo "FAIL (signal $SIG)"
        else
            echo "FAIL (exit $RC)"
        fi
        [[ -n "$OUTPUT" ]] && echo "      $OUTPUT"
        FAILURES+=("$STEM")
        FAILED=$((FAILED + 1))
    fi
done

echo "$SEP"
echo "Results: $PASSED passed, $FAILED failed, $SKIPPED skipped"

if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo ""
    echo "Failures:"
    for f in "${FAILURES[@]}"; do echo "  $f"; done
fi
echo ""

[[ $FAILED -eq 0 ]]
