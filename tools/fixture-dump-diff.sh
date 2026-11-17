#!/bin/bash
# Fixture dump diff (docs/design/typed-core.md §7): every output cell of every
# fixture, as text, from two builds, and the cells that differ.
#
#   tools/fixture-dump-diff.sh [options] BASE HEAD
#
# BASE and HEAD are each either
#   * a build directory holding test/pmml_tester.exe (as is); or
#   * a git revision: it is checked out in a detached worktree under --work
#     and test/pmml_tester.exe is built there (Release). By default this
#     checkout's test/pmml_tester.cc is copied over the revision's, so both
#     sides dump through the same driver and only the library differs (as
#     tools/bench-ab.sh does with the benchmark driver); --tester own keeps
#     the revision's own tester. Builds are kept under --work and reused;
#     worktrees are removed on exit.
# Both sides score the same fixtures, --fixtures (default: this checkout's
# test/data), with `pmml_tester.exe <zip> <inputs> --dump <csv>`: the
# predicted value and every OutputField of every row, as JPMML would write
# them, "N/A" for no value, "ERROR: ..." for a row that throws, and for a
# time-series fixture the forecast (and variance) of every step. A model that
# does not load is one cell, "LOAD-ERROR: ...". tools/fixture-dump-cells.py
# writes them as OUT/{base,head}/<fixture>.cells (<row> TAB <column> TAB
# <text>) and lists every cell whose text differs:
#   <fixture> TAB row <n> TAB <column> TAB <base text> TAB -> TAB <head text>
# then a summary line. A refactoring commit shows no cell; a fix commit shows
# only the cells its message lists.
#
# Options:
#   --fixtures DIR|REV  the fixture tree (a directory holding model/ and
#                       dataset/, or a revision whose test/data is used;
#                       default: this checkout's test/data)
#   --tester MODE       checkout (default) or own; revisions only
#   --only NAMES        comma-separated subset of fixtures
#   --work DIR          worktrees and builds of revisions
#                       (default ${TMPDIR:-/tmp}/cpmml-dump-diff)
#   --out DIR           the dumps and the report (default WORK/run-<date>)
#   --jobs N            parallel build and scoring jobs (default 8)
#
# Exit status 0 when no cell differs, 1 when some do, 2 on a usage or build
# error. Requires cmake, git and python3 (standard library only).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
FIXTURES=$REPO/test/data
TESTER=checkout
ONLY=
WORK=${TMPDIR:-/tmp}/cpmml-dump-diff
OUT=
JOBS=8

die() { echo "fixture-dump-diff.sh: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
  case $1 in
    --fixtures) FIXTURES=$2; shift 2;;
    --tester) TESTER=$2; shift 2;;
    --only) ONLY=$2; shift 2;;
    --work) WORK=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --jobs) JOBS=$2; shift 2;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0;;
    --*) die "unknown option $1";;
    *) break;;
  esac
done
[ $# -eq 2 ] || die "expected BASE and HEAD (see --help)"
BASE=$1; HEAD=$2
case $TESTER in checkout|own) ;; *) die "--tester is checkout or own";; esac
[[ $JOBS =~ ^[1-9][0-9]*$ ]] || die "--jobs needs a positive integer"
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
OUT=${OUT:-$WORK/run-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)

WORKTREES=$OUT/worktrees.txt
: > "$WORKTREES"
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() {
  local src
  while read -r src; do
    git -C "$REPO" worktree remove --force "$src" > /dev/null 2>&1 || true
  done < "$WORKTREES"
  git -C "$REPO" worktree prune
}
trap cleanup EXIT

resolve_rev() {  # REV -> 12-digit sha, or dies
  local sha
  sha=$(git -C "$REPO" rev-parse --verify --quiet "$1^{commit}") || die "not a revision or directory: $1"
  echo "${sha:0:12}"
}

build_rev() {  # REV -> prints the build directory
  local rev=$1 sha src dir
  sha=$(resolve_rev "$rev") || exit 2
  src=$WORK/src-$sha
  if [ ! -d "$src" ]; then
    git -C "$REPO" worktree add --quiet --detach "$src" "$sha" >&2 || die "cannot check out $rev in $src"
  fi
  echo "$src" >> "$WORKTREES"
  if [ "$TESTER" = checkout ]; then
    cmp -s "$REPO/test/pmml_tester.cc" "$src/test/pmml_tester.cc" \
      || cp "$REPO/test/pmml_tester.cc" "$src/test/pmml_tester.cc"
  else
    git -C "$src" checkout --quiet -- test/pmml_tester.cc
  fi
  dir=$WORK/build-$sha-$TESTER
  if [ ! -f "$dir/CMakeCache.txt" ]; then
    cmake -S "$src" -B "$dir" -DCMAKE_BUILD_TYPE=Release > "$dir.configure.log" 2>&1 \
      || die "configure of $rev failed: $dir.configure.log"
  fi
  local hint=
  [ "$TESTER" = checkout ] && hint=" (this checkout's tester may not build against $rev: try --tester own)"
  cmake --build "$dir" --target pmml_tester.exe -j "$JOBS" > "$dir.build.log" 2>&1 \
    || die "build of $rev failed: $dir.build.log$hint"
  echo "$dir"
}

side() {  # BASE|HEAD argument -> prints the build directory
  if [ -d "$1" ]; then
    (cd "$1" && pwd)
  else
    build_rev "$1" || exit 2
  fi
}
# A die inside $(...) ends only the subshell (bash 3.2 has no inherit_errexit).
BASE_BUILD=$(side "$BASE") || exit 2
HEAD_BUILD=$(side "$HEAD") || exit 2
for d in "$BASE_BUILD" "$HEAD_BUILD"; do
  [ -x "$d/test/pmml_tester.exe" ] || die "no test/pmml_tester.exe in $d"
done

if [ ! -d "$FIXTURES" ]; then
  sha=$(resolve_rev "$FIXTURES") || exit 2
  dir=$WORK/fixtures-$sha
  if [ ! -d "$dir/test/data" ]; then
    mkdir -p "$dir"
    git -C "$REPO" archive "$sha" test/data | tar -x -C "$dir" || die "cannot extract test/data of $FIXTURES"
  fi
  FIXTURES=$dir/test/data
fi
FIXTURES=$(cd "$FIXTURES" && pwd)
[ -d "$FIXTURES/model" ] && [ -d "$FIXTURES/dataset" ] || die "no model/ and dataset/ in $FIXTURES"

only=()
if [ -n "$ONLY" ]; then
  IFS=, read -ra names <<< "$ONLY"
  only=(--only "${names[@]}")
fi
CELLS=$REPO/tools/fixture-dump-cells.py
{
  echo "base: $BASE ($BASE_BUILD)"
  echo "head: $HEAD ($HEAD_BUILD)"
  echo "fixtures: $FIXTURES"
  [ -d "$BASE" ] && [ -d "$HEAD" ] || echo "tester: $TESTER"
} > "$OUT/report.txt"
python3 "$CELLS" dump --exe "$BASE_BUILD/test/pmml_tester.exe" --fixtures "$FIXTURES" --out "$OUT/base" \
  -j "$JOBS" ${only[@]+"${only[@]}"} > /dev/null
python3 "$CELLS" dump --exe "$HEAD_BUILD/test/pmml_tester.exe" --fixtures "$FIXTURES" --out "$OUT/head" \
  -j "$JOBS" ${only[@]+"${only[@]}"} > /dev/null
status=0
python3 "$CELLS" diff "$OUT/base" "$OUT/head" >> "$OUT/report.txt" || status=$?
cat "$OUT/report.txt"
echo "report: $OUT/report.txt"
[ "$status" -le 1 ] || die "diff failed"
exit "$status"
