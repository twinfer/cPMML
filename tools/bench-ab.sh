#!/bin/bash
# Interleaved A/B performance gate (docs/design/typed-core.md §6.4).
#
#   tools/bench-ab.sh [options] BASE HEAD
#
# BASE and HEAD are each either
#   * a git revision: it is checked out in a detached worktree under --work,
#     this checkout's benchmark/model_benchmark.cc and benchmark/CMakeLists.txt
#     are copied over it (so both sides run the same driver and only the
#     library differs), and model_benchmark.exe is built once per --builds
#     entry: "seq" (-DTHREADS=1) and "omp" (-DTHREADS=4, with
#     -DOpenMP_ROOT=$(brew --prefix libomp) where Homebrew is present). Builds
#     are kept under --work and reused; or
#   * a build directory holding benchmark/model_benchmark.exe (as is, e.g. the
#     working tree's build). Both sides must then be directories; their build
#     is labelled "build" and --builds is not used.
# Running the same build directory as BASE and HEAD is an A/A run.
#
# Options:
#   --builds LIST   seq, omp or seq,omp (revisions only; default seq)
#   --modes LIST    eval, read-all or eval,read-all (default eval,read-all)
#   --rounds N      rounds per model (default 15); NOISY models get 2N+1 (31)
#   --models LIST   comma-separated subset (default: all of benchmark_all)
#   --work DIR      worktrees and builds of revisions
#                   (default ${TMPDIR:-/tmp}/cpmml-bench-ab)
#   --out DIR       raw logs and the report (default WORK/run-<date>)
#   --jobs N        parallel build jobs (default 8)
#
# The model list is benchmark/CMakeLists.txt: every add_model_benchmark(NAME
# DATASET ...) line (benchmark/data/model/NAME.xml over
# benchmark/data/dataset/DATASET.csv) and add_fixture_benchmark(NAME REPEAT
# ...) line (test/data/model/NAME.zip over test/data/dataset/NAME.csv, scored
# REPEAT times). Both sides read the models and datasets of this checkout.
#
# A warm-up round (every model, build, mode and side once, discarded) comes
# first, so that round 1 does not pay for cold caches. Each round then runs,
# for every model and mode, base and head back to back, base first in odd
# rounds and head first in even ones. tools/bench-compare.py then
# applies the gate; a model that fails is re-run once with fresh rounds and
# fails only if it fails again. Output: OUT/raw.txt, OUT/rerun.txt and
# OUT/report.txt. Exit status 0 when the gate passes, 1 when it fails.
#
# The caller's paths do not reach the measured processes (§6.4, "One process
# image"). The initial stack holds argv, the environment and the executable's
# path, so their total length moves everything on the stack; on arm64 macOS
# one 16-byte position of it makes HousingRF*_PCA load 1.33x slower, which
# failed an A/A pair whose build paths differed by 11 characters. Every run
# therefore goes through OUT/run: a copy of each side's binary,
# OUT/run/{base,head}.BUILD/model_benchmark.exe, and two symlinks to the data,
# OUT/run/bm (benchmark/data) and OUT/run/td (test/data). It runs with OUT/run
# as its working directory, relative paths only and an empty environment
# (env -i), so base and head of a build and mode get the same argv but for
# "base"/"head", and none of it depends on BASE, HEAD, OUT, WORK, the
# checkout's path or the caller's environment or working directory. Modes are
# not made equal to each other (--read-all is one more argument): the gate
# never compares eval with read-all.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
BUILDS=seq
MODES=eval,read-all
ROUNDS=15
MODELS=
WORK=${TMPDIR:-/tmp}/cpmml-bench-ab
OUT=
JOBS=8

die() { echo "bench-ab.sh: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
  case $1 in
    --builds) BUILDS=$2; shift 2;;
    --modes) MODES=$2; shift 2;;
    --rounds) ROUNDS=$2; shift 2;;
    --models) MODELS=$2; shift 2;;
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
[[ $ROUNDS =~ ^[1-9][0-9]*$ ]] || die "--rounds needs a positive integer"
for m in ${MODES//,/ }; do case $m in eval|read-all) ;; *) die "unknown mode $m";; esac; done
for b in ${BUILDS//,/ }; do case $b in seq|omp) ;; *) die "unknown build $b";; esac; done
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
OUT=${OUT:-$WORK/run-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"

# ---- the model list --------------------------------------------------------
# Lines of SPEC: "<name> <noisy 0|1> <model_benchmark args...>".
SPEC=$OUT/models.txt
: > "$SPEC"
while read -r kind name arg rest; do
  noisy=0; [[ " $rest " == *" NOISY "* ]] && noisy=1
  case $kind in
    add_model_benchmark)
      [ -f "$REPO/benchmark/data/model/$name.xml" ] && [ -f "$REPO/benchmark/data/dataset/$arg.csv" ] \
        || die "$name: missing benchmark/data/model/$name.xml or benchmark/data/dataset/$arg.csv"
      args="bm/model/$name.xml bm/dataset/$arg.csv";;
    add_fixture_benchmark)
      [ -f "$REPO/test/data/model/$name.zip" ] && [ -f "$REPO/test/data/dataset/$name.csv" ] \
        || die "$name: missing test/data/model/$name.zip or test/data/dataset/$name.csv"
      args="--zip --repeat $arg td/model/$name.zip td/dataset/$name.csv";;
  esac
  if [ -n "$MODELS" ] && [[ ",$MODELS," != *",$name,"* ]]; then continue; fi
  echo "$name $noisy $args" >> "$SPEC"
done < <(sed -n -E 's/^(add_model_benchmark|add_fixture_benchmark)\(([^)]*)\).*/\1 \2/p' "$REPO/benchmark/CMakeLists.txt")
if [ -n "$MODELS" ]; then
  for m in ${MODELS//,/ }; do grep -q "^$m " "$SPEC" || die "unknown model $m"; done
fi
[ -s "$SPEC" ] || die "no models selected"

# ---- the binaries ----------------------------------------------------------
# Worktrees are removed on exit (the builds stay under WORK for the next run).
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
# shellcheck disable=SC2329  # invoked through eval below
build_rev() {  # REV BUILD -> prints the build directory
  local rev=$1 build=$2 sha src dir threads extra=()
  sha=$(git -C "$REPO" rev-parse --verify --quiet "$rev^{commit}") || die "not a revision or build directory: $rev"
  sha=${sha:0:12}
  src=$WORK/src-$sha
  if [ ! -d "$src" ]; then
    git -C "$REPO" worktree add --quiet --detach "$src" "$sha" >&2 || die "cannot check out $rev in $src"
  fi
  echo "$src" >> "$WORKTREES"
  for f in benchmark/model_benchmark.cc benchmark/CMakeLists.txt; do
    cmp -s "$REPO/$f" "$src/$f" || cp "$REPO/$f" "$src/$f"
  done
  dir=$WORK/build-$sha-$build
  threads=1
  if [ "$build" = omp ]; then
    threads=4
    if command -v brew > /dev/null && brew --prefix libomp > /dev/null 2>&1; then
      extra=(-DOpenMP_ROOT="$(brew --prefix libomp)")
    fi
  fi
  if [ ! -f "$dir/CMakeCache.txt" ]; then
    cmake -S "$src" -B "$dir" -DCMAKE_BUILD_TYPE=Release -DTHREADS="$threads" ${extra[@]+"${extra[@]}"} \
      > "$dir.configure.log" 2>&1 || die "configure of $rev ($build) failed: $dir.configure.log"
  fi
  cmake --build "$dir" --target model_benchmark.exe -j "$JOBS" > "$dir.build.log" 2>&1 \
    || die "build of $rev ($build) failed: $dir.build.log"
  if [ "$build" = omp ] && ! grep -q -- '-DMULTITHREADING' "$dir/benchmark/CMakeFiles/model_benchmark.exe.dir/flags.make" 2>/dev/null; then
    die "$rev ($build): OpenMP was not found, the build is sequential ($dir.configure.log)"
  fi
  echo "$dir"
}

if [ -d "$BASE" ] || [ -d "$HEAD" ]; then
  [ -d "$BASE" ] && [ -d "$HEAD" ] || die "BASE and HEAD must both be build directories or both revisions"
  BUILDS=build
  # shellcheck disable=SC2034  # read through eval as BASE_$b / HEAD_$b
  BASE_build=$(cd "$BASE" && pwd)
  # shellcheck disable=SC2034
  HEAD_build=$(cd "$HEAD" && pwd)
else
  for b in ${BUILDS//,/ }; do
    eval "BASE_$b=\$(build_rev \"\$BASE\" $b)"
    eval "HEAD_$b=\$(build_rev \"\$HEAD\" $b)"
  done
fi
for b in ${BUILDS//,/ }; do
  for s in BASE HEAD; do
    d=
    eval "d=\$${s}_$b"
    [ -x "$d/benchmark/model_benchmark.exe" ] || die "no benchmark/model_benchmark.exe in $d"
  done
done

# ---- the run directory (see the header: the caller's paths stay out) -------
RUN=$OUT/run
rm -rf "$RUN"
mkdir -p "$RUN"
ln -s "$REPO/benchmark/data" "$RUN/bm"
ln -s "$REPO/test/data" "$RUN/td"
# Copies, not symlinks: macOS also puts the executable's inode on the stack, in
# hex ("executable_file=0x<dev>,0x<inode>"), so all copies are made together
# and remade until their inodes have the same number of hex digits. The copy
# still loads its build directory's library (absolute build-tree rpath).
copy_binaries() {
  local b s d
  for b in ${BUILDS//,/ }; do
    for s in base head; do
      if [ "$s" = base ]; then eval "d=\$BASE_$b"; else eval "d=\$HEAD_$b"; fi
      mkdir -p "$RUN/$s.$b"
      rm -f "$RUN/$s.$b/model_benchmark.exe"
      cp "$d/benchmark/model_benchmark.exe" "$RUN/$s.$b/model_benchmark.exe"
    done
  done
}
inode_digits() {  # the distinct hex lengths of the copies' inodes
  local f
  for f in "$RUN"/*.*/model_benchmark.exe; do
    # shellcheck disable=SC2012  # our own file names; ls -i is portable
    printf '%x\n' "$(ls -i "$f" | awk '{print $1}')" | awk '{print length($0)}'
  done | sort -u | wc -l | tr -d ' '
}
tries=0
copy_binaries
while [ "$(inode_digits)" != 1 ]; do
  tries=$((tries + 1))
  [ $tries -le 20 ] || die "cannot give the binaries in $RUN inodes of one hex length"
  copy_binaries
done

{
  for s in BASE HEAD; do
    v=; eval "v=\$$s"
    sha=
    [ -d "$v" ] || sha=" ($(git -C "$REPO" rev-parse --short=12 "$v^{commit}"), driver from $REPO)"
    echo "$(echo "$s" | tr "[:upper:]" "[:lower:]"): $v$sha"
  done
  for b in ${BUILDS//,/ }; do eval "echo \"  $b: base \$BASE_$b, head \$HEAD_$b\""; done
  echo "modes: $MODES; rounds: $ROUNDS (NOISY: $((2 * ROUNDS + 1)))"
  echo "run: ./{base,head}.BUILD/model_benchmark.exe in $RUN, relative data paths, env -i"
  echo "host: $(uname -srm)"
} > "$OUT/setup.txt"

# ---- the rounds ------------------------------------------------------------
# run_rounds LOG FIRST LAST [FILTER] : rounds FIRST..LAST (NOISY models go to
# FIRST + 2*(LAST-FIRST+1)), restricted to FILTER lines "<build> <mode> <model>".
run_one() {  # BUILD MODE ROUND SIDE NAME ARGS...
  local build=$1 mode=$2 round=$3 side=$4 name=$5 out flag=()
  shift 5
  [ "$mode" = read-all ] && flag=(--read-all)
  # Same length for base and head: ./base.B or ./head.B, relative data paths,
  # cwd OUT/run, no environment.
  out=$(cd "$RUN" && env -i "./$side.$build/model_benchmark.exe" ${flag[@]+"${flag[@]}"} "$@" 2>&1) \
    || die "$name ($build $mode $side) failed: $out"
  [[ $out == *"Evaluate time:"* ]] || die "$name ($build $mode $side): unexpected output: $out"
  echo "$build $mode $round $side $name $out"
}
warm_up() {  # one discarded run of every model, build, mode and side
  local name noisy args b m
  while read -r name noisy args; do
    for b in ${BUILDS//,/ }; do
      for m in ${MODES//,/ }; do
        # shellcheck disable=SC2086
        run_one "$b" "$m" 0 base "$name" $args > /dev/null
        # shellcheck disable=SC2086
        run_one "$b" "$m" 0 head "$name" $args > /dev/null
      done
    done
  done < "$SPEC"
  echo "warm-up done" >&2
}
run_rounds() {
  local log=$1 first=$2 last=$3 filter=${4:-} r name noisy args top n=$(( $3 - $2 + 1 ))
  top=$(( first + 2 * n ))
  for (( r = first; r <= top; r++ )); do
    while read -r name noisy args; do
      if [ "$noisy" = 1 ]; then [ "$r" -le "$top" ] || continue; else [ "$r" -le "$last" ] || continue; fi
      for b in ${BUILDS//,/ }; do
        for m in ${MODES//,/ }; do
          [ -z "$filter" ] || grep -qx "$b $m $name" "$filter" || continue
          # shellcheck disable=SC2086
          if (( r % 2 )); then
            run_one "$b" "$m" "$r" base "$name" $args >> "$log"
            run_one "$b" "$m" "$r" head "$name" $args >> "$log"
          else
            run_one "$b" "$m" "$r" head "$name" $args >> "$log"
            run_one "$b" "$m" "$r" base "$name" $args >> "$log"
          fi
        done
      done
    done < "$SPEC"
    echo "round $r/$top done" >&2
  done
}

: > "$OUT/raw.txt"
warm_up
run_rounds "$OUT/raw.txt" 1 "$ROUNDS"
status=0
python3 "$REPO/tools/bench-compare.py" "$OUT/raw.txt" --fail-list "$OUT/failed.txt" > "$OUT/first-report.txt" || status=$?
[ $status -le 1 ] || die "bench-compare.py failed on $OUT/raw.txt"
if [ -s "$OUT/failed.txt" ]; then
  echo "re-running $(wc -l < "$OUT/failed.txt" | tr -d ' ') failing model(s) with fresh rounds" >&2
  : > "$OUT/rerun.txt"
  run_rounds "$OUT/rerun.txt" 1 "$ROUNDS" "$OUT/failed.txt"
  status=0
  python3 "$REPO/tools/bench-compare.py" "$OUT/raw.txt" --rerun "$OUT/rerun.txt" > "$OUT/report.txt" || status=$?
else
  cp "$OUT/first-report.txt" "$OUT/report.txt"
fi
cat "$OUT/setup.txt" "$OUT/report.txt"
exit $status
