#!/bin/bash
# Compile the sketch with the repository's flags on each available compiler.
D=$(cd "$(dirname "$0")" && pwd); R=$(git -C "$D" rev-parse --show-toplevel)
FLAGS="-std=c++20 -Wall -O3 -I$R/src"
run() { # name compiler extra
  echo "== $1"; $2 $FLAGS $3 $D/typed_value_test.cc -o $D/t_$1 && ( [ -n "$4" ] && $4 $D/t_$1 || $D/t_$1 )
  for n in neg_missing_case neg_uncategorised; do
    if $2 $FLAGS -fsyntax-only $D/$n.cc 2>$D/$1_$n.err; then echo "$n: COMPILED (unexpected)"; else echo "$n: rejected as expected ($(grep -m1 -o 'error: .*' $D/$1_$n.err | cut -c1-110))"; fi
  done
}
run clang /usr/bin/c++ "-arch arm64"
echo "== clang with the P8 temporal alternatives"; /usr/bin/c++ $FLAGS -DPMML_WITH_TEMPORAL=1 $D/typed_value_test.cc -o $D/t_temporal && $D/t_temporal | grep -E "sizeof|passed"
command -v g++-16 >/dev/null && run gcc g++-16 ""
command -v em++ >/dev/null && run emcc em++ "-fexceptions -sALLOW_MEMORY_GROWTH=1" node
