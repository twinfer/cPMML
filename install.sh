#! /bin/bash
set -euo pipefail

# `cmake -j 4 ..` was configuring, not building: -j is a build-mode flag and
# cmake in configure mode rejects it, so this script never got past its first
# line on a strict shell and silently mis-parsed on a lenient one.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
sudo env "PATH=$PATH" cmake --install build

rm -rf build
if command -v ldconfig > /dev/null; then sudo ldconfig; fi
