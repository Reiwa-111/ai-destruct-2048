#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/native"
OUTPUT="$ROOT/native/libdestroyer_ntuple.so"
TEMP_OUTPUT="$OUTPUT.tmp.$$"
trap 'rm -f "$TEMP_OUTPUT"' EXIT

g++ -O3 -march=native -DNDEBUG -std=c++17 -fPIC -shared \
  "$ROOT/native/ntuple.cpp" \
  "$ROOT/native/search.cpp" \
  -pthread \
  -o "$TEMP_OUTPUT"

mv -f "$TEMP_OUTPUT" "$OUTPUT"

echo "Built $OUTPUT"
