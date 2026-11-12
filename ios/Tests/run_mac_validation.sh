#!/usr/bin/env bash
set -euo pipefail

BELTRAMI_REPOSITORY="$(cd "$(dirname "$0")/../.." && pwd)"
BELTRAMI_BUILD="${BELTRAMI_METAL_BUILD_DIR:-${TMPDIR:-/tmp}/beltrami-metal-validation}"
BELTRAMI_PYTHON_COMMAND="${BELTRAMI_PYTHON:-python3}"
mkdir -p "$BELTRAMI_BUILD"

"$BELTRAMI_PYTHON_COMMAND" "$BELTRAMI_REPOSITORY/ios/Tests/generate_fixtures.py" "$BELTRAMI_BUILD/fixtures"
xcrun -sdk macosx metal -fno-fast-math -c \
  "$BELTRAMI_REPOSITORY/ios/BeltramiMetal/BeltramiKernels.metal" -o "$BELTRAMI_BUILD/Beltrami.air"
xcrun -sdk macosx metallib "$BELTRAMI_BUILD/Beltrami.air" -o "$BELTRAMI_BUILD/Beltrami.metallib"
xcrun swiftc -O \
  "$BELTRAMI_REPOSITORY/ios/BeltramiMetal/BeltramiMetal.swift" \
  "$BELTRAMI_REPOSITORY/ios/Tests/main.swift" -o "$BELTRAMI_BUILD/validate"
"$BELTRAMI_BUILD/validate" --library "$BELTRAMI_BUILD/Beltrami.metallib" \
  --fixtures "$BELTRAMI_BUILD/fixtures" --benchmarks --output "$BELTRAMI_BUILD/report.json"
printf 'Validation report: %s\n' "$BELTRAMI_BUILD/report.json"
