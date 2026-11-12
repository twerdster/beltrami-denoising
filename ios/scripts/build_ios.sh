#!/usr/bin/env bash
set -euo pipefail

BELTRAMI_REPOSITORY="$(cd "$(dirname "$0")/../.." && pwd)"
BELTRAMI_BUILD="${BELTRAMI_IOS_BUILD_DIR:-$BELTRAMI_REPOSITORY/.build-ios}"
BELTRAMI_PYTHON_COMMAND="${BELTRAMI_PYTHON:-python3}"
BELTRAMI_FIXTURES=false
BELTRAMI_FLAT_FIXTURES=""

usage() {
  cat <<'USAGE'
Usage: ios/scripts/build_ios.sh [--fixtures | --flat-fixtures DIRECTORY]

Build the Release arm64 iPhone app with Xcode. TEAM_ID signs for a physical
device; without it, build an unsigned app for compilation checks only.

  --fixtures                 Generate the 18 procedural Python references.
  --flat-fixtures DIRECTORY  Adapt existing flat references (standard library only).
  --help                     Show this help.

Optional environment: TEAM_ID, BUNDLE_ID, BELTRAMI_PYTHON,
BELTRAMI_IOS_BUILD_DIR (default: repository/.build-ios).
USAGE
}

while (($#)); do
  case "$1" in
    --fixtures) BELTRAMI_FIXTURES=true; shift ;;
    --flat-fixtures)
      if (($# < 2)); then usage >&2; exit 2; fi
      BELTRAMI_FLAT_FIXTURES="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if $BELTRAMI_FIXTURES && [[ -n "$BELTRAMI_FLAT_FIXTURES" ]]; then
  printf 'Choose either --fixtures or --flat-fixtures.\n' >&2
  exit 2
fi
if [[ -n "$BELTRAMI_FLAT_FIXTURES" ]]; then
  "$BELTRAMI_PYTHON_COMMAND" "$BELTRAMI_REPOSITORY/ios/scripts/prepare_fixtures.py" \
    --flat-dir "$BELTRAMI_FLAT_FIXTURES"
elif $BELTRAMI_FIXTURES; then
  "$BELTRAMI_PYTHON_COMMAND" "$BELTRAMI_REPOSITORY/ios/scripts/prepare_fixtures.py"
fi

BELTRAMI_SETTINGS=()
if [[ -n "${TEAM_ID:-}" ]]; then
  BELTRAMI_SETTINGS+=("DEVELOPMENT_TEAM=$TEAM_ID" "CODE_SIGN_STYLE=Automatic" -allowProvisioningUpdates)
else
  BELTRAMI_SETTINGS+=(CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO "CODE_SIGN_IDENTITY=")
fi
if [[ -n "${BUNDLE_ID:-}" ]]; then
  BELTRAMI_SETTINGS+=("PRODUCT_BUNDLE_IDENTIFIER=$BUNDLE_ID")
fi

xcodebuild -project "$BELTRAMI_REPOSITORY/ios/BeltramiApp.xcodeproj" \
  -target BeltramiApp -configuration Release -sdk iphoneos -arch arm64 \
  "SYMROOT=$BELTRAMI_BUILD/Build/Products" "OBJROOT=$BELTRAMI_BUILD/Build/Intermediates.noindex" \
  "${BELTRAMI_SETTINGS[@]}" build
printf '\nBuilt app: %s/Build/Products/Release-iphoneos/BeltramiApp.app\n' "$BELTRAMI_BUILD"
if [[ -z "${TEAM_ID:-}" ]]; then
  printf 'Unsigned compilation check. Set TEAM_ID to sign for a physical device.\n'
fi
