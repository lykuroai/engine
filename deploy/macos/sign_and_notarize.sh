#!/usr/bin/env bash
# macOS Developer ID signing + notarization + stapling for a staged
# Lykuro Native Inference Engine package (spec/addendum §34: Developer ID
# sign, notarization, Hardened Runtime).
#
#   deploy/macos/sign_and_notarize.sh <staged_dir> [out_dir]
#
# <staged_dir> is the tree produced by tools/make_package.sh for the
# macos-metal profile (contains bin/, config/, launchd/, ...).
#
# Pipeline:
#   1. codesign every Mach-O in bin/ with Developer ID Application +
#      Hardened Runtime (--options runtime) + secure timestamp +
#      deploy/macos/entitlements.plist.
#   2. pkgbuild a component .pkg and sign it with Developer ID Installer.
#   3. notarize the .pkg via `xcrun notarytool submit --wait`.
#   4. staple the ticket and verify with spctl (Gatekeeper).
#
# Credentials are read from the environment so no secret is ever written
# to the repo:
#   DEVELOPER_ID_APP        e.g. "Developer ID Application: e-Business Solutions Inc. (TEAMID)"
#   DEVELOPER_ID_INSTALLER  e.g. "Developer ID Installer: e-Business Solutions Inc. (TEAMID)"
#   PKG_IDENTIFIER          default ai.lykuro.native-engine
#   Notarization — EITHER a stored keychain profile:
#     NOTARY_PROFILE        keychain profile name (xcrun notarytool store-credentials)
#   OR explicit App Store Connect credentials:
#     NOTARY_APPLE_ID, NOTARY_TEAM_ID, NOTARY_PASSWORD (app-specific password)
#
# FAIL-SAFE: if DEVELOPER_ID_APP is unset the script prints a notice and
# exits 0 without producing a signed artifact, so it can be wired into an
# unsigned dev/CI build harmlessly. It never fabricates a signature.
set -euo pipefail

CODESIGN_ONLY=0
DEV_MODE=0
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --codesign-only) CODESIGN_ONLY=1; shift;;
    --dev) DEV_MODE=1; shift;;      # ad-hoc sign, no Developer Program
    *) echo "unknown flag: $1" >&2; exit 2;;
  esac
done

STAGE="${1:?staged package dir (from make_package.sh)}"
OUT_DIR="${2:-$(dirname "$STAGE")}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENTITLEMENTS="$ROOT/deploy/macos/entitlements.plist"
PKG_IDENTIFIER="${PKG_IDENTIFIER:-ai.lykuro.native-engine}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: Developer ID signing must run on macOS" >&2; exit 2
fi
[[ -d "$STAGE/bin" ]] || { echo "ERROR: $STAGE has no bin/ (not a staged package?)" >&2; exit 2; }

# Identity resolution — three postures:
#   Developer ID set        -> full sign + notarize + staple (distribution)
#   --dev (no Developer ID) -> ad-hoc sign + Hardened Runtime (internal test)
#   neither                 -> no-op exit 0 (leave the tree unsigned)
if [[ -n "${DEVELOPER_ID_APP:-}" ]]; then
  SIGN_IDENTITY="$DEVELOPER_ID_APP"
elif [[ "$DEV_MODE" -eq 1 ]]; then
  SIGN_IDENTITY="-"                 # ad-hoc
  echo "NOTICE: --dev: ad-hoc signing (Phase 1, no Apple Developer Program)."
else
  cat >&2 <<'EOF'
NOTICE: no signing identity -> SKIPPED (tree left unsigned).
        Phase 1 (internal test, no Developer Program): re-run with --dev
          for an ad-hoc-signed package.
        Phase 2 (downloads.lykuro.ai): export the Developer ID identities
          and notary credentials (see this script's header) and re-run.
EOF
  exit 0
fi
# Phase 1 (ad-hoc) needs library validation disabled so the process can
# load the bundled dylibs (ad-hoc has no Team ID for LV to match). Phase 2
# signs every bundled dylib with the same Developer ID, so LV passes with
# the strict empty entitlements.
if [[ "$DEV_MODE" -eq 1 ]]; then
  ENTITLEMENTS="$ROOT/deploy/macos/entitlements.dev.plist"
fi
command -v codesign >/dev/null || { echo "ERROR: codesign not found" >&2; exit 2; }
command -v pkgbuild >/dev/null || { echo "ERROR: pkgbuild not found" >&2; exit 2; }
command -v xcrun >/dev/null || { echo "ERROR: xcrun not found" >&2; exit 2; }
[[ -f "$ENTITLEMENTS" ]] || { echo "ERROR: entitlements not found: $ENTITLEMENTS" >&2; exit 2; }

VERSION_BASENAME="$(basename "$STAGE")"          # lykuro-native-engine-macos-metal-<ver>
VERSION="${VERSION_BASENAME##*-}"

# Ad-hoc signatures cannot carry an Apple secure timestamp.
TS_FLAG=(--timestamp)
[[ "$SIGN_IDENTITY" == "-" ]] && TS_FLAG=(--timestamp=none)

echo "==> 1/4 codesign Mach-O binaries (Hardened Runtime)"
sign_one() {
  local f="$1"
  file "$f" | grep -q 'Mach-O' || return 0
  echo "    signing $f"
  codesign --force "${TS_FLAG[@]}" --options runtime \
    --entitlements "$ENTITLEMENTS" --sign "$SIGN_IDENTITY" "$f"
  codesign --verify --strict "$f"
}
# Bundled dylibs first (inner code), then the executables (outer).
if [[ -d "$STAGE/lib" ]]; then
  while IFS= read -r lib; do sign_one "$lib"; done \
    < <(find "$STAGE/lib" -type f)
fi
while IFS= read -r bin; do sign_one "$bin"; done \
  < <(find "$STAGE/bin" -type f)

if [[ "$CODESIGN_ONLY" -eq 1 ]]; then
  echo "OK: binaries codesigned (--codesign-only); skipping pkg/notarize"
  exit 0
fi

if [[ "$DEV_MODE" -eq 1 ]]; then
  # Phase 1: ad-hoc component .pkg for internal distribution. No Developer
  # ID, so no productsign/notarize/staple. Gatekeeper will quarantine a
  # downloaded copy; internal testers clear it once per host.
  DEVPKG="$OUT_DIR/${VERSION_BASENAME}-dev-adhoc.pkg"
  pkgbuild --root "$STAGE" --identifier "$PKG_IDENTIFIER" \
    --version "$VERSION" \
    --install-location "/usr/local/lykuro-native-engine" "$DEVPKG"
  cat <<EOF
OK: ad-hoc dev package -> $DEVPKG
    Internal install: sudo installer -pkg "$DEVPKG" -target /
    If copied between Macs and Gatekeeper-quarantined, clear it with:
      xattr -dr com.apple.quarantine /usr/local/lykuro-native-engine
    This package is NOT for downloads.lykuro.ai (Phase 2 requires
    Developer ID + notarization).
EOF
  shasum -a 256 "$DEVPKG"
  exit 0
fi

echo "==> 2/4 build + sign component .pkg"
PKG_UNSIGNED="$OUT_DIR/${VERSION_BASENAME}-unsigned.pkg"
PKG_SIGNED="$OUT_DIR/${VERSION_BASENAME}.pkg"
# Install under /usr/local/lykuro-native-engine (relocatable root).
pkgbuild --root "$STAGE" \
  --identifier "$PKG_IDENTIFIER" \
  --version "$VERSION" \
  --install-location "/usr/local/lykuro-native-engine" \
  "$PKG_UNSIGNED"

if [[ -n "${DEVELOPER_ID_INSTALLER:-}" ]]; then
  productsign --sign "$DEVELOPER_ID_INSTALLER" "$PKG_UNSIGNED" "$PKG_SIGNED"
  rm -f "$PKG_UNSIGNED"
else
  echo "    WARNING: DEVELOPER_ID_INSTALLER unset; .pkg not installer-signed" >&2
  mv "$PKG_UNSIGNED" "$PKG_SIGNED"
fi

echo "==> 3/4 notarize (xcrun notarytool submit --wait)"
if [[ -n "${NOTARY_PROFILE:-}" ]]; then
  xcrun notarytool submit "$PKG_SIGNED" --keychain-profile "$NOTARY_PROFILE" --wait
elif [[ -n "${NOTARY_APPLE_ID:-}" && -n "${NOTARY_TEAM_ID:-}" && -n "${NOTARY_PASSWORD:-}" ]]; then
  xcrun notarytool submit "$PKG_SIGNED" \
    --apple-id "$NOTARY_APPLE_ID" --team-id "$NOTARY_TEAM_ID" \
    --password "$NOTARY_PASSWORD" --wait
else
  echo "ERROR: no notary credentials (set NOTARY_PROFILE or NOTARY_APPLE_ID/TEAM_ID/PASSWORD)" >&2
  exit 2
fi

echo "==> 4/4 staple + Gatekeeper verify"
xcrun stapler staple "$PKG_SIGNED"
xcrun stapler validate "$PKG_SIGNED"
# Gatekeeper assessment for an installer package.
spctl --assess --type install --verbose=4 "$PKG_SIGNED" || {
  echo "ERROR: spctl rejected the package" >&2; exit 1; }

echo "OK: signed + notarized + stapled -> $PKG_SIGNED"
shasum -a 256 "$PKG_SIGNED"
