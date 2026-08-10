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
if [[ "${1:-}" == "--codesign-only" ]]; then CODESIGN_ONLY=1; shift; fi

STAGE="${1:?staged package dir (from make_package.sh)}"
OUT_DIR="${2:-$(dirname "$STAGE")}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENTITLEMENTS="$ROOT/deploy/macos/entitlements.plist"
PKG_IDENTIFIER="${PKG_IDENTIFIER:-ai.lykuro.native-engine}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: Developer ID signing must run on macOS" >&2; exit 2
fi
[[ -d "$STAGE/bin" ]] || { echo "ERROR: $STAGE has no bin/ (not a staged package?)" >&2; exit 2; }

if [[ -z "${DEVELOPER_ID_APP:-}" ]]; then
  cat >&2 <<'EOF'
NOTICE: DEVELOPER_ID_APP not set -> signing/notarization SKIPPED.
        The staged package remains unsigned (dev posture). To produce a
        signed+notarized package, export the Developer ID identities and
        notary credentials (see the header of this script) and re-run.
EOF
  exit 0
fi
command -v codesign >/dev/null || { echo "ERROR: codesign not found" >&2; exit 2; }
command -v pkgbuild >/dev/null || { echo "ERROR: pkgbuild not found" >&2; exit 2; }
command -v xcrun >/dev/null || { echo "ERROR: xcrun not found" >&2; exit 2; }
[[ -f "$ENTITLEMENTS" ]] || { echo "ERROR: entitlements not found: $ENTITLEMENTS" >&2; exit 2; }

VERSION_BASENAME="$(basename "$STAGE")"          # lykuro-native-engine-macos-metal-<ver>
VERSION="${VERSION_BASENAME##*-}"

echo "==> 1/4 codesign Mach-O binaries (Hardened Runtime)"
while IFS= read -r bin; do
  # Only sign actual Mach-O executables.
  if file "$bin" | grep -q 'Mach-O'; then
    echo "    signing $bin"
    codesign --force --timestamp --options runtime \
      --entitlements "$ENTITLEMENTS" \
      --sign "$DEVELOPER_ID_APP" "$bin"
    codesign --verify --strict --verbose=2 "$bin"
  fi
done < <(find "$STAGE/bin" -type f)

if [[ "$CODESIGN_ONLY" -eq 1 ]]; then
  echo "OK: binaries codesigned (--codesign-only); skipping pkg/notarize"
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
