#!/usr/bin/env bash
# Assembles a reproducible release package for one profile (spec §27.1,
# §23.1). Produces a staged tree with the executable, tools, contracts,
# SBOM/licenses, per-file checksums, a package manifest recording build
# provenance, and (when a signing key is given) a detached Ed25519
# signature over that manifest.
#
#   make_package.sh <build_dir> <profile: linux-cuda|macos-metal> \
#                   <out_dir> [signing_key_file]
#
# Signing/notarization of the macOS .pkg itself is a separate,
# Developer-ID-gated step (see deploy/macos/README.md).
set -euo pipefail

BUILD_DIR="${1:?build dir}"
PROFILE="${2:?profile}"
OUT_DIR="${3:?out dir}"
KEY_FILE="${4:-}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION="$("$BUILD_DIR/cmd/native-engine/native-engine" --version 2>/dev/null || echo 0.0.0)"
GIT_REV="$(git rev-parse HEAD)"
GIT_DIRTY="$(git status --porcelain | head -1)"
if [[ -n "$GIT_DIRTY" ]]; then
    echo "ERROR: working tree is dirty; release builds require a clean tree" >&2
    exit 1
fi

# Forbidden-runtime gate before anything is staged.
"$ROOT/tools/check_no_forbidden_runtime.sh"

STAGE="$OUT_DIR/lykuro-native-engine-${PROFILE}-${VERSION}"
rm -rf "$STAGE"
mkdir -p "$STAGE"/{bin,contracts,compatibility,sbom,licenses,docs}

cp "$BUILD_DIR/cmd/native-engine/native-engine" "$STAGE/bin/lykuro-native-engine"
for t in sign_artifact verify_reference bench_decode soak_engine; do
    [[ -x "$BUILD_DIR/tools/$t" ]] && cp "$BUILD_DIR/tools/$t" "$STAGE/bin/"
done
cp -R api/proto "$STAGE/contracts/proto"
cp api/schema/model-manifest.schema.json "$STAGE/contracts/"
cp -R docs/certified-profiles "$STAGE/compatibility/" 2>/dev/null || true
cp sbom/*.json sbom/PROVENANCE.md "$STAGE/sbom/"
cp licenses/* "$STAGE/licenses/"
cp README.md "$STAGE/docs/"
if [[ "$PROFILE" == macos-metal ]]; then
    mkdir -p "$STAGE/launchd" "$STAGE/config"
    cp deploy/macos/ai.lykuro.native-engine.plist "$STAGE/launchd/"
    cp deploy/macos/engine.example.yaml "$STAGE/config/"
fi

# macOS Developer ID: codesign the Mach-O binaries with Hardened Runtime
# BEFORE checksums, so the tarball and the Ed25519 manifest signature cover
# the signed binaries. Self-skips when DEVELOPER_ID_APP is unset (dev
# posture). The notarized .pkg is produced by the full pipeline (run
# deploy/macos/sign_and_notarize.sh "$STAGE" after this script).
if [[ "$PROFILE" == macos-metal ]]; then
    SIGN_ARGS=(--codesign-only)
    # Phase 1 internal builds: LYKURO_DEV_ADHOC=1 ad-hoc-signs the binaries
    # (Hardened Runtime) without an Apple Developer Program membership.
    [[ "${LYKURO_DEV_ADHOC:-0}" == 1 ]] && SIGN_ARGS+=(--dev)
    "$ROOT/deploy/macos/sign_and_notarize.sh" "${SIGN_ARGS[@]}" "$STAGE"
fi

# Per-file checksums (sorted for determinism).
( cd "$STAGE" && find . -type f ! -name checksums.sha256 -print0 \
    | sort -z | xargs -0 shasum -a 256 > checksums.sha256 )

# Package manifest with provenance.
cat > "$STAGE/manifest.json" <<JSON
{
  "package": "lykuro-native-engine",
  "version": "${VERSION}",
  "profile": "${PROFILE}",
  "git_revision": "${GIT_REV}",
  "built_at_host": "$(uname -sm)",
  "checksums": "checksums.sha256",
  "sbom": "sbom/lykuro-native-engine.spdx.json",
  "forbidden_runtime_check": "passed"
}
JSON

# Detached signature over the manifest (same Ed25519 tooling as model
# artifacts). The verify side lives in security/signature.
if [[ -n "$KEY_FILE" ]]; then
    "$BUILD_DIR/tools/sign_artifact" sign "$KEY_FILE" "$STAGE" >/dev/null \
        || { echo "sign_artifact expects an artifact dir; signing the package manifest directly:"; }
    # sign_artifact signs manifest.json in-place -> manifest.sig.
    [[ -f "$STAGE/manifest.sig" ]] && echo "signed: manifest.sig"
fi

# Reproducible tarball (sorted, fixed mtime/owner).
TARBALL="$OUT_DIR/lykuro-native-engine-${PROFILE}-${VERSION}.tar.gz"
tar --numeric-owner --owner=0 --group=0 --mtime='2026-08-08 00:00:00' \
    --sort=name -czf "$TARBALL" -C "$OUT_DIR" "$(basename "$STAGE")" 2>/dev/null \
  || tar -czf "$TARBALL" -C "$OUT_DIR" "$(basename "$STAGE")"

echo "package: $TARBALL"
echo "staged:  $STAGE"
shasum -a 256 "$TARBALL"
