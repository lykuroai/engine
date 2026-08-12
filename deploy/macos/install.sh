#!/usr/bin/env bash
# Ollama-style installer for the Lykuro Native Inference Engine on macOS
# (Apple Silicon). Fetches the release binary with curl — which, unlike a
# browser download, sets no Gatekeeper quarantine — so the ad-hoc-signed
# binary runs immediately WITHOUT a Developer ID certificate or manual
# xattr step. (A downloads.lykuro.ai production build will additionally be
# Developer ID signed + notarized; that is Phase 2.)
#
#   curl -fsSL https://raw.githubusercontent.com/lykuroai/engine/main/deploy/macos/install.sh | bash
#
# Env overrides: LYKURO_VERSION (default v1.0.0), LYKURO_PREFIX
# (default /usr/local/bin), LYKURO_SHA256 (verify the download).
set -euo pipefail

VERSION="${LYKURO_VERSION:-v1.0.1}"
PREFIX="${LYKURO_PREFIX:-/usr/local/bin}"
ASSET="lykuro-native-engine-macos-arm64"
DEST="$PREFIX/native-engine"
URL="https://github.com/lykuroai/engine/releases/download/${VERSION}/${ASSET}"
# Integrity: the download is over GitHub HTTPS. For an extra check, set
# LYKURO_SHA256 to the value shown on the release page. (Not pinned here so
# the installer never goes stale when a release asset is refreshed.)
SHA256="${LYKURO_SHA256:-}"

[ "$(uname -s)" = "Darwin" ] || { echo "macOS only" >&2; exit 1; }
[ "$(uname -m)" = "arm64" ] || { echo "Apple Silicon (arm64) only" >&2; exit 1; }

echo "==> downloading $ASSET ($VERSION)"
tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
curl -fsSL -o "$tmp" "$URL"

if [ -n "$SHA256" ]; then
  got="$(shasum -a 256 "$tmp" | awk '{print $1}')"
  [ "$got" = "$SHA256" ] || { echo "checksum mismatch: $got != $SHA256" >&2; exit 1; }
  echo "==> checksum OK"
fi

chmod +x "$tmp"
# curl leaves no com.apple.quarantine; strip it anyway for belt-and-braces.
xattr -d com.apple.quarantine "$tmp" 2>/dev/null || true

echo "==> installing to $DEST"
if mkdir -p "$PREFIX" 2>/dev/null && [ -w "$PREFIX" ]; then
  mv "$tmp" "$DEST"
else
  sudo mkdir -p "$PREFIX"
  sudo mv "$tmp" "$DEST"
fi
trap - EXIT
chmod +x "$DEST" 2>/dev/null || sudo chmod +x "$DEST"

echo "==> installed: native-engine"
"$DEST" --version
cat <<EOF
Get started (Ollama-style):
  native-engine pull Qwen/Qwen2.5-0.5B-Instruct
  native-engine run ~/.lykuro/models/Qwen_Qwen2.5-0.5B-Instruct "Hello"
  native-engine serve --config engine.json      # gRPC server
EOF
if ! command -v native-engine >/dev/null 2>&1; then
  echo "NOTE: $PREFIX is not on your PATH — add it or run $DEST directly." >&2
fi
