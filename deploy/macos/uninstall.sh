#!/bin/bash
# Uninstall the Lykuro Native Inference Engine (addendum §24). Run as root.
#
#   sudo /Library/Application\ Support/Lykuro/NativeEngine/uninstall.sh [--purge]
#
# Default: stop + remove the daemon, binaries, bundled libs, and logs, but
# PRESERVE operator data (secrets/ and the model artifact) so an accidental
# uninstall is recoverable. --purge additionally removes secrets, the model
# store, and the dedicated service user.
set -u

INSTALL_ROOT="/Library/Application Support/Lykuro/NativeEngine"
MODEL_ROOT="/Library/Application Support/Lykuro/Models"
LOG_DIR="/Library/Logs/Lykuro/NativeEngine"
LABEL="ai.lykuro.native-engine"
PLIST_DST="/Library/LaunchDaemons/$LABEL.plist"
SVC_USER="_lykuro"
SVC_GROUP="_lykuro"
PKG_ID="ai.lykuro.native-engine"

PURGE=0
[ "${1:-}" = "--purge" ] && PURGE=1
[ "$(id -u)" -eq 0 ] || { echo "must run as root (sudo)" >&2; exit 1; }

# 1. Stop + unload the daemon (ignore if not loaded).
/bin/launchctl bootout "system/$LABEL" 2>/dev/null || true
/bin/rm -f "$PLIST_DST"

# 2. Remove program files (keep secrets + models unless --purge).
/bin/rm -rf "$INSTALL_ROOT/bin" "$INSTALL_ROOT/lib" "$INSTALL_ROOT/launchd" \
           "$INSTALL_ROOT/contracts" "$INSTALL_ROOT/compatibility" \
           "$INSTALL_ROOT/sbom" "$INSTALL_ROOT/licenses" "$INSTALL_ROOT/docs" \
           "$INSTALL_ROOT/checksums.sha256" "$INSTALL_ROOT/manifest.json" \
           "$INSTALL_ROOT/manifest.sig" "$INSTALL_ROOT/enable_service.sh" \
           "$INSTALL_ROOT/uninstall.sh"
/bin/rm -rf "$LOG_DIR"

# 3. Forget the pkg receipt so a reinstall is clean.
/usr/sbin/pkgutil --forget "$PKG_ID" >/dev/null 2>&1 || true

if [ "$PURGE" -eq 1 ]; then
  echo "uninstall: --purge removing secrets, models, and service user"
  /bin/rm -rf "$INSTALL_ROOT/secrets" "$INSTALL_ROOT/config" "$INSTALL_ROOT" \
             "$MODEL_ROOT"
  /usr/bin/dscl . -delete "/Users/$SVC_USER" 2>/dev/null || true
  /usr/bin/dscl . -delete "/Groups/$SVC_GROUP" 2>/dev/null || true
else
  echo "uninstall: preserved $INSTALL_ROOT/secrets and $MODEL_ROOT"
  echo "           (re-run with --purge to remove them and the $SVC_USER user)"
fi

echo "OK: $LABEL uninstalled."
