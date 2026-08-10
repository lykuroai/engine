#!/bin/bash
# Activate the Lykuro Native Inference Engine as a LaunchDaemon (addendum
# §23.3/§24). Run as root AFTER placing secrets + config. Idempotent:
# creates the dedicated non-root service user if missing, fixes ownership,
# installs the plist, and bootstraps the daemon.
#
#   sudo /Library/Application\ Support/Lykuro/NativeEngine/enable_service.sh
set -eu

INSTALL_ROOT="/Library/Application Support/Lykuro/NativeEngine"
LOG_DIR="/Library/Logs/Lykuro/NativeEngine"
LABEL="ai.lykuro.native-engine"
PLIST_DST="/Library/LaunchDaemons/$LABEL.plist"
SVC_USER="_lykuro"
SVC_GROUP="_lykuro"

[ "$(id -u)" -eq 0 ] || { echo "must run as root (sudo)" >&2; exit 1; }
[ -x "$INSTALL_ROOT/bin/lykuro-native-engine" ] || {
  echo "engine not installed at $INSTALL_ROOT" >&2; exit 1; }
[ -f "$INSTALL_ROOT/config/engine.json" ] || {
  echo "missing $INSTALL_ROOT/config/engine.json (configure before enabling)" >&2
  exit 1; }

# Create the hidden service group + user in the system range if absent.
if ! /usr/bin/dscl . -read "/Groups/$SVC_GROUP" >/dev/null 2>&1; then
  gid=$(/usr/bin/dscl . -list /Groups PrimaryGroupID | /usr/bin/awk '$2>200&&$2<400{print $2}' | /usr/bin/sort -n | /usr/bin/tail -1)
  gid=$(( ${gid:-300} + 1 ))
  /usr/bin/dscl . -create "/Groups/$SVC_GROUP"
  /usr/bin/dscl . -create "/Groups/$SVC_GROUP" PrimaryGroupID "$gid"
  echo "created group $SVC_GROUP ($gid)"
fi
if ! /usr/bin/dscl . -read "/Users/$SVC_USER" >/dev/null 2>&1; then
  gid=$(/usr/bin/dscl . -read "/Groups/$SVC_GROUP" PrimaryGroupID | /usr/bin/awk '{print $2}')
  uid=$(/usr/bin/dscl . -list /Users UniqueID | /usr/bin/awk '$2>200&&$2<400{print $2}' | /usr/bin/sort -n | /usr/bin/tail -1)
  uid=$(( ${uid:-300} + 1 ))
  /usr/bin/dscl . -create "/Users/$SVC_USER"
  /usr/bin/dscl . -create "/Users/$SVC_USER" UniqueID "$uid"
  /usr/bin/dscl . -create "/Users/$SVC_USER" PrimaryGroupID "$gid"
  /usr/bin/dscl . -create "/Users/$SVC_USER" UserShell /usr/bin/false
  /usr/bin/dscl . -create "/Users/$SVC_USER" NFSHomeDirectory /var/empty
  /usr/bin/dscl . -create "/Users/$SVC_USER" RealName "Lykuro Native Engine Service"
  /usr/bin/dscl . -create "/Users/$SVC_USER" IsHidden 1
  echo "created user $SVC_USER ($uid)"
fi

# Ownership: service user owns its tree; secrets stay 0700.
/usr/sbin/chown -R "$SVC_USER:$SVC_GROUP" "$INSTALL_ROOT" "$LOG_DIR"
/bin/chmod 700 "$INSTALL_ROOT/secrets"

# Install the LaunchDaemon plist (root-owned, per launchd requirements).
/bin/cp "$INSTALL_ROOT/launchd/$LABEL.plist" "$PLIST_DST"
/usr/sbin/chown root:wheel "$PLIST_DST"
/bin/chmod 644 "$PLIST_DST"

# (Re)bootstrap the daemon.
/bin/launchctl bootout "system/$LABEL" 2>/dev/null || true
/bin/launchctl bootstrap system "$PLIST_DST"
/bin/launchctl enable "system/$LABEL"
/bin/launchctl kickstart -k "system/$LABEL"

echo "OK: $LABEL bootstrapped. Verify readiness per docs/operations/runbook.md (§3)."
