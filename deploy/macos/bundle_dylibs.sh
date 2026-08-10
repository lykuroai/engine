#!/usr/bin/env bash
# Make a staged macOS package self-contained (spec §23.2: no Homebrew in
# production). Copies every non-system dynamic library the staged binaries
# depend on into <stage>/lib, rewrites all install names to @rpath, and
# adds an @loader_path-relative rpath so the binaries resolve the bundled
# copies instead of /opt/homebrew or /usr/local.
#
#   deploy/macos/bundle_dylibs.sh <staged_dir>
#
# Path rewriting ONLY — it does not sign. Rewriting invalidates any
# existing signature, so run this BEFORE codesigning (sign_and_notarize.sh
# signs lib/ and bin/ afterward). System libraries under /usr/lib and
# /System are left as dynamic references (they are part of the OS ABI).
set -euo pipefail

STAGE="${1:?staged package dir}"
BIN="$STAGE/bin"
LIB="$STAGE/lib"
[[ -d "$BIN" ]] || { echo "ERROR: $STAGE has no bin/" >&2; exit 2; }
command -v otool >/dev/null || { echo "ERROR: otool not found" >&2; exit 2; }
command -v install_name_tool >/dev/null || {
  echo "ERROR: install_name_tool not found" >&2; exit 2; }
mkdir -p "$LIB"

is_system() {  # 0 (true) for OS-provided libraries we must not bundle
  case "$1" in
    /usr/lib/*|/System/*) return 0;;
    *) return 1;;
  esac
}

# Resolve a dependency string to an on-disk path (handles @rpath entries
# already pointing at our bundle).
resolve() {
  local dep="$1"
  case "$dep" in
    @rpath/*) echo "$LIB/${dep#@rpath/}";;
    @loader_path/*|@executable_path/*) echo "";;  # already relative
    *) echo "$dep";;
  esac
}

declare -a QUEUE=()
seen_lib() { [[ -f "$LIB/$1" ]]; }

# Rewrite one Mach-O file's dependency references; queue newly-copied libs.
rewrite() {
  local file="$1"
  local dep base src
  while IFS= read -r dep; do
    [[ -z "$dep" ]] && continue
    is_system "$dep" && continue
    case "$dep" in @loader_path/*|@executable_path/*) continue;; esac
    base="$(basename "$dep")"
    if ! seen_lib "$base"; then
      src="$(resolve "$dep")"
      if [[ -z "$src" || ! -f "$src" ]]; then
        echo "WARN: cannot resolve $dep (from $(basename "$file"))" >&2
        continue
      fi
      cp "$src" "$LIB/$base"; chmod u+w "$LIB/$base"
      install_name_tool -id "@rpath/$base" "$LIB/$base"
      QUEUE+=("$LIB/$base")
    fi
    install_name_tool -change "$dep" "@rpath/$base" "$file" 2>/dev/null || true
  done < <(otool -L "$file" | tail -n +2 | awk '{print $1}')
}

# Remove any build-time rpath that points outside the bundle (e.g. the
# linker bakes in /opt/homebrew/lib), so dyld only ever searches our lib/.
strip_bad_rpaths() {
  local file="$1" p
  while IFS= read -r p; do
    case "$p" in
      /opt/homebrew/*|/usr/local/*)
        install_name_tool -delete_rpath "$p" "$file" 2>/dev/null || true;;
    esac
  done < <(otool -l "$file" | awk '/LC_RPATH/{r=1} r&&/ path /{print $2; r=0}')
}

# Seed with the executables, then drain the queue (transitive closure).
for b in "$BIN"/*; do
  file "$b" | grep -q 'Mach-O' || continue
  strip_bad_rpaths "$b"
  install_name_tool -add_rpath "@loader_path/../lib" "$b" 2>/dev/null || true
  rewrite "$b"
done
while [[ ${#QUEUE[@]} -gt 0 ]]; do
  next="${QUEUE[0]}"; QUEUE=("${QUEUE[@]:1}")
  # Bundled libs sit beside each other; @rpath -> the lib dir itself.
  strip_bad_rpaths "$next"
  install_name_tool -add_rpath "@loader_path" "$next" 2>/dev/null || true
  rewrite "$next"
done

echo "bundled $(ls -1 "$LIB" | wc -l | tr -d ' ') dylib(s) into $LIB"
# Fail closed: no staged binary may still reference Homebrew / /usr/local.
if otool -L "$BIN"/* "$LIB"/* 2>/dev/null \
     | grep -E '/opt/homebrew|/usr/local' ; then
  echo "ERROR: residual non-bundled dependency remains" >&2
  exit 1
fi
echo "OK: no /opt/homebrew or /usr/local references remain"
