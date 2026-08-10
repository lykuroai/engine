#!/usr/bin/env bash
# Verify a built engine binary is self-contained: no third-party runtime
# dependency leaks in via dynamic linking (spec §0.2 / §23.2). This guards
# the single-binary guarantee against regressions where a build picks up a
# Homebrew / non-static gRPC/protobuf/abseil/OpenSSL again.
#
#   tools/check_selfcontained.sh <binary> [<binary> ...]
#
# Policy (per platform):
#   macOS  — allow only /usr/lib/* and /System/Library/Frameworks/*.
#            Forbid /opt/homebrew, /usr/local, and any @rpath/@loader_path
#            dependency (a single static binary has none).
#   Linux  — allow the C/C++ runtime (libc/libm/libstdc++/libgcc_s/ld-linux/
#            libpthread/libdl/librt/vdso), the OS-managed OpenSSL
#            (libssl/libcrypto — apt-patched, dynamic by design), and the
#            CUDA/NVIDIA runtime. Forbid grpc/protobuf/absl/re2/cares and
#            any dependency resolved from a home/opt/local path.
#
# Exit 0 = clean, 1 = a forbidden dependency was found, 2 = usage error.
set -euo pipefail

[ $# -ge 1 ] || { echo "usage: $0 <binary> [<binary> ...]" >&2; exit 2; }

os="$(uname -s)"
rc=0

check_macos() {
  local bin="$1" bad
  # Dependency install-names, minus the binary's own id line.
  bad="$(otool -L "$bin" | tail -n +2 | awk '{print $1}' \
    | grep -vE '^/usr/lib/|^/System/Library/Frameworks/' || true)"
  if [ -n "$bad" ]; then
    echo "FAIL $bin — non-system dynamic dependencies:" >&2
    echo "$bad" | sed 's/^/    /' >&2
    return 1
  fi
  echo "ok   $bin (system libs + Apple frameworks only)"
}

check_linux() {
  local bin="$1" deps bad
  deps="$(ldd "$bin" 2>/dev/null || true)"
  # Forbidden: the bundled-stack libraries, or any dep resolved from a
  # non-system prefix (home/opt/local) — EXCEPT the CUDA/NVIDIA runtime,
  # which legitimately lives under /usr/local/cuda and must stay dynamic.
  bad="$(printf '%s\n' "$deps" \
    | grep -iE 'libgrpc|libprotobuf|libabsl|libre2|libcares|libupb|=> (/home|/opt|/usr/local)/' \
    | grep -viE 'cuda|nvidia' \
    || true)"
  if [ -n "$bad" ]; then
    echo "FAIL $bin — forbidden dynamic dependencies:" >&2
    echo "$bad" | sed 's/^/    /' >&2
    return 1
  fi
  echo "ok   $bin (runtime + system OpenSSL + CUDA only)"
}

for bin in "$@"; do
  [ -f "$bin" ] || { echo "FAIL $bin — not found" >&2; rc=1; continue; }
  case "$os" in
    Darwin) check_macos "$bin" || rc=1;;
    Linux)  check_linux "$bin" || rc=1;;
    *) echo "unsupported OS: $os" >&2; exit 2;;
  esac
done

[ "$rc" -eq 0 ] && echo "self-contained: OK" || echo "self-contained: FAIL" >&2
exit "$rc"
