#!/usr/bin/env bash
# Build gRPC (and its abseil/protobuf/re2/c-ares/zlib deps) from source as
# STATIC libraries, so the engine links one self-contained binary with no
# Homebrew/dylib runtime dependency (spec §0.2 / §23.2). This mirrors the
# Linux host, which builds gRPC from source.
#
#   third_party/build_grpc_static.sh [install_prefix]
#
# Pinned to the SBOM version (gRPC 1.66.2 -> vendors protobuf 27.x). SSL is
# the system OpenSSL, linked statically, so gRPC and the engine's own
# signature code share one crypto implementation (no boringssl symbol
# clash).
set -euo pipefail

# Per-platform gRPC pin (both build a single self-contained binary):
#   macOS  -> 1.83.0 (matches Homebrew; 1.66.2's vendored abseil mis-emits
#             -msse4.1 on arm64 under Apple clang 21)
#   Linux  -> 1.66.2 (the CUDA-host pin; builds fine with gcc)
UNAME="$(uname -s)"
if [ "$UNAME" = "Darwin" ]; then
  DEFAULT_TAG="v1.83.0"
  DEFAULT_OPENSSL="/opt/homebrew/opt/openssl"
  JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
else
  DEFAULT_TAG="v1.66.2"
  DEFAULT_OPENSSL=""   # let CMake find the system OpenSSL
  JOBS="$(nproc 2>/dev/null || echo 4)"
fi
TAG="${GRPC_TAG:-$DEFAULT_TAG}"
PREFIX="${1:-$HOME/.local/grpc-static}"
SRC="${GRPC_SRC:-$HOME/.cache/grpc-src-$TAG}"
OPENSSL_ROOT="${OPENSSL_ROOT_DIR:-$DEFAULT_OPENSSL}"

echo "==> gRPC $TAG -> static install at $PREFIX (src $SRC)"
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 -b "$TAG" https://github.com/grpc/grpc "$SRC"
  git -C "$SRC" submodule update --init --recursive --depth 1
fi

SSL_ARGS=(-DgRPC_SSL_PROVIDER=package -DOPENSSL_USE_STATIC_LIBS=ON)
[ -n "$OPENSSL_ROOT" ] && SSL_ARGS+=(-DOPENSSL_ROOT_DIR="$OPENSSL_ROOT")

cmake -S "$SRC" -B "$SRC/build-static" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DgRPC_INSTALL=ON \
  -DgRPC_BUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DgRPC_ABSL_PROVIDER=module \
  -DgRPC_PROTOBUF_PROVIDER=module \
  -DgRPC_RE2_PROVIDER=module \
  -DgRPC_CARES_PROVIDER=module \
  -DgRPC_ZLIB_PROVIDER=module \
  "${SSL_ARGS[@]}" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

cmake --build "$SRC/build-static" -j"$JOBS"
cmake --install "$SRC/build-static"

echo "==> done. Static archives:"
ls "$PREFIX"/lib/libgrpc++.a "$PREFIX"/lib/libprotobuf.a 2>/dev/null || true
echo "Configure the engine with: -DCMAKE_PREFIX_PATH=$PREFIX"
