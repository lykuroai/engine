#!/usr/bin/env bash
# Build a self-contained Linux/arm64 (aarch64) engine binary in a native
# arm64 container (fast on Apple Silicon — no emulation). CPU backend +
# gRPC; gRPC/protobuf/abseil are static, leaving only the C/C++ runtime and
# system OpenSSL dynamic (single-binary policy, spec §23.2).
#
#   deploy/linux/build_arm64_docker.sh
#
# Output: dist/lykuro-native-engine-linux-arm64
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/dist" "$ROOT/.cache/arm64/grpc-static" "$ROOT/.cache/arm64/grpc-src"

# Cache the static gRPC install + source across runs (the engine build may
# iterate; gRPC does not need rebuilding each time).
docker run --rm --platform linux/arm64 \
  -v "$ROOT:/src" -w /src \
  -v "$ROOT/.cache/arm64/grpc-static:/opt/grpc-static" \
  -v "$ROOT/.cache/arm64/grpc-src:/tmp/grpc-src" \
  ubuntu:22.04 bash -eu -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq build-essential ninja-build git \
      pkg-config libssl-dev ca-certificates file python3-pip >/dev/null
    # Ubuntu 22.04 ships CMake 3.22; the engine needs >= 3.24.
    pip3 install -q "cmake>=3.24" >/dev/null
    hash -r
    echo "arch: $(uname -m)  cmake: $(cmake --version | head -1)"

    # 1. static gRPC (Linux pin 1.66.2, aarch64) — skip if already cached.
    if [ ! -f /opt/grpc-static/lib/libgrpc++.a ]; then
      GRPC_SRC=/tmp/grpc-src bash third_party/build_grpc_static.sh /opt/grpc-static
    else
      echo "gRPC static: cached, skipping rebuild"
    fi

    # 2. engine: CPU backend + gRPC, static third-party stack
    cmake -S . -B build/arm64 -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DLYKURO_ENABLE_GRPC=ON -DLYKURO_BUILD_TESTS=OFF \
      -DCMAKE_PREFIX_PATH=/opt/grpc-static
    cmake --build build/arm64 --target native-engine -j"$(nproc)"

    cp build/arm64/cmd/native-engine/native-engine \
       dist/lykuro-native-engine-linux-arm64
    echo "=== ldd (self-contained check) ==="
    ldd dist/lykuro-native-engine-linux-arm64 || true
    file dist/lykuro-native-engine-linux-arm64
  '

echo "=== host-side verification ==="
"$ROOT/tools/check_selfcontained.sh" "$ROOT/dist/lykuro-native-engine-linux-arm64" 2>/dev/null \
  || echo "(run check on a Linux host; macOS ldd differs)"
file "$ROOT/dist/lykuro-native-engine-linux-arm64"
shasum -a 256 "$ROOT/dist/lykuro-native-engine-linux-arm64"
