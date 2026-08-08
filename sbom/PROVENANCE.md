# Build Provenance

Provenance record for Lykuro Native Inference Engine release artifacts
(spec §22.3, §27.1). One record is emitted per built package by
`tools/make_package.sh` and embedded in the package `manifest.json`.

## Source

- Repository: `https://github.com/lykuroai/engine`
- Revision: recorded as the exact `git rev-parse HEAD` at build time
- Working tree: release builds require a clean tree (no uncommitted
  changes); the packaging script aborts otherwise

## Build environments (as-built, 2026-08-08)

### Linux + CUDA profile
- OS: Ubuntu 22.04.2 LTS (x86_64)
- Compiler: GCC 11.4.0
- CUDA: 12.8 (nvcc cuda_12.8.r12.8)
- gRPC: 1.66.2 (built from source)
- Protobuf: 27.2
- OpenSSL: 3.0.2 (system)
- CMake: 4.4.2, Ninja 1.13.2
- Target GPU arch: sm_75 (Turing), sm_86 (Ampere)

### macOS + Metal profile
- OS: macOS 26.5.2 (25F84), arm64
- Compiler: Apple clang 21.0.0
- Frameworks: Metal / MetalPerformanceShaders / MPSGraph (macOS 26 SDK)
- Protobuf: 35.1, OpenSSL: 3.6.3
- CMake: 4.4.2, Ninja 1.13.2

## Reproducibility

- Release preset (`CMakePresets.json: release`) fixes `CMAKE_BUILD_TYPE`,
  disables sanitizers, and pins the generator (Ninja).
- Third-party versions are pinned per build host and listed in the SBOM.
- Non-reproducible inputs to avoid: no `Date.now()`-style timestamps are
  compiled in; the engine version string is the only build-stamped value.

## Signing

- Linux tarball: SHA-256 checksum + detached Ed25519 signature over the
  package `manifest.json` (same key infrastructure as model artifacts).
- macOS pkg: Developer ID codesign + Apple notarization — **deferred**
  (no Developer ID / notarization account in the current environment;
  see `deploy/macos/README.md`).

## Vulnerability posture

- Dependencies are limited to widely-audited low-level SDKs (see SBOM).
- No runtime network egress and no dynamic dependency download (spec
  §0.2): the engine holds no fetch/registry client code.
- CVE scanning of the pinned dependency set is a CI gate (§30.1); results
  are attached to the release as `vulnerability-report.json` when the
  scanner runs.
