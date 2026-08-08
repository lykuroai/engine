# Third-Party Notices

The Lykuro Native Inference Engine links the low-level dependencies below.
It contains **no third-party inference runtime** (Ollama, llama.cpp, vLLM,
TGI, mlx-lm) in source, binary, or transitive form (spec §0.2, AT-01 /
AT-M03). The JSON parser, SHA-256, safetensors reader, byte-level BPE
tokenizer, and all inference kernels are first-party (`core/`, `model/`,
`security/`, `backends/`).

Machine-readable inventory: `sbom/lykuro-native-engine.spdx.json`.

---

## Production dependencies

### Protocol Buffers — BSD-3-Clause
Copyright Google LLC.
<https://github.com/protocolbuffers/protobuf/blob/main/LICENSE>

### gRPC — Apache-2.0
Copyright The gRPC Authors.
<https://github.com/grpc/grpc/blob/master/LICENSE>

### OpenSSL (libcrypto) — Apache-2.0
Copyright The OpenSSL Project Authors.
<https://www.openssl.org/source/license.html>

### NVIDIA CUDA Toolkit (cudart) — NVIDIA CUDA EULA *(Linux profile only)*
Copyright NVIDIA Corporation. Approved low-level GPU SDK, redistributed
per the CUDA EULA's runtime-redistributable component list.
<https://docs.nvidia.com/cuda/eula/>

### Apple Metal / MetalPerformanceShaders / MPSGraph — Apple SDK *(macOS profile only)*
Copyright Apple Inc. System frameworks; not redistributed (present on the
host OS).
<https://developer.apple.com/metal/>

---

## Development-only dependencies (never in a production artifact)

### GoogleTest — BSD-3-Clause
Copyright Google LLC. Test binaries only; excluded from all release
packages.
<https://github.com/google/googletest/blob/main/LICENSE>

---

## Full Apache-2.0 / BSD-3-Clause texts

The complete license texts are bundled in the release package under
`licenses/` (`Apache-2.0.txt`, `BSD-3-Clause.txt`) alongside this file.
