# Compatibility Matrix

As-built compatibility for Lykuro Native Inference Engine (spec §28.2,
Metal addendum §26.1). Rows marked "verified" were exercised on real
hardware; anything else is declared as untested and must not be presented
as certified.

## Versioning

| Component | Value |
|---|---|
| Engine version | 1.0.0 |
| Data / Control API | `lykuro.nie.v1` (proto package) |
| Backend ABI | `nie_abi_1` |
| Manifest schema | `1` (`api/schema/model-manifest.schema.json`) |
| Certified Profile schema | dev-measured (see `docs/certified-profiles/`) |

Platform ↔ Engine API compatibility policy: **N / N-1** (§28.2).

## Backend × hardware (verified)

| Backend | Hardware | Precision | Status |
|---|---|---|---|
| CPU reference | any x86_64 / arm64 | FP32 | verified (correctness oracle) |
| CUDA | RTX 3060 (sm_86) | BF16 weights / INT8 / INT4 | verified: oracle 3/3, 24h soak running |
| CUDA | GTX 1650 (sm_75) | BF16 weights | verified: parity 6/6 |
| CUDA tensor-parallel (2-way) | RTX 3060 + GTX 1650 | FP32 | verified: oracle 3/3 |
| Metal (MPSGraph) | Apple M4 Pro | FP32 | verified: oracle 3/3 |
| Metal (MPSGraph) | Apple M4 Pro | FP16 | verified: argmax/greedy 3/3 |

## Toolchain (as-built)

| Profile | OS | Compiler | GPU stack |
|---|---|---|---|
| linux-cuda | Ubuntu 22.04.2 | GCC 11.4 | CUDA 12.8, driver 570.211 |
| macos-metal | macOS 26.5.2 | Apple clang 21.0 | Metal / MPSGraph (macOS 26 SDK) |

## Model families

| Architecture | Weight format | Status |
|---|---|---|
| `approved_qwen_decoder_v1` (Qwen2 dense) | safetensors (BF16/FP16/FP32) | verified with Qwen2.5-0.5B-Instruct |

## Not certified / untested (explicit)

- CUDA compute capabilities other than sm_75 / sm_86.
- Apple Silicon other than M4 Pro; macOS versions other than 26.5.2.
- Sharded safetensors, MoE, embedding, and vision models.
- Multi-GPU beyond the 2-way tensor-parallel PoC; NCCL transport.
- 24h soak on Metal (deferred); GPU OOM / thermal-throttle recovery under
  sustained production load.
- Any driver / CUDA / macOS version not listed above — re-verify per
  release against this matrix.
