# Release Notes

## v1.0.0 (2026-08-08)

First release of the Lykuro Native Inference Engine — a from-scratch
local LLM inference engine with no third-party inference runtime, per
LYK-NIE-SD-001 and the Metal addendum LYK-NIE-ADD-METAL-001.

### Highlights

- **Own engine, end to end.** In-tree JSON parser, SHA-256, safetensors
  reader, byte-level BPE tokenizer, chat template, sampler, scheduler,
  KV cache, and all inference kernels. No Ollama / llama.cpp / vLLM /
  TGI / mlx-lm in source, binary, or transitive form (AT-01 / AT-M03,
  enforced by `tools/check_no_forbidden_runtime.sh`).
- **Correctness first.** CPU FP32 reference as the oracle; every backend
  is verified against the real Qwen2.5-0.5B-Instruct checkpoint using an
  HF-transformers reference (logits + greedy agreement), and is
  bit-exact run-to-run.
- **Two GPU backends behind one interface.**
  - **CUDA** (Linux): BF16-resident weights, fused decode kernels, CUDA
    Graphs, paged KV + scoped prefix cache, INT8/INT4 weight-only
    quantization, and a 2-way tensor-parallel PoC. ~150 tok/s single /
    ~450 tok/s at batch 16 on an RTX 3060 (dev figures).
  - **Metal** (macOS, Apple Silicon): MPSGraph forward with resident
    unified-memory buffers, chunked prefill, graph pre-warm, and FP16
    (weights/activations/KV, FP32-safe reductions) halving resident VRAM.
- **Security & serving.** mTLS gRPC with Control/Data identity
  separation, Ed25519 manifest signature verification (fail-closed),
  content-free logs/metrics, loopback metrics endpoint, and a strict
  fail-closed JSON config.
- **Stability.** Per-request bookkeeping is bounded by in-flight work; a
  600s mixed-load soak and 100 unload/reload cycles ran leak-free (24h
  CUDA soak in progress).

### Packaging

- `linux-cuda` and `macos-metal` profiles via `tools/make_package.sh`:
  staged tree, sorted checksums, provenance manifest, Ed25519-signed
  manifest, deterministic tarball. SBOM (SPDX 2.3) and full license
  texts included.

### Known limitations / not certified

- Certified Profiles are **dev-measured**, not production-certified: 24h
  Metal soak, formal security review, and signed-artifact-only
  measurement are pending.
- macOS package signing (Developer ID) and notarization are deferred
  (no credentials in the build environment); custom Metal kernels
  (precompiled `metallib`) require an Xcode CI runner — the MVP is
  MPSGraph-only.
- Hardware/OS coverage is limited to the entries in
  `docs/compatibility-matrix.md`. Sharded weights, MoE, embeddings,
  vision, and NCCL-based multi-GPU are out of scope for this release.
- Quantized (INT8/INT4/FP16) models change greedy output relative to the
  FP32 oracle by design; per-model quality gating belongs to the offline
  evaluation pipeline.

See `docs/DEFINITION_OF_DONE.md` for the full per-item status against
both specifications.
