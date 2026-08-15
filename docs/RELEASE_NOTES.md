# Release Notes

## Unreleased

### CUDA decode overhaul — ahead of Ollama on RTX 3060

- Vectorized 8-wide weight loads (per-element nibble reads were an 8x
  DRAM amplification that left cuda-q4 no faster than BF16), warp-per-
  row B=1 GEMV kernels with shared-memory-staged activations, and a
  quantized lm head in the quantized modes (the BF16 head read was the
  single largest decode cost).
- GreedyRun ported to CUDA: up to 16 back-to-back decode-graph replays
  with a two-stage on-GPU argmax feeding the next replay's embedding
  gather — one host round trip per run instead of per token.
- The CLI/HTTP surface gains cuda-q8[:N] / cuda-q4[:N] (the backend's
  INT8/INT4 weight-only quantization was previously unreachable).
- RTX 3060, 256-token generations, host-local client measurement, vs
  Ollama 0.21.0 q4_K_M:
  Qwen2.5-0.5B: 434 tok/s, TTFT 45 ms  (Ollama 246 / 116 ms; was 140)
  Qwen2.5-1.5B: 183 tok/s, TTFT 140 ms (Ollama 152 / 119 ms; was 66)

### Greedy GPU pipeline + split-row attention — ahead of MLX on 0.5B

- Greedy decoding now runs up to 16 speculative steps per command
  buffer with on-GPU argmax (Sampler's exact rule: max, ties to the
  lowest index) and on-GPU embedding gather, and always keeps one
  speculative batch committed ahead of the one being consumed — the
  CPU/GPU round trip disappears from the decode loop. Temperature > 0
  keeps the per-token Decode + CPU sampler path. New GenerativeModel
  API: SupportsGreedyRun/GreedyRun (backend-optional).
- Attention switches to split-row flash decoding (heads x segments,
  fixed-order merge, deterministic); prompt prefill batches whole
  chunks into one command buffer.
- Qwen2.5-0.5B on an M4 Pro: 485 tok/s / TTFT 37 ms (was 278/42; MLX
  4bit self-reported: 441; Ollama q4_K_M: 246/103 ms). Qwen2.5-1.5B:
  210 tok/s (MLX: 214, Ollama: 154).
- Parity tests force GreedyRun to match sequential Decode + greedy
  argmax bit-exactly, cover the long-context attention kernels, and
  exercise speculative-trajectory divergence/flush.

## v1.0.3 (2026-08-15)

### Metal kernel backend — faster than Ollama on Apple Silicon

New hand-written Metal compute-kernel backend (`metal-fast` FP16,
`metal-q8` INT8, `metal-q4` INT4 weight-only; runtime-compiled MSL, no
MPSGraph) replacing per-token graph execution with one fused command
buffer per token. INT8/INT4 quantization follows the CUDA backend's
scheme; activations stay FP16 with FP32 accumulation everywhere; all
reductions are fixed-order, so output remains bit-exact run-to-run.

`metal-q4` becomes the default backend on macOS for `run` and the HTTP
API (`metal` — the FP32 MPSGraph parity anchor — and every other
backend remain selectable via `--backend` / `hardware.backend`, which
now also accepts the new names in engine.json).

Measured on an M4 Pro (median decode, 256-token generations, same
client-side methodology for both engines) vs Ollama 0.32.5 q4_K_M:

- Qwen2.5-0.5B: **278 tok/s, TTFT 42 ms** (Ollama: 246 tok/s, 103 ms;
  previous engine best: 87 tok/s FP32 short-ctx)
- Qwen2.5-1.5B: **163 tok/s, TTFT 90 ms** (Ollama: 154 tok/s, 101 ms)

Parity tests (`metal_fast_parity_test`) gate the path: FP16 tracks the
CPU reference within tolerance on a teacher-forced trajectory, the
quantized modes within quantization tolerance, and all modes are
deterministic across runs.

## v1.0.2 (2026-08-15)

Robustness fixes for the HTTP compat API (`serve`) and its CLI clients.

- The server now ignores SIGPIPE, so a client aborting a stream can no
  longer kill the process.
- Generation stops as soon as a streaming client disconnects instead of
  running the request to completion.
- The CLI accepts the model names that `/v1/models` and `/api/tags`
  report, so listed names can be passed back to `run`/`generate` as-is.

## v1.0.1 (2026-08-12)

- `GET /api/version` now identifies the engine: the response gains an
  `"engine":"lykuro-native-engine"` field alongside `"version"`.

## v1.0.0 (2026-08-11)

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
- **Stability — both 24h soaks passed.** CUDA: 237,009 requests, 0 failed,
  RSS bit-identical, 100 reload cycles leak-free. Metal: 137,440 completed,
  0 failed, phys_footprint flat (~2656MB over 24h) after fixing a
  per-request autorelease leak; 100 reload cycles leak 11.7MB. Per-request
  bookkeeping is bounded by in-flight work.
- **Unified Memory admission (Metal).** Load-time weight admission plus
  runtime staged watermarks — soft sheds new sequences, hard caps KV
  growth — with committed-KV accounting.
- **Supply-chain gate.** `tools/scan_vulnerabilities.sh` cross-checks the
  SBOM against a VEX ledger and OSV, fails closed on unreviewed deps or
  un-waived CRITICAL/HIGH; emits `vulnerability-report.json` in CI.

### Ollama-style unified commands

Every operation is a `native-engine <subcommand>`:
- `pull <hf_repo> [out]` — download a HF checkpoint and convert it to a
  Lykuro artifact **natively (no Python)**.
- `run <model_or_repo> ["prompt"] [--backend … --max-tokens N --temperature T
  --system "…"]` — config-less inference. Accepts a local artifact dir OR a
  HF repo id, which auto-pulls if not already local (single-command run).
  Streams output, or starts an interactive chat with no prompt.
- `serve --config <path>` — the gRPC engine server (legacy `--config` still
  works).
- `convert <hf_dir> <out_dir>` — HF checkpoint → artifact.
- `serve --http [--port 11434]` — Ollama- and OpenAI-compatible HTTP API
  (/api/generate, /api/chat, /api/tags, /api/pull; /v1/chat/completions,
  /v1/completions, /v1/models). Models given by HF repo id auto-pull.

### Packaging

- `linux-cuda` and `macos-metal` profiles via `tools/make_package.sh`:
  staged tree, sorted checksums, provenance manifest, Ed25519-signed
  manifest, deterministic tarball. SBOM (SPDX 2.3) and full license
  texts included.
- **Single self-contained binary.** gRPC/protobuf/abseil/OpenSSL are
  statically linked (`release-static` preset, `third_party/build_grpc_static.sh`);
  macOS links only `/usr/lib` + Apple frameworks, Linux the C/C++ runtime
  + system OpenSSL + CUDA. A cross-platform gate
  (`tools/check_selfcontained.sh`) fails the build on any forbidden dynamic
  dependency.
- **macOS install (§24).** pre/postinstall prechecks, `enable_service.sh`
  (non-root `_lykuro` LaunchDaemon), and `uninstall.sh`. Two-phase signing:
  Phase 1 `--dev` ad-hoc (internal test), Phase 2 Developer ID + notarize.
- **Operations runbook** (`docs/operations/runbook.md`): deploy, monitor,
  Drain/Resume update, versioned-package rollback, recovery.

### Known limitations / not certified

- **This is a pre-release.** Public signed binaries are not attached:
  macOS Developer ID signing + Apple notarization (Phase 2,
  downloads.lykuro.ai) await Apple Developer Program enrollment. The
  sign/notarize/staple pipeline is implemented and self-skipping until the
  certificate is present. Build from source with the `release-static`
  preset, or use the internal Phase 1 `--dev` ad-hoc package.
- Certified Profiles are **dev-measured**, not production-certified:
  formal security review, signed-artifact-only measurement, and
  cross-host variance data are pending (the 24h soaks themselves passed).
- Custom Metal kernels (precompiled `metallib`) require an Xcode CI
  runner — the MVP is MPSGraph-only.
- Hardware/OS coverage is limited to the entries in
  `docs/compatibility-matrix.md`. Sharded weights, MoE, embeddings,
  vision, and NCCL-based multi-GPU are out of scope for this release.
- Quantized (INT8/INT4/FP16) models change greedy output relative to the
  FP32 oracle by design; per-model quality gating belongs to the offline
  evaluation pipeline.

See `docs/DEFINITION_OF_DONE.md` for the full per-item status against
both specifications.
