# Lykuro Native Inference Engine

LYK-NIE-SD-001 v1.0 に基づく独自推論エンジン。第三者推論 Runtime
(Ollama / llama.cpp / vLLM / TGI) のコードを一切含まない。

仕様: `docs/claude_code_lykuro_native_inference_engine_complete_spec_v1_0.md`
Phase 0 報告: `docs/phase0-report.md`

## Build

```sh
cmake --preset dev                       # Debug + ASan/UBSan
cmake --preset dev -DLYKURO_ENABLE_GRPC=ON   # + gRPC API server
cmake --build --preset dev
ctest --preset dev
```

必要 toolchain: CMake ≥3.24, Ninja, C++20 compiler。
gRPC 有効時: protobuf + grpc (dev 環境では Homebrew 可)。
GoogleTest は dev 専用依存。

## 実装状況(2026-08-07)

| Phase | 状態 |
|---|---|
| 0 Feasibility/Contract | 完了(proto・schema・build・policy) |
| 1 Correctness PoC | 完了(manifest/loader/tokenizer/CPU reference forward/golden) |
| 2 Engine MVP | コア完了(scheduler/KV/streaming/cancel/deadline/gRPC Data+Control) |
| 3 Hardening | 未着手 |
| 4 Performance | 未着手 |
| 5 Expansion | 未着手 |

## 未実装・未検証(spec §35 に基づく明示)

- **CUDA backend**: 開発環境が macOS のため未実装・未検証。
  `backends/cuda/` は Linux + `LYKURO_ENABLE_CUDA=ON` 用の placeholder。
  実 GPU での benchmark / certified profile 数値は存在しない。
- **manifest.sig 署名検証**: Phase 3。未署名 artifact は
  `allow_unsigned_dev` フラグなしでは load 不可(fail-closed)。
- **mTLS / service identity 認可**: Phase 3。現状の gRPC server は
  credentials 注入構造のみ(テストは loopback insecure)。
- **metrics HTTP endpoint / diagnose bundle**: 未実装。
- **CountTokens RPC**: 未実装(UNIMPLEMENTED を返す)。
- **sharded safetensors / prefix cache / paged KV / 量子化 / multi-GPU**:
  Phase 2 以降の対象外機能。
- **外部 oracle との correctness 照合**: golden test は本実装の
  再現性 anchor であり、HF transformers 等の独立 oracle との照合は
  実 Qwen checkpoint 入手後に実施(Phase 1 完了条件の残項目)。

## Repository 構成

spec §4.1 のとおり。`api/proto` が Data/Control API contract、
`model/architectures/qwen` が approved_qwen_decoder_v1 の
CPU reference 実装(FP32、correctness oracle 用)。
