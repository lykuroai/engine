# Lykuro Native Inference Engine

LYK-NIE-SD-001 v1.0 に基づく独自推論エンジン。第三者推論 Runtime
(Ollama / llama.cpp / vLLM / TGI) のコードを一切含まない。

仕様: `docs/claude_code_lykuro_native_inference_engine_complete_spec_v1_0.md`
Phase 0 報告: `docs/phase0-report.md`

## Build

```sh
cmake --preset dev -DLYKURO_ENABLE_GRPC=ON    # Debug + ASan/UBSan + gRPC
cmake --build --preset dev
ctest --preset dev

cmake --preset release -DLYKURO_ENABLE_GRPC=ON
cmake --build --preset release
```

必要 toolchain: CMake ≥3.24, Ninja, C++20 compiler。
gRPC 有効時: protobuf + grpc + OpenSSL(dev 環境では Homebrew 可)。
GoogleTest は dev 専用依存。

## 実行

```sh
# artifact の署名(release pipeline の代替、dev 用)
./build/release/tools/sign_artifact keygen signer
./build/release/tools/sign_artifact sign signer.key /models/current

# engine 起動
./build/release/cmd/native-engine/native-engine --config engine.json
```

config は strict JSON(§26 相当、unknown key 拒否)。`mtls_required: true`
では server cert / key / client CA のパスが必須。`trusted_signing_keys`
(Ed25519 公開鍵 hex)か `allow_unsigned_dev`(開発専用)のどちらかが
なければ起動を拒否する(fail-closed)。

## 実装状況(2026-08-07)

| Phase | 状態 |
|---|---|
| 0 Feasibility/Contract | 完了(proto・schema・build・policy) |
| 1 Correctness PoC | 完了(manifest/loader/tokenizer/CPU reference forward/golden) |
| 2 Engine MVP | 完了(scheduler/KV/streaming/cancel/deadline/gRPC Data+Control/CountTokens/metrics) |
| 3 Hardening | CPU 検証可能分は完了(Ed25519 署名検証、mTLS + service identity 認可、fail-closed config) |
| 4 Performance | 未着手(GPU 環境必須) |
| 5 Expansion | 未着手 |

テスト: 164 件(unit / golden / integration / gRPC e2e / mTLS)、
dev(ASan/UBSan)・release 両構成で全て成功。

## 未実装・未検証(spec §35 に基づく明示)

- **CUDA backend**: 開発環境が macOS のため未実装・未検証。
  `backends/cuda/` は Linux + `LYKURO_ENABLE_CUDA=ON` 用の placeholder。
  実 GPU での benchmark / certified profile 数値は存在しない。
  推論は correctness oracle 用の CPU reference(FP32)で動作する。
- **外部 oracle との correctness 照合**: golden test は本実装の再現性
  anchor であり、HF transformers 等の独立 oracle との照合は実 Qwen
  checkpoint 入手後に実施する。
- **sharded safetensors**: MVP は単一 shard のみ(複数 shard は拒否)。
- **prefix cache / paged KV / 量子化 / multi-GPU / MoE / embeddings**:
  Phase 4〜5 の対象。
- **soak(24h)/ fuzz / GPU OOM recovery**: GPU 環境および長時間実行
  環境が必要なため未実施。
- **SBOM / signed OCI image / Kubernetes manifest**: release pipeline
  整備時に実施(§27)。

## Repository 構成

spec §4.1 のとおり。`api/proto` が Data/Control API contract、
`model/architectures/qwen` が approved_qwen_decoder_v1 の
CPU reference 実装。`security/` に自前 SHA-256 と OpenSSL ベースの
Ed25519 署名検証。`tests/fixtures/certs` はテスト専用 CA(本番不使用)。
