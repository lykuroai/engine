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
| 2 Engine MVP | 完了(scheduler/KV/streaming/cancel/deadline/gRPC Data+Control/CountTokens/metrics/**CUDA backend**) |
| 3 Hardening | CPU 検証可能分は完了(Ed25519 署名検証、mTLS + service identity 認可、fail-closed config) |
| 4 Performance | 未着手(最適化 kernel・batching 強化は今後) |
| 5 Expansion | 未着手 |

テスト: macOS 164 件(ASan/UBSan・release 両構成)+
Linux/GPU **173 件**(CUDA + gRPC + OpenSSL フル構成)、全て成功。
検証環境: Ubuntu 22.04、RTX 3060(sm_86)/ GTX 1650(sm_75)、
CUDA 12.8、driver 570.211、gRPC v1.66(system OpenSSL 3.0)。

CUDA backend の検証内容:
- CPU reference との logits parity(≤1e-3)、greedy 軌跡一致
- bit-exact な run-to-run 決定性
- gRPC 経由の GPU serving 出力が CPU serving と完全一致
- production 形態(signed artifact + mTLS + CUDA + metrics)での
  `native-engine` バイナリ起動・モデルロード・graceful shutdown

## 外部 oracle 照合(実 Qwen checkpoint、2026-08-08 実施)

Qwen2.5-0.5B-Instruct(BF16、988MB)を `tools/convert_hf_qwen.py` で
artifact 化し、HF transformers FP32 を oracle として照合
(`tools/verify_reference`、英語・数式・日本語の 3 プロンプト):

| Backend | prefill logits 最大誤差 | argmax | greedy 32 token |
|---|---|---|---|
| CUDA(RTX 3060) | ≤2e-5 | 3/3 一致 | **3/3 完全一致** |
| CPU reference | ≤9e-5 | 3/3 一致 | **3/3 完全一致** |

注: oracle は純粋 greedy(HF checkpoint 同梱の repetition_penalty=1.1
は無効化。本 Engine は仕様 §18.1 どおり MVP では repetition penalty
非対応)。

## 未実装・未検証(spec §35 に基づく明示)

- **certified profile の正式発行**: Phase 4 最適化(bf16 weight 常駐 +
  fp32 accumulate 自前 GEMV、cuBLAS 撤去)後の実測(Qwen2.5-0.5B、
  single sequence、dev 計測値):

  RTX 3060(sm_86)、chunked GEMM prefill + bf16 GEMV decode +
  split-K(flash-decoding 方式)attention:

  | prompt tokens | TTFT | decode tok/s |
  |---:|---:|---:|
  | 19 | 41 ms | 152 |
  | 329 | 423 ms | 146 |
  | 1289 | 2.2 s(prefill 584 tok/s) | 136 |
  | 2569 | 5.9 s | 125 |

  multi-sequence batched decode(aggregate、RTX 3060):

  | batch | aggregate tok/s |
  |---:|---:|
  | 1 | 152 |
  | 4 | 251 |
  | 8 | 285 |
  | 16 | 450 |

  GTX 1650(sm_75): 短 prompt で decode 67 tok/s。
  最適化前(fp32 weight + cuBLAS SGEMV、逐次 prefill、単純 attention)
  は decode 約 18 tok/s、TTFT 85 ms(18 tok)、ctx 1289 で 58 tok/s。
  数値は §25.3 の完全な benchmark(soak / saturation)を経ていないため
  certified profile としては未発行。

  **Paged KV / prefix cache(§16.3)実装済み**: 64 token block pool +
  per-sequence block table(物理割当はオンデマンド、sequence あたりの
  事前確保を撤廃)。prefix cache は request の `allow_prefix_cache`
  opt-in(既定 OFF、§16.2)で、chain hash は tenant/project scope を
  seed に含むため cross-scope 再利用は構造的に不可能(§16.1)。
  refcount + LRU eviction。残る Phase 4 項目は INT8/INT4 量子化。
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
