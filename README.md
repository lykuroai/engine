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
| 4 Performance | 主要項目完了(bf16常駐/fusion/graphs/paged KV/prefix cache/量子化) |
| 5 Expansion | **2-GPU tensor parallel(correctness PoC)実装・実機検証済み** |

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

## Metal Backend(LYK-NIE-ADD-METAL-001、Phase 0-1 完了)

Apple Silicon 向け MPSGraph backend(`backends/metal/`、
`LYKURO_ENABLE_METAL=ON`)。既存 `GenerativeModel` 境界への追加で
Core/API は無変更(§5.1)。実機 **Apple M4 Pro / 64GB Unified Memory**
(仕様の主検証プロファイル)で検証:

- Phase 0: As-Built 報告(`docs/metal/as-built-report.md`)+ MPSGraph
  1 operator 実機実行
- Phase 1: FP32 compute の correctness PoC — tiny model CPU parity
  (1e-3、greedy 20 step 一致、決定性)、**実 Qwen2.5-0.5B の oracle
  3/3 完全一致(logits ≤5e-5、greedy 96/96)**
- 構成: weights は unified memory の MTLBuffer を placeholder 直結
  (コピー無し)、context bucket(128)別に graph をキャッシュ、
  RoPE cos/sin は host 供給、KV は bounded contiguous(§16 MVP)

**Phase 2(engine/serving 統合)完了**: `hardware.backend: metal` で
gRPC serving(CPU serving と出力一致を e2e テストで確認)、engine の
streaming/cancel/deadline を Metal 上で検証、load 前の Unified Memory
admission(working set × (1−15%) budget との fail-closed 照合、§10)、
macOS ネイティブ production 形態(signed artifact + mTLS + Metal +
metrics + graceful shutdown)での `native-engine` 起動確認。

**Phase 4(性能)第一弾完了**(Apple M4 Pro、Qwen2.5-0.5B、FP32):
feeds 辞書の per-token 構築を排除(常駐 shared MTLBuffer 直書き +
bucket 別キャッシュ)、resultsDictionary で staging buffer へ直接出力
(readBytes×49 排除)、multi-token prefill graph(P=32、末尾は
overlap chunk で logits 行を固定)、load 時 graph pre-warm。

| 指標 | 最適化前 | 最適化後 |
|---|---:|---:|
| decode(短 ctx) | 72 tok/s | **87 tok/s** |
| TTFT(39 tok prompt) | 1330 ms | **43 ms** |
| prefill 実効(329 tok) | ~24 ms/tok | **~153 tok/s** |

decode は ctx bucket 拡大で逓減(bucket256 で ~48 tok/s)。
計測知見: 「chunked prefill 後の decode 劣化」に見えた現象は decode
graph 初回コンパイルの計上位置の差であり、pre-warm で解消。

Metal 未実装項目: FP16 化(operator 別 tolerance 定義とセット)、
watermark 段階制御(warning/critical、§10.4)、custom Metal Kernel
(本開発機に metal compiler 無し)、launchd/pkg/Developer ID 署名/
notarization(Phase 3 — 環境なし)、Metal error code の proto 反映、
decode の bucket 逓減対策・batched decode(Phase 4 続き)。

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
  refcount + LRU eviction。

  **INT8/INT4 weight-only 量子化(Phase 4)実装済み**: projection 重みを
  load 時に量子化(INT8: per-row absmax、INT4: 128 要素 group scale +
  nibble pack)。embed/lm_head は checkpoint dtype 維持。activation と
  accumulation は FP32 のまま、決定性も維持。実測(Qwen2.5-0.5B、
  RTX 3060):

  - 重み VRAM: bf16 988MB → int8 約 630MB → int4 約 451MB
  - 品質(FP32 oracle 比): int8 は argmax 3/3 維持だが greedy は途中
    分岐(logit 摂動の複利。量子化モデルの品質判定は offline 評価
    pipeline の責務であり、oracle 完全一致は原理的に失われる)。
    int4 は argmax flip も 1 件発生
  - 速度: 横ばい(148.6 → 148.4 / 142.9 tok/s)。**この構成の decode は
    帯域律速ではなくカーネル起動オーバーヘッド律速**(実効 146GB/s ≪
    ピーク 360GB/s)であることが判明。量子化の速度便益はより大きな
    model か CUDA Graphs 導入(今後の課題)とセットで発現する

  **CUDA Graphs(decode fast path)実装済み**: per-token 入力(token /
  position / block table pointer)を全て device メモリ経由にし、decode
  pipeline を (batch bucket {1,2,4,8,16} × splits bucket {1,8,32}) キーで
  lazy capture・replay。pad 行は予約 scratch block に書き実データを
  汚さない。実測では throughput は同等〜微増(batch1: 153 tok/s、
  batch16: 448 tok/s)で、**律速はカーネル起動 API ではなく、依存
  チェーン上の小カーネル実行レイテンシ**と判明(per-token の host 側
  作業は約 290 launch → memcpy 3 回 + graph launch 1 回に削減され、
  engine スレッドの CPU 余力は大幅改善)。次の速度改善は kernel fusion。

  **Kernel fusion(decode、batch ≤ 8)実装済み**: QKV 3 projection を
  1 launch に融合(RMSNorm は per-sequence scale の事前計算 + 積み込み
  fold)、gate/up/SwiGLU を 1 カーネル化、residual add を projection
  epilogue に融合、RoPE q+k / K+V scatter 統合、head へ final norm
  fold。依存チェーンは ~18 → ~10 kernels/layer。実測(RTX 3060):

  | batch | fusion 前 | fusion 後 |
  |---:|---:|---:|
  | 1 | 153 | **169**(+10%) |
  | 2 | 182 | **242**(+33%) |
  | 8 | 286 | **335**(+17%) |
  | ctx1289 (b1) | 138 | **150**(+8%) |

  int8 も 171 tok/s と bf16 超え(チェーン短縮で帯域削減が発現)。

  **2-GPU tensor parallel(Phase 5、correctness PoC)実装済み**:
  Megatron 方式(Q/K/V head 分割・KV cache は各 GPU が自 head 分のみ
  所有、o_proj/down_proj 列分割 + host 固定順 all-reduce、gate/up 行
  分割)。embed/lm_head は device 0 常駐。分割不能な形状は
  unsupported_model で拒否。**実機の異種ペア(RTX 3060 + GTX 1650)で
  実 Qwen 0.5B の oracle 3/3 完全一致(logits ≤2e-5、greedy 96/96)、
  bit-exact 決定性を確認**。PCIe + host reduce のため速度は単一 GPU 比
  で遅く(このモデルでは意図どおり)、目的は TP アーキテクチャの
  正しさ検証と単一 GPU VRAM 超モデルへの布石。NCCL 化・per-shard
  GEMM 最適化は今後。

  残項目: 24h soak、offline 量子化 pipeline、bucket16 GEMM fusion、
  NCCL ベース TP、CPU/AMD serving。
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
