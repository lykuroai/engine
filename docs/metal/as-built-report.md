# Metal Backend Phase 0 — As-Built 調査報告

文書: LYK-NIE-ADD-METAL-001 §0.2 に基づく実装開始前調査。2026-08-08 時点。

## 1. Repository・Build・Toolchain

| 項目 | As-Built |
|---|---|
| Repository | `lykuroai/engine`(main、32+ commits) |
| Build | CMake ≥3.24 + Ninja、C++20、preset(dev/release/release-cuda) |
| 言語 | C++20(core)、CUDA C++(Linux backend)、Metal 追加は Objective-C++(.mm) |
| Gate | `LYKURO_ENABLE_METAL`(新設、macOS arm64 のみ) |

## 2. Hardware Backend Interface と CUDA Backend の現行実装

既存の backend 抽象は **`GenerativeModel` interface**(`model/architectures/generative_model.h`)である。仕様 §8 の論理 `HardwareBackend` に対して、本実装は model 単位の抽象(CreateSequence / Prefill / Decode / DecodeBatch、host FP32 logits 返却)を採用しており、CUDA 実装 `QwenCudaModel`・TP 実装 `QwenTpModel`・CPU reference `QwenModel` が同一契約で挿さる。**Metal も同じ境界に `QwenMetalModel` を追加する(同義 Interface を新設しない、§8 準拠)**。

デバイス検査は `backends/cuda/cuda_backend.h` の Discover/Check に相当する Metal 版(MTLDevice / hasUnifiedMemory / recommendedMaxWorkingSetSize)を新設する。

## 3. Memory Ownership

- KV cache: sequence 単位所有(SequenceState、destruction で解放)。CUDA は paged pool + block table。Metal MVP は仕様どおり bounded contiguous(Unified Memory 上、MTLBuffer storageModeShared)から開始
- Weight: model 所有、load〜unload。Activation/scratch: model 所有 work buffer(単一 worker thread 前提)

## 4. Architecture Plugin と対象 model

`approved_qwen_decoder_v1`(Qwen2 系 Dense)。CPU reference(FP32 oracle)、CUDA、TP で検証済み。実 checkpoint(Qwen2.5-0.5B-Instruct、BF16)の変換・oracle 照合パイプラインが確立済みで、**同一 artifact・同一 oracle JSON を Metal 検証にそのまま再利用できる**。

## 5. Data API / Control API / Error Code / Version

`lykuro.nie.v1` proto(Data: Generate/Stream/Cancel/CountTokens/Capabilities、Control: Load/Unload/Drain/Resume/Status/Capacity/Manifest)。engine は backend 非依存なので **API 変更は不要**(§5.1 不変部分を維持)。error code は §21 の Metal 追加 code を `ErrorCode` enum + proto へ追加予定(後方互換の追記のみ)。

## 6. Apple Silicon 検証機

| 項目 | 実測 |
|---|---|
| Chip | **Apple M4 Pro** |
| Unified Memory | **64 GB**(hasUnifiedMemory = true) |
| recommendedMaxWorkingSetSize | **55.7 GB** |
| macOS | 26.5.2 (25F84) |
| Toolchain | CommandLineTools のみ(**Xcode / `metal` compiler 無し**) |

仕様の主検証プロファイル(M4 / 64GB)に合致。

## 7. MTLDevice / Metal feature / MPSGraph

Phase 0 probe(2026-08-08 実行)で確認:

- `MTLCreateSystemDefaultDevice` 成功、unified memory = true
- **MPSGraph matmul を GPU 実行し host 計算と完全一致(max_diff = 0)**
- → Phase 0 終了条件「共通 Core を変更せず 1 operator を実機実行」達成

`metal` shader compiler が無いため、**custom Metal Kernel(metallib 事前 compile)は本開発機では作成不能**。仕様の実装優先順(§12.2: MPSGraph → Metal Kernel → MLX → CPU fallback)に従い、**MVP は MPSGraph のみで構成**する。metallib を要する custom kernel は未実装項目として明記する(production 配布要件「事前 compile 済み metallib」には Xcode を持つ CI runner が必要)。

## 8. MLX C++

不使用(MVP は MPSGraph で完結)。oracle は既存 HF transformers FP32(dev 限定)を再利用。

## 9. Package / Signing / Notarization

**未実施**。Developer ID 証明書・notarization 環境が本開発機に無い。§23〜25(pkg、launchd、Hardened Runtime、notarization)は Phase 3 項目として未実装・未検証と明記する。

## 10. Golden 比較方法

既存資産を再利用: (a) tiny fixture の CPU parity テスト(tolerance 1e-3、FP32 時)、(b) 実 checkpoint + `tools/verify_reference`(HF FP32 oracle、logits 2e-2 / greedy 32 token)。Metal backend を `verify_reference` の backend として追加する。FP16 化時は operator 別 tolerance を再定義(§14.1)。

## 11. 実機 CI / benchmark / soak

本開発機(M4 Pro)がそのまま Mac 実機 runner に相当。`bench_decode` / `soak_engine` は backend 引数拡張で再利用可能。

## 12. 論理 Component 対応表(Reuse / Extend / New)

| 仕様 Component | 対応 | 既存/新規 |
|---|---|---|
| Engine Core / Scheduler / KV(論理) / Streaming | Reuse | `core/` 無変更 |
| Data / Control API | Reuse | `api/` 無変更 |
| Model Artifact / Manifest / 署名 | Reuse | `model/`, `security/` 無変更 |
| Backend 境界 | Reuse | `GenerativeModel`(既存抽象) |
| Metal device discovery / capability | New | `backends/metal/metal_backend.{h,mm}` |
| Metal model(MPSGraph forward) | New | `backends/metal/qwen_metal_model.{h,mm}` |
| Unified Memory budget / admission | New | Metal backend 内 + Capacity 拡張 |
| Metal error code | Extend | `core/engine/error.h` + proto(追記) |
| Config(metal section) | Extend | `core/engine/config.*`(追記) |
| macOS package / launchd / notarization | New(Phase 3) | 未着手 |
| custom Metal Kernel / metallib | New(Phase 4) | 未着手(toolchain 制約) |

## Phase 0 結論

- 共通 Core への破壊的変更は不要(既存 backend 抽象がそのまま使える)
- MPSGraph 経路で Phase 1(correctness PoC)へ進行可能
- 制約: custom kernel は toolchain 不足で不可、signing/notarization は環境なし — いずれも未実装として管理
