# Phase 0 報告 — Feasibility・Contract 確定

文書: LYK-NIE-SD-001 §0.1 に基づく実装開始前報告。2026-08-07 時点。

## 1. Repository・Build・Toolchain

| 項目 | 内容 |
|---|---|
| Repository | `lykuro-native-inference-engine`(本repo、新規、branch: `main`) |
| 言語 | C++20(Engine core)、CUDA C++(GPU backend、Linux限定) |
| Build | CMake ≥3.24 + Ninja、`CMakePresets.json` で reproducible preset(`dev` / `release` / `release-cuda`) |
| Compiler(開発機) | Apple clang 21.0.0(macOS / Darwin 25.5) |
| Compiler(production) | Linux GCC 12+ / Clang 16+(CI で確定) |
| CUDA toolchain | **開発機(macOS)には存在しない**。CUDA backend は `LYKURO_ENABLE_CUDA=ON` の Linux build でのみ有効。macOS では interface + stub のみビルド |
| Sanitizer | Debug build で ASan/UBSan 既定 ON |

## 2. Dependency 一覧・License・SBOM

方針: allowlist + version pin + vendor不使用(Phase 1 は標準ライブラリのみ)。

| Dependency | 用途 | 区分 | License |
|---|---|---|---|
| C++20 標準ライブラリ | 全般 | production | — |
| CUDA Toolkit / cuBLAS | GPU backend | production(Linux、承認済み低レベルSDK) | NVIDIA EULA(SBOM記載) |
| gRPC + Protocol Buffers | Internal API | production(Phase 2 で導入、version pin) | Apache-2.0 |
| GoogleTest | test | dev only | BSD-3-Clause |
| CMake / Ninja | build | dev only | — |

- JSON parser・SHA-256・safetensors parser は**自前実装**(第三者推論Runtime由来コードのコピー禁止、依存最小化のため)。
- 第三者推論 Runtime(Ollama / llama.cpp / vLLM / TGI)は source・binary・transitive いずれも不使用。CI で SBOM grep により検査する。
- SBOM: release build で SPDX 形式生成(Phase 3 で CI 組込み)。
- dynamic download 禁止: runtime の network egress は設計上存在しない(fetch 系コードを持たない)。

## 3. Target GPU・Driver・Compatibility

- MVP target: NVIDIA CUDA、単一GPU、明示 device ID。
- **開発機に GPU がないため、GPU 実行・benchmark・certified profile 数値は本環境では未検証**。Linux + NVIDIA 環境が別途必要(§35 の未検証報告に従う)。
- 本環境で検証可能: manifest/loader/tokenizer/scheduler/sampler/CPU reference forward の correctness。

## 4. Model Manager / Gateway との API Contract

- `api/proto/lykuro/nie/v1/` に定義: `common.proto`(error code・finish reason・role)、`data.proto`(Generate / GenerateStream / Cancel / CountTokens / GetCapabilities)、`control.proto`(LoadModel / UnloadModel / Drain / Resume / GetModelStatus / GetCapacity / GetManifest)。
- Control API は Model Manager の service identity のみ。Gateway からの Control 呼出しは `authentication_failed`。
- Transport: mTLS gRPC(Phase 2〜)。health/metrics は管理 network 限定 HTTP。public bind 禁止。
- Versioning: proto package `lykuro.nie.v1`、Platform ↔ Engine は N/N-1。

## 5. Model 対象範囲

- Family: Qwen 系 Dense Decoder-only(architecture ID: `approved_qwen_decoder_v1`)。
- Weight: Safetensors(single / sharded)。Precision: BF16 / FP16(CPU reference は FP32 で計算)。
- Manifest schema: `api/schema/model-manifest.schema.json`(schema_version "1"、unknown field 拒否)。

## 6. Artifact 署名・Digest 検証

- `checksums.sha256` + manifest 内 per-file SHA-256 を自前 SHA-256 実装で検証。
- `manifest.sig`: Ed25519 署名を trusted key で検証(Phase 3 で鍵管理と共に実装。Phase 1 は digest 検証 + 署名検証の interface stub を持ち、**署名未検証の状態では production load を拒否する設計**)。
- path traversal / absolute path / symlink / executable / unknown field は loader で拒否。

## 7. Thread・Queue・Memory Ownership 設計

- 論理 thread: API I/O → bounded Admission Queue → Scheduler thread → Tokenizer worker pool → GPU worker(device ごとに1本、command 投入を集約)→ Streaming output threads → Metrics thread。
- 全 queue bounded。request ownership は Request Registry が一意管理し、cancel/deadline を各 stage で確認。
- GPU memory: weight / KV / workspace の pool 分離。KV cache は sequence 単位 ownership + tenant/project scope metadata。release は idempotent。

## 8. Test Oracle・Golden Vector・Benchmark

- Unit: GoogleTest(dev dependency)。loader negative・tokenizer round-trip・sampler・scheduler・config を対象。
- Golden: 固定 prompt → expected token IDs / greedy output。reference oracle は開発環境限定で HuggingFace transformers(Python、dev only)により生成し、`tests/golden/` に静的 fixture として commit。production package に Python を含めない。
- Benchmark: GPU 環境確保後に §25.3 の項目を実施。本環境では実施しない(未実施と報告する)。

## 9. Phase ごとの対象・非対象

| Phase | 対象 | 非対象 |
|---|---|---|
| 0(本報告) | repo・build・proto・schema・policy 文書 | 推論実装 |
| 1 | manifest/loader、tokenizer/template、Qwen CPU reference forward、greedy、golden test | GPU 実行、streaming、batching |
| 2 | scheduler、KV cache、sampling、streaming、gRPC server、CUDA backend(Linux) | 量子化、prefix cache、multi-GPU |
| 3 | mTLS、署名検証、OOM recovery、soak/fuzz、signed package | — |
| 4〜5 | performance、拡張 | — |

## 10. Security Threat Model・Trust Boundary

§5.1 のとおり。要点:

- Model Artifact は**未信頼入力**として扱う(signature → digest → schema → size/offset → shape/dtype の順で検証、いずれか失敗で load 拒否)。
- Engine API は mTLS + service identity。egress deny by default(network client コード自体を持たない)。
- Prompt/Response/Token/KV を log・metrics・trace・disk・Control Plane へ出さない。error message は静的文字列 + 数値 detail のみ。
- production: non-root、read-only rootfs、capability drop、core dump 制限。

## 未決事項(§34 対応)

D-01〜D-14 は推奨初期値を採用して実装を進めるが、GPU 実測と security review 完了までproduction 確定としない。
