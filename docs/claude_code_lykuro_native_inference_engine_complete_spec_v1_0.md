# Lykuro Native Inference Engine 完全仕様書

| 項目 | 内容 |
|---|---|
| 文書番号 | LYK-NIE-SD-001 |
| 版 | v1.0 Claude Code Edition |
| 制定日 | 2026-08-07 |
| 作成 | 株式会社eビジネスソリューション / Lykuro.ai |
| Project | lykuro-native-inference-engine |
| 対象 | Native Inference Engine本体、Model Architecture Plugin、GPU Backend |
| 機密区分 | 社内・提案先限定 |
| 実装担当 | Claude CodeおよびNative Engine開発担当者 |
| 関連文書 | LYK-NLP-SD-001、LYK-PLG-BD-001、LYK-PLG-ADD-001 |

---

## 0. Claude Codeへの実装指示

本書は、Lykuro Native Inference Engineを独立projectとして開発するための完全仕様である。

Claude Codeは、既存repositoryが存在する場合は実装前に全体構成を調査し、本書との対応表を作成する。新規repositoryの場合も、Phase 0でAPI、build、dependency、test、security方針を確定してから推論実装へ進むこと。

### 0.1 実装開始前の必須報告

1. repository、branch、build system、言語、compiler、CUDA toolchain
2. dependency一覧、license、link方式、SBOM生成方法
3. target GPU、driver、CUDA compatibility matrix
4. Model Manager、GatewayとのAPI contract
5. model family、architecture、weight format、precisionの対象範囲
6. model artifact署名・digest検証方式
7. thread、queue、GPU worker、memory ownership設計
8. test oracle、golden vector、benchmark方法
9. Phaseごとの実装対象と非対象
10. security threat modelとtrust boundary

### 0.2 絶対条件

- Ollama、llama.cpp、vLLM、TGIその他の第三者推論Runtimeを内蔵・再配布しない。
- 第三者推論Runtimeのsource、binary、container、scheduler、model loaderをコピーしない。
- Native EngineはLykuro独自sourceとして実装する。
- GPU driver、CUDA/ROCm、BLAS、NCCL等の低レベルSDKは承認済みdependencyとして利用できる。
- Production packageへPython、Node.js、npm、nvmを必須dependencyとして含めない。
- EngineはPublic OpenAI互換API、tenant認証、課金、Conversation Memoryを持たない。
- EngineのAPIはLykuro Local LLM Platformからのみ利用する。
- Prompt、Response、Token、KV CacheをLykuro Control Planeへ送信しない。
- PromptとResponseの通常logを禁止する。
- 署名・digest・license・architecture検証を通過していないmodelをloadしない。
- 未対応modelをbest effortで実行しない。
- 不明なshape、dtype、special token、architecture parameterを黙って推測しない。
- OOM、deadline超過、cancel、client切断時にresourceを確実に解放する。
- KV Cacheを会話記憶の正本にしない。
- 実行していないtestを成功と報告しない。

### 0.3 開発方針

- correctnessをperformanceより先に検証する。
- 一つのmodel family、weight format、GPU構成から開始する。
- model固有処理をArchitecture Pluginへ隔離する。
- core scheduler、memory、streaming、metricsをmodel非依存にする。
- production codeでremote code executionを許可しない。
- runtime中のpublic registry accessとdynamic dependency downloadを禁止する。
- errorを握りつぶさず、安定したerror codeへ変換する。

---

## 1. 目的

検証済みの企業ローカルmodelを顧客GPU/CPUへ安全にloadし、高性能かつ再現可能な推論を提供するLykuro独自Engineを開発する。

### 1.1 提供価値

- 第三者推論Runtimeへの製品依存を排除する。
- model load、scheduler、KV Cache、GPU memoryをLykuroが制御する。
- Strict Local Modeと署名済みmodel運用を標準化する。
- Lykuro Model Manager、Distributed Coordinator、R(m)評価と統合する。
- model familyごとの性能・品質をLykuroが認証する。

### 1.2 Native Engineが行うこと

- model artifact検証
- weight load
- tokenizerとchat template処理
- prefill/decode
- sampling
- streaming
- request scheduling
- batching
- KV Cache
- GPU memory管理
- health、capacity、metrics
- safe load/unload

### 1.3 Native Engineが行わないこと

- 外部利用者認証
- tenant契約・RBAC
- Virtual Key
- billing
- Conversation Memory
- RAG document管理
- MCP tool実行
- Public Model Registryからのdownload
- 未承認Cloud LLMへのfallback
- 第三者Runtimeのinstall/update

---

## 2. 対象範囲

### 2.1 MVP対象

| 項目 | MVP |
|---|---|
| Model family | Qwen系の承認済み1〜2 architecture |
| Model type | Decoder-only Dense Transformer |
| Weight | Safetensors |
| Precision | BF16 / FP16 |
| Hardware | NVIDIA CUDA |
| GPU count | 単一GPU |
| Input | normalized messagesまたはprompt |
| Output | text generation、streaming |
| Scheduler | bounded queue、priority、deadline、cancel |
| Batch | basic continuous batching |
| Cache | per-sequence KV Cache、prefix cacheは後続 |
| Deployment | signed OCI imageまたは承認済みbinary package |

### 2.2 Phase 2以降

- INT8/INT4量子化
- block-based/paged KV Cache
- prefix cache
- embedding model
- multiple model instances
- multi-GPU
- tensor/pipeline parallelism
- CPU/AMD backend
- MoE
- Vision/Audio
- speculative decoding
- structured output acceleration

### 2.3 対象外

- model training
- fine-tuning
- arbitrary Python model code
- arbitrary Hugging Face remote code
- plugin marketplace
- Internet越しのdistributed inference
- unverified weight conversion
- Phase 1での独自Flash Attention相当最適化

---

## 3. 用語

| 用語 | 定義 |
|---|---|
| Engine | Native Inference Engine process全体 |
| Model Artifact | manifest、weight、tokenizer、license参照を含む署名済み配布単位 |
| Architecture Plugin | model固有config、weight mapping、forward、tokenizer規則 |
| Model Instance | Engineへloadされ推論可能なmodelの実体 |
| Sequence | 一つの生成要求と生成途中のtoken状態 |
| Prefill | input tokenから初期KV Cacheを構築する処理 |
| Decode | 1 tokenずつ次tokenを生成する処理 |
| Batch | 同時にGPUへ投入するsequence集合 |
| KV Cache | Attention用の一時Key/Value memory |
| Admission | requestを実行可能としてqueueへ受け入れる判断 |
| Certified Profile | model、GPU、driver、precision、context等の検証済み組合せ |
| Control API | Model Managerがload/unload等に使用するinternal API |
| Data API | Gateway/Local Platformが推論に使用するinternal API |

---

## 4. Project・技術構成

### 4.1 Repository

~~~text
lykuro-native-inference-engine/
├── api/
│   ├── proto/
│   └── schema/
├── cmd/
│   └── native-engine/
├── core/
│   ├── engine/
│   ├── scheduler/
│   ├── batching/
│   ├── memory/
│   ├── generation/
│   └── streaming/
├── model/
│   ├── manifest/
│   ├── loader/
│   ├── tokenizer/
│   └── architectures/
│       └── qwen/
├── backends/
│   ├── cuda/
│   ├── rocm/
│   └── cpu/
├── security/
├── observability/
├── deploy/
│   ├── container/
│   └── kubernetes/
├── tests/
│   ├── unit/
│   ├── golden/
│   ├── integration/
│   ├── security/
│   ├── performance/
│   └── soak/
├── tools/
└── docs/
~~~

### 4.2 推奨言語

| 領域 | 推奨 |
|---|---|
| Engine core | C++20 |
| GPU backend | CUDA C++ |
| Build | CMake + reproducible preset |
| Internal contract | Protocol Buffers / gRPC |
| Health/Metrics | HTTP |
| Test tools | C++および開発環境限定のPython可 |

Production artifactへPython runtimeを含めない。最終言語はPhase 0のbenchmarkと既存資産調査で確定する。

### 4.3 Dependency policy

- allowlist方式
- version pin
- license review
- CVE scan
- SBOM
- source/binary provenance
- reproducible build
- dynamic download禁止
- unused dependency禁止

第三者推論Runtimeをtransitive dependencyとして取り込まない。

---

## 5. 全体アーキテクチャ

~~~mermaid
flowchart TD
    A["Lykuro Local LLM Platform"] --> B["Internal API Server"]
    B --> C["Admission / Scheduler"]
    C --> D["Tokenizer / Input Builder"]
    D --> E["Batch / Generation Engine"]
    E --> F["Architecture Plugin"]
    F --> G["CUDA Backend"]
    G --> H["GPU / KV Cache / Weight"]
~~~

### 5.1 Trust Boundary

| Boundary | 信頼 | 対策 |
|---|---|---|
| Local Platform → Engine | 認証済みserviceのみ | mTLS、service identity、scope |
| Engine → Model Artifact | 未信頼入力 | signature、digest、schema、size、shape |
| Engine → GPU Driver | OS dependency | compatibility matrix、health |
| Engine → Metrics | metadataのみ | content/secret禁止 |
| Engine → Network | deny by default | egress禁止、internal allowlist |

### 5.2 Runtime process

MVPは一つのEngine processにつき一つのactive Model Instanceを基本とする。複数model同時loadはPhase 2とする。

利点:

- failure domainが明確
- GPU memory予測が容易
- update/rollbackが容易
- model間のmemory競合を回避
- container resource設定が単純

---

## 6. Component設計

| Component | 責務 |
|---|---|
| Internal API Server | request受信、auth、schema、stream |
| Admission Controller | model、capacity、deadline、quota検査 |
| Request Registry | request state、cancel、deadline |
| Tokenizer Manager | tokenize、detokenize、special token |
| Prompt Builder | normalized messageからmodel templateへ変換 |
| Scheduler | queue、priority、fairness、batch候補 |
| Batch Manager | prefill/decode batch構成 |
| Generation Engine | forward、sampling、stop |
| KV Cache Manager | allocate、ownership、release、eviction |
| Model Loader | manifest、weight、shape、device load |
| Architecture Registry | certified plugin解決 |
| Hardware Manager | GPU、VRAM、driver、temperature、health |
| Metrics/Diagnostics | metrics、structured log、support bundle |

---

## 7. Process・Thread Model

### 7.1 論理Thread

~~~text
API I/O Threads
  -> Admission Queue
  -> Scheduler Thread
  -> Tokenizer Worker Pool
  -> GPU Worker per Device
  -> Output / Streaming Threads
  -> Metrics Thread
~~~

### 7.2 原則

- queueはすべてboundedとする。
- request ownershipを一意にする。
- GPU command投入をdevice workerへ集約する。
- cancel/deadlineを各stageで確認する。
- streaming clientの遅延でGPU workerをblockしない。
- shutdown時はdrain、cancel、resource releaseを順序化する。

### 7.3 Backpressure

| 箇所 | 制御 |
|---|---|
| API | 最大接続・最大body |
| Admission | queue depth、token budget |
| Tokenizer | worker queue |
| GPU | active sequence、KV Cache |
| Streaming | bounded output buffer |

上限超過時は429またはresource_exhaustedを返し、無制限にmemoryへ積まない。

---

## 8. 外部Componentとの契約

### 8.1 Lykuro Local LLM Platform

Platformが担当:

- tenant/RBAC
- Policy
- model routing
- Conversation Memory
- model approval
- usage aggregation
- request priority上限

Engineが担当:

- model instance検証
- local capacity
- inference
- internal usage計測
- cancel

EngineはPlatformから渡されたtenant情報を認証判断に使用しないが、resource isolationとmetricsのscopeとして利用する。

### 8.2 Model Manager

Model Managerのみが次を実行できる。

- load
- unload
- drain
- update準備
- model manifest照会

Gateway serviceはControl APIを利用できない。

### 8.3 Conversation Memory

- Engineは会話履歴を保存しない。
- Platformが必要contextを毎requestで送る。
- EngineはKV Cache handleを返せるが、永続conversation IDと同一視しない。
- KV Cache miss時はfull inputで再実行できるcontractとする。

---

## 9. Internal API

### 9.1 Transport

- ProductionはmTLS付きgRPCを推奨する。
- health、ready、metricsは管理network限定HTTPで提供できる。
- JSON debug bridgeはdevelopment buildだけに許可する。
- public portへbindしない。

### 9.2 Data API

| RPC | 用途 |
|---|---|
| Generate | 非streaming生成 |
| GenerateStream | streaming生成 |
| Cancel | request cancel |
| CountTokens | model固有token数 |
| GetCapabilities | instance能力 |

### 9.3 Control API

| RPC | 用途 |
|---|---|
| LoadModel | 署名済みartifactをload |
| UnloadModel | active modelをunload |
| Drain | 新規request受付停止 |
| Resume | 受付再開 |
| GetModelStatus | model state |
| GetCapacity | device、memory、queue |
| GetManifest | load済みmanifest metadata |

### 9.4 推論Request

~~~json
{
  "request_id": "req_01J_example",
  "trace_id": "trc_example",
  "tenant_scope": "tn_hash",
  "project_scope": "prj_hash",
  "model_instance_id": "mi_qwen_01",
  "input": {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "質問"}
    ]
  },
  "generation": {
    "max_output_tokens": 512,
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40,
    "seed": 1234,
    "stop": []
  },
  "scheduling": {
    "priority": 50,
    "deadline_unix_ms": 1786000000000
  },
  "cache": {
    "reuse_handle": null,
    "allow_prefix_cache": false
  }
}
~~~

### 9.5 入力制約

- request_id必須
- model_instance_id必須
- promptまたはmessagesのどちらか一方
- input byte上限
- message数上限
- role allowlist
- max_output_tokens上限
- temperature/top_p/top_k範囲
- stop sequence数・長さ上限
- deadline上限
- priorityはPlatform上限をEngineでclamp

### 9.6 Response

~~~json
{
  "request_id": "req_01J_example",
  "model_instance_id": "mi_qwen_01",
  "output_text": "回答",
  "finish_reason": "stop",
  "usage": {
    "input_tokens": 128,
    "output_tokens": 42,
    "total_tokens": 170
  },
  "timing": {
    "queue_ms": 4,
    "tokenize_ms": 2,
    "prefill_ms": 80,
    "decode_ms": 520,
    "total_ms": 606
  },
  "cache": {
    "reuse_handle": null
  }
}
~~~

### 9.7 Streaming Event

~~~text
response.started
response.output_text.delta
response.usage
response.completed
response.error
~~~

eventはrequest_id、monotonic sequence、timestampを持つ。

---

## 10. Model Artifact・Manifest

### 10.1 Artifact構成

~~~text
model-artifact/
├── manifest.json
├── manifest.sig
├── checksums.sha256
├── config/
│   ├── model.json
│   ├── tokenizer.json
│   └── chat_template.json
├── weights/
│   ├── model-00001-of-000xx.safetensors
│   └── model.safetensors.index.json
├── license/
│   └── review.json
└── evaluation/
    └── certified-profile.json
~~~

### 10.2 Manifest必須項目

~~~json
{
  "schema_version": "1",
  "artifact_id": "ma_qwen_01",
  "model_family": "qwen",
  "architecture": "approved_qwen_decoder_v1",
  "model_version": "example",
  "weight_format": "safetensors",
  "precision": "bf16",
  "vocab_size": 0,
  "hidden_size": 0,
  "num_layers": 0,
  "num_attention_heads": 0,
  "num_key_value_heads": 0,
  "head_dim": 0,
  "max_context_tokens": 0,
  "tokenizer_type": "approved_qwen_tokenizer_v1",
  "chat_template_id": "qwen_chat_v1",
  "files": [],
  "license_review_id": "lic_01",
  "certified_profiles": [],
  "created_at": "2026-08-07T00:00:00Z"
}
~~~

0は例示placeholderであり、production manifestでは正の検証済み値を必須とする。

### 10.3 検証

- schema version
- signature chain
- checksum
- file count/size
- path traversal
- duplicate tensor
- missing tensor
- dtype
- tensor shape
- architecture parameter
- tokenizer special token
- chat template ID
- license review
- certified hardware profile
- Engine ABI compatibility

### 10.4 禁止

- executable file
- arbitrary Python
- remote URL
- symlink
- absolute path
- archive外参照
- dynamic shared object
- unsigned plugin
- unknown config fieldの黙認

---

## 11. Model Architecture Plugin

### 11.1 方針

model固有差分をPluginへ隔離し、Engine coreへfamily固有条件分岐を増殖させない。

PluginはLykuro source tree内でbuild・署名された静的または承認済みmoduleとする。顧客が任意Pluginをuploadする機能は提供しない。

### 11.2 Interface

~~~cpp
class ModelArchitecture {
public:
    virtual ValidationResult Validate(const ModelManifest& manifest) const = 0;
    virtual MemoryEstimate EstimateMemory(const ModelManifest& manifest,
                                          const RuntimeProfile& profile) const = 0;
    virtual TokenizerSpec Tokenizer() const = 0;
    virtual PromptTemplateSpec PromptTemplate() const = 0;
    virtual Status LoadWeights(const ArtifactReader& artifact,
                               DeviceContext& device) = 0;
    virtual Status Prefill(const BatchInput& input,
                           KVCacheView& cache,
                           Tensor& logits) = 0;
    virtual Status Decode(const DecodeInput& input,
                          KVCacheView& cache,
                          Tensor& logits) = 0;
    virtual Capabilities Capabilities() const = 0;
    virtual ~ModelArchitecture() = default;
};
~~~

### 11.3 Registry

~~~text
architecture_id
plugin_version
engine_abi_min
engine_abi_max
supported_precision
supported_hardware
supported_capabilities
signature
status
~~~

### 11.4 追加Process

1. architecture proposal
2. config/weight mapping設計
3. reference output作成
4. unit/golden test
5. memory/performance test
6. safety/license review
7. certified profile発行
8. signed release

---

## 12. Model Loader

### 12.1 Load Flow

~~~mermaid
flowchart TD
    A["Load Request"] --> B["Manifest Verify"]
    B --> C["Plugin Resolve"]
    C --> D["Memory Estimate"]
    D --> E["Stage Weight"]
    E --> F["GPU Load"]
    F --> G["Smoke Inference"]
    G --> H["Ready"]
~~~

### 12.2 Memory事前計算

必要memory:

~~~text
weight_bytes
+ kv_cache_reservation
+ activation_workspace
+ tokenizer/runtime_overhead
+ communication_workspace
+ safety_reserve
~~~

available VRAMより大きい場合はload前に拒否する。

### 12.3 Safetensors Loader

- header size上限
- JSON schema検証
- tensor offset境界
- overlap禁止
- file size一致
- dtype allowlist
- alignment
- expected tensor name
- expected shape
- shard index一致
- memory-map/read-only

### 12.4 Staging

- active modelとstaging modelの同時memoryを考慮する。
- 同時load不可の場合はdrain/unloadを要求する。
- staging失敗時はactive modelを維持する。
- smoke test成功前にreadyを返さない。

### 12.5 Unload

1. drain
2. active request完了またはdeadline cancel
3. KV Cache release
4. GPU synchronization
5. weight release
6. allocator検査
7. state更新

---

## 13. Tokenizer・Prompt Template

### 13.1 Tokenizer

- certified tokenizerだけを使用する。
- tokenizer file digestをmanifestで固定する。
- Unicode normalization policyをversion化する。
- special token IDを検証する。
- encode/decode round-trip testを持つ。
- invalid UTF-8処理を定義する。
- token countとactual encodeの差異をなくす。

### 13.2 Prompt Template

normalized messageをarchitecture固有templateへ変換する。

対象role:

- system
- developer
- user
- assistant
- tool

MVPで未対応roleを黙って連結しない。

### 13.3 Template Security

- templateは承認済みIDから選択する。
- arbitrary template codeを実行しない。
- input本文をtemplate命令として解釈しない。
- special token injectionをtestする。
- tool schema sizeを制限する。

### 13.4 Token Budget

~~~text
input_tokens
+ max_output_tokens
<= certified_context_limit
~~~

超過時はEngineがtruncateせず、context_length_exceededを返す。履歴要約・削除はPlatformの責務とする。

---

## 14. 推論Pipeline

### 14.1 Standard Flow

1. API schema validation
2. service authentication
3. model instance/state検査
4. deadline/cancel検査
5. prompt build
6. tokenize
7. admission
8. KV Cache allocate
9. prefill
10. decode loop
11. sampling
12. stop/limit判定
13. detokenize
14. streaming
15. usage/timing確定
16. cache/resource release

### 14.2 Prefill

- variable length inputを扱う。
- padding/maskをplugin仕様で処理する。
- prompt token上限を検査する。
- prefill timeoutを持つ。
- large promptがdecode requestを永久にblockしない。

### 14.3 Decode

- active sequenceをiterationごとに選択する。
- cancel/deadlineをiteration間で確認する。
- finished sequenceをbatchから除外する。
- output token上限を強制する。
- invalid logits/NaN/Infを検知する。

### 14.4 Finish Reason

| Reason | 条件 |
|---|---|
| stop | EOSまたはstop sequence |
| length | max_output_tokens |
| cancelled | client/Platform cancel |
| deadline | deadline超過 |
| error | Engine failure |
| content_filter | 将来のlocal output filter |

---

## 15. Scheduler・Admission

### 15.1 Request state

~~~mermaid
stateDiagram-v2
    [*] --> received
    received --> queued
    queued --> admitted
    queued --> rejected
    admitted --> prefill
    prefill --> decoding
    decoding --> completed
    queued --> cancelled
    admitted --> cancelled
    prefill --> failed
    decoding --> failed
~~~

### 15.2 Admission条件

- Engine ready
- Model Instance ready
- request schema valid
- context limit
- queue capacity
- sequence capacity
- estimated KV memory
- deadline feasible
- priority allowed
- tenant/project resource scope

### 15.3 Priority

0〜100の内部priorityを定義し、Platformが決めた上限をclampする。

推奨class:

| Class | Range |
|---|---:|
| batch/background | 0〜29 |
| normal | 30〜69 |
| interactive | 70〜89 |
| critical approved | 90〜100 |

### 15.4 Fairness

- weighted fair queueを基本とする。
- 一つのtenant/projectが全slotを占有しない。
- agingで長期待ちを軽減する。
- criticalの常時占有を上限で防止する。
- scheduling decisionをmetricへ記録する。

### 15.5 Continuous Batching

MVP要件:

- prefill batchとdecode batchを区別する。
- batch token上限を設定する。
- max sequencesを設定する。
- batch形成待ち時間に上限を持つ。
- deadlineが近いrequestを考慮する。
- cancelled sequenceを速やかに除外する。

### 15.6 Overload

- queue full: resource_exhausted
- KV shortage: capacity_exhausted
- deadline infeasible: deadline_rejected
- draining: engine_draining

無制限retryを誘発しないためretryable flagと推奨backoffを返す。

---

## 16. KV Cache・Memory管理

### 16.1 原則

- request/sequenceごとにownershipを持つ。
- tenant/project scopeをcache metadataへ持つ。
- 別scopeへのcache reuseを禁止する。
- reference countとgenerationを検証する。
- releaseをidempotentにする。
- reassign前にmetadataを消去する。

### 16.2 MVP

- contiguousまたはsequence単位allocation
- fixed maximum context
- active request終了時release
- cache reuseは既定OFF
- OOM前のadmission拒否

### 16.3 Phase 2

- block allocator
- paged KV storage
- prefix cache
- LRU/priority eviction
- cache handle
- fragmentation metric

### 16.4 KV Memory概算

一般的な概算:

~~~text
kv_bytes
= 2
* num_layers
* num_key_value_heads
* head_dim
* bytes_per_element
* total_cached_tokens
~~~

architecture固有差分はPluginのEstimateMemoryで上書きする。

### 16.5 GPU Allocator

- weight、KV、workspaceのpoolを分離する。
- high-water markを記録する。
- fragmentationを監視する。
- safety reserveを使い切らない。
- allocator errorをfatal/isolatableに分類する。
- unload後のleak testを行う。

### 16.6 Memory Security

- freed blockのmetadataをclearする。
- debug dumpにtensor本文を含めない。
- core dumpをproductionで禁止または暗号化・制限する。
- swap/page fileへのsecret leakageを評価する。
- multi-tenant cache reuseを禁止する。

---

## 17. Hardware Backend

### 17.1 Backend Interface

~~~cpp
class HardwareBackend {
public:
    virtual DeviceInventory Discover() = 0;
    virtual Compatibility CheckCompatibility(const CertifiedProfile&) = 0;
    virtual MemoryStatus Memory() = 0;
    virtual Status Allocate(const AllocationRequest&, Allocation&) = 0;
    virtual Status Execute(const ExecutionPlan&) = 0;
    virtual Status Synchronize() = 0;
    virtual HealthStatus Health() = 0;
    virtual ~HardwareBackend() = default;
};
~~~

### 17.2 CUDA MVP

検査項目:

- driver version
- runtime version
- compute capability
- device count
- total/free VRAM
- ECC status
- temperature
- power/throttle
- exclusive/process mode
- certified profile一致

### 17.3 Device選択

MVPは明示device IDを必須とする。自動的に全GPUを使用しない。

~~~yaml
hardware:
  backend: cuda
  device_ids: [0]
  vram_reserve_mb: 2048
~~~

### 17.4 OOM処理

1. 新規admission停止
2. failing requestをerror
3. GPU状態確認
4. request resource解放
5. allocator health検査
6. safeならresume
7. unsafeならdegraded/restart要求

OOM後にsilent continuationしない。

### 17.5 Future Multi-GPU

- Tensor Parallel
- Pipeline Parallel
- Expert Parallel
- NCCL health
- rank lifecycle
- collective timeout
- partial rank failure

MVP contractへ未実装flagを明記する。

---

## 18. Generation・Sampling

### 18.1 MVP parameter

- max_output_tokens
- temperature
- top_p
- top_k
- seed
- stop
- repetition penaltyは承認後

### 18.2 Validation

| Parameter | Rule |
|---|---|
| max_output_tokens | 1〜certified上限 |
| temperature | 0以上、上限設定 |
| top_p | 0超〜1 |
| top_k | 0以上、上限設定 |
| seed | signed integer範囲 |
| stop | 個数・byte/token上限 |

### 18.3 Deterministic Mode

- temperature 0相当のgreedy
- seed固定
- certified hardware/profile
- deterministic kernel利用可否をmetadataへ記録
- version変更時の完全一致を保証しない場合は明示する

### 18.4 Sampler

logits処理順をversion化する。

~~~text
validate logits
-> repetition/presence processing
-> temperature
-> top-k
-> top-p
-> sample
-> stop/eos check
~~~

### 18.5 Logprobs

MVPでは任意。対応する場合:

- top-N上限
- memory/latency影響
- invalid value検査
- streaming schema

---

## 19. Streaming・Cancel

### 19.1 Streaming

- monotonic event sequence
- UTF-8境界を壊さない
- partial token buffer
- bounded channel
- slow consumer検知
- final usage event
- finish reason

### 19.2 Cancel

cancel source:

- client disconnect
- explicit Cancel RPC
- deadline
- Platform policy
- engine drain

cancelはidempotentとし、queued、prefill、decodeの各stageで処理する。

### 19.3 Slow Consumer

- output buffer上限
- timeout
- GPU generationを無制限継続しない
- cancel後にsequence/KVを解放
- errorにretryable=falseを設定

---

## 20. State Model

### 20.1 Engine state

~~~mermaid
stateDiagram-v2
    [*] --> starting
    starting --> idle
    idle --> loading
    loading --> ready
    loading --> degraded
    ready --> draining
    draining --> idle
    ready --> degraded
    degraded --> recovering
    recovering --> ready
    recovering --> stopped
~~~

### 20.2 Model Instance state

| State | 内容 |
|---|---|
| registered | metadataのみ |
| validating | artifact検証 |
| staging | CPU/GPU load |
| testing | smoke inference |
| ready | 推論可能 |
| draining | 新規受付停止 |
| unloading | resource解放 |
| failed | load/runtime失敗 |
| unloaded | 非常駐 |

### 20.3 Readiness

ready条件:

- service auth準備
- active model ready
- plugin/certified profile一致
- GPU healthy
- allocator healthy
- scheduler accepting
- config署名valid

---

## 21. Error設計

### 21.1 Error schema

~~~json
{
  "request_id": "req_example",
  "code": "context_length_exceeded",
  "message": "Input and requested output exceed the certified context limit.",
  "retryable": false,
  "component": "admission",
  "details": {
    "limit_tokens": 32768
  }
}
~~~

本文、prompt、token原文、path、credentialをerrorへ含めない。

### 21.2 Error code

| Code | Retry | 条件 |
|---|---:|---|
| invalid_request | × | schema不正 |
| authentication_failed | 条件 | service identity不正 |
| model_not_loaded | ○ | model非ready |
| unsupported_model | × | architecture非対応 |
| artifact_verification_failed | × | signature/digest |
| context_length_exceeded | × | token上限 |
| resource_exhausted | ○ | queue/sequence上限 |
| capacity_exhausted | ○ | KV/VRAM不足 |
| deadline_rejected | × | 実行前に期限不可能 |
| deadline_exceeded | 条件 | 実行中期限 |
| request_cancelled | × | cancel |
| engine_draining | ○ | maintenance |
| gpu_oom | 条件 | GPU OOM |
| gpu_unhealthy | ○ | device異常 |
| inference_failed | 条件 | forward/sampling |
| stream_consumer_slow | × | client遅延 |
| internal_error | 条件 | 未分類 |

---

## 22. Security

### 22.1 Network

- egress deny
- internal ingress allowlist
- mTLS
- service identity
- port最小化
- management/data plane分離
- debug endpoint production無効

### 22.2 Process

- non-root
- read-only root filesystem
- no-new-privileges
- capability drop
- seccomp/AppArmor等
- model weight read-only mount
- tmpfs size上限
- core dump制限

### 22.3 Model Supply Chain

- signed artifact
- signed Engine image
- SBOM
- provenance
- vulnerability scan
- license review
- rollback artifact保持
- trusted key rotation

### 22.4 Content

- request/response log禁止
- metric label禁止
- trace attribute禁止
- diagnose bundle禁止
- crash dump禁止
- sampling debug dump production禁止

### 22.5 DoS

- input byte/token上限
- max output
- max concurrent
- queue上限
- deadline
- stop sequence上限
- tokenizer complexity test
- malformed artifact検査
- slow consumer cancel

### 22.6 Plugin Security

- Lykuro署名済みPluginのみ
- arbitrary upload禁止
- dynamic code download禁止
- ABI/version検証
- Plugin capability allowlist
- Plugin load前のhash検証

---

## 23. Data・Retention

Native Engineは永続的なPrompt/Response保存を行わない。

| Data | 保存 |
|---|---|
| Prompt/Response | 0日 |
| Token IDs | request終了時削除 |
| KV Cache | request/session policy、標準5〜30分以内 |
| Weight | model lifecycle期間 |
| Usage metadata | local Platformへ返却 |
| Metrics | 本文なし、運用policy |
| Error log | 本文なし、標準30日 |
| Crash dump | production既定OFF |

Conversation保存30日はPlatform側の責務であり、Engineは保持しない。

---

## 24. Observability

### 24.1 Metrics

| Category | Metrics |
|---|---|
| Request | received、admitted、rejected、completed、failed |
| Queue | depth、wait、priority、age |
| Tokenizer | latency、input tokens、error |
| Prefill | latency、tokens/sec、batch size |
| Decode | latency、tokens/sec、active sequences |
| Streaming | buffer、slow consumer、cancel |
| KV | allocated、free、fragmentation、eviction |
| GPU | VRAM、utilization、temperature、power、ECC |
| Model | load time、state、version |
| Engine | state、uptime、restart、build |

### 24.2 Metric label制限

使用可能:

- engine_id
- model_instance_id
- model_family
- plugin_version
- device_id
- result_code
- priority_class

禁止:

- prompt/response
- request content
- API key
- user ID原文
- conversation title
- unbounded request ID label

### 24.3 Structured Log

~~~json
{
  "timestamp": "2026-08-07T00:00:00Z",
  "level": "INFO",
  "component": "scheduler",
  "event": "request_completed",
  "request_id": "req_example",
  "model_instance_id": "mi_qwen_01",
  "result_code": "success",
  "input_tokens": 128,
  "output_tokens": 42,
  "latency_ms": 606
}
~~~

### 24.4 Diagnose Bundle

- build/version
- configのsecret除去版
- model manifest metadata
- device/driver
- recent error code
- metric snapshot
- allocator summary

本文、weight、token、KV、credentialを含めない。

---

## 25. Performance・Capacity

### 25.1 Certified Profile

model、GPU、driver、Engine、precision、contextごとにprofileを作る。

~~~yaml
profile_id: cp_qwen_gpu_example
engine_version: 1.0.0
architecture_plugin: approved_qwen_decoder_v1
model_artifact_id: ma_qwen_01
hardware:
  gpu_model: certified-device
  gpu_count: 1
precision: bf16
context_tokens: 8192
max_sequences: 8
targets:
  gateway_excluded: true
  ttft_p95_ms: contract-value
  output_tokens_per_second: contract-value
  scheduler_p95_ms: contract-value
~~~

数値は実GPU benchmark後に固定し、文書内で推測しない。

### 25.2 Capacity API

返却項目:

- total/free VRAM
- reserved VRAM
- active sequences
- max sequences
- queued requests
- max queue
- KV used/free
- loaded model
- context limit
- estimated admission tokens
- health/throttle

### 25.3 Benchmark

- cold load
- warm load
- single request
- concurrent request
- short/long input
- short/long output
- streaming
- cancel
- queue saturation
- 1時間/24時間soak
- unload/reload cycle

### 25.4 Correctness First

performance optimization前後でgolden correctness、distribution、stop、seed、usageを再検証する。

---

## 26. Configuration

### 26.1 Engine config

~~~yaml
engine:
  id: nie-node-01
  listen_address: 127.0.0.1
  grpc_port: 19443
  state_dir: /var/lib/lykuro/native-engine
  log_level: info

security:
  mtls_required: true
  server_cert_ref: file:///run/secrets/server.crt
  server_key_ref: file:///run/secrets/server.key
  client_ca_ref: file:///run/secrets/client-ca.crt
  egress_disabled: true

model:
  artifact_path: /models/current
  architecture_allowlist:
    - approved_qwen_decoder_v1

hardware:
  backend: cuda
  device_ids: [0]
  vram_reserve_mb: 2048

scheduler:
  max_queue: 256
  max_sequences: 8
  max_batch_tokens: 8192
  max_wait_ms: 10

generation:
  max_input_tokens: 28672
  max_output_tokens: 4096

observability:
  metrics_enabled: true
  metrics_address: 127.0.0.1
  metrics_port: 19090
  content_logging: false
~~~

### 26.2 Secret

secret値を通常configへ記載しない。file referenceまたはsecret manager mountを使用する。

### 26.3 Validation

- unknown key拒否または明示warning
- range
- path
- permission
- certificate
- device
- certified profile
- incompatible combination

---

## 27. Deployment・Package

### 27.1 Production artifact

- signed OCI imageまたはsigned native package
- checksum
- SBOM
- provenance
- vulnerability report
- release note
- compatibility matrix
- rollback artifact

### 27.2 含めないもの

- Ollama
- llama.cpp
- vLLM
- TGI
- third-party model server
- model weight本体
- GPU driver
- Node.js/npm
- Python runtime

承認済み低レベルshared libraryを含める場合はSBOMとlicense noticeへ記載する。

### 27.3 Filesystem

~~~text
/opt/lykuro/native-engine/       executable
/etc/lykuro/native-engine/       non-secret config
/var/lib/lykuro/native-engine/   state
/var/log/lykuro/native-engine/   content-free logs
/models/                         read-only approved model mount
/run/secrets/                    secret mount
~~~

### 27.4 Kubernetes

- GPU request/limit
- node selector
- taint/toleration
- readiness/liveness
- termination grace
- PodDisruptionBudget
- NetworkPolicy
- read-only model volume
- secret volume
- no public Service

### 27.5 Host prerequisite

- certified OS/kernel
- certified GPU/driver
- container runtimeまたはpackage runtime
- local disk/VRAM
- time sync
- internal DNS
- certificate

installerはGPU driverを自動更新しない。

---

## 28. Update・Rollback

### 28.1 Version

- Engine semantic version
- API version
- Engine ABI
- Plugin version
- Manifest schema
- Certified Profile version

### 28.2 Compatibility

| 組合せ | 方針 |
|---|---|
| Platform ↔ Engine API | N/N-1 |
| Engine ↔ Plugin ABI | manifestでrange |
| Engine ↔ Model Artifact | certified matrix |
| Engine ↔ Driver | certified matrix |

### 28.3 Update Flow

1. new artifact署名検証
2. compatibility確認
3. capacity/drain
4. current state snapshot
5. new version起動
6. model load/smoke
7. readiness
8. traffic切替
9. observation
10. old version停止

### 28.4 Auto Rollback

- process crash
- readiness timeout
- model load failure
- smoke inference failure
- GPU unhealthy
- error rate threshold
- severe performance regression

model weight破壊・変更をrollback時に行わない。

---

## 29. 障害・復旧

| 障害 | 検知 | 処理 |
|---|---|---|
| model artifact不正 | load verify | load拒否 |
| GPU OOM | backend error | admission停止、recover/degraded |
| GPU lost | health/driver | failed、Pod/process restart |
| tokenizer error | request | invalid_request/inference_failed |
| NaN/Inf | logits check | request失敗、model degraded評価 |
| queue saturation | scheduler | resource_exhausted |
| slow consumer | stream buffer | cancel |
| Platform切断 | stream/auth | request cancelまたはdeadline |
| Metrics停止 | exporter | inference継続、alert |
| Plugin crash | process | Engine restart、rollback |
| disk full | state/log | log抑制、alert、safe stop |

### 29.1 Startup Recovery

- incomplete load stateを破棄
- signed config再検証
- GPU health
- model artifact再検証またはcached verification policy
- smoke inference
- ready

### 29.2 Safe Shutdown

1. draining
2. new request拒否
3. active request期限待ち
4. remaining cancel
5. stream final error
6. GPU sync
7. KV/weight release
8. metrics flush
9. stop

---

## 30. Test仕様

### 30.1 Unit

- manifest parser
- signature/digest
- Safetensors bounds/shape
- plugin registry
- tokenizer encode/decode
- prompt template
- token budget
- sampler
- stop sequence
- scheduler fairness
- deadline/cancel
- memory estimate
- error mapping
- config validation

### 30.2 Golden Correctness

- fixed prompt/token
- expected token IDs
- expected first-token logits tolerance
- greedy output
- seeded sampling distribution
- EOS/stop
- long context boundary
- batch vs single consistency

test oracleはdevelopment環境で利用できるが、第三者Runtimeをproduction packageへ含めない。

### 30.3 Loader Negative

- invalid signature
- wrong digest
- truncated file
- offset overlap
- path traversal
- duplicate tensor
- missing tensor
- wrong shape/dtype
- unknown architecture
- incompatible Engine ABI
- insufficient VRAM

### 30.4 Scheduler

- FIFO within same weight
- weighted fairness
- priority clamp
- aging
- queue full
- cancel queued
- cancel decode
- deadline before/after admission
- mixed prompt length
- slow stream consumer

### 30.5 Memory

- allocation/release
- double free防止
- cache scope isolation
- fragmentation
- OOM recovery
- unload/reload 100 cycle以上
- 24時間soak
- no growth after steady state

### 30.6 Streaming

- event order
- event sequence
- UTF-8
- client disconnect
- final usage
- error途中
- backpressure

### 30.7 Security

- unauthenticated API
- wrong client certificate
- Control/Data scope violation
- egress attempt
- secret/log leakage
- prompt in metrics/trace
- malicious artifact
- malformed tokenizer
- model path traversal
- unsigned plugin
- oversized request
- fuzz API/parser

### 30.8 Performance

- certified profileごとのbenchmark
- cold/warm
- TTFT
- input/output TPS
- scheduler overhead
- queue saturation
- GPU utilization
- VRAM
- batch scaling
- performance regression threshold

### 30.9 Deployment

- container non-root
- read-only filesystem
- no egress
- readiness/liveness
- graceful drain
- rolling update
- rollback
- SBOMに禁止Runtimeがない

---

## 31. 受入基準

| ID | 項目 | 合格条件 |
|---|---|---|
| AT-01 | 独自性 | 禁止された第三者推論Runtimeのsource/binary/imageが存在しない |
| AT-02 | Model限定 | allowlist外architectureをloadできない |
| AT-03 | Artifact | signature、digest、shape不正を拒否 |
| AT-04 | Correctness | certified golden testに合格 |
| AT-05 | API | Generate、Stream、Cancel、Control APIがcontractどおり |
| AT-06 | Isolation | Engine APIを認証済みPlatform/Managerだけが利用可能 |
| AT-07 | 本文非保存 | prompt/responseがlog、metric、trace、diskに残らない |
| AT-08 | Scheduler | bounded queue、priority、fairness、deadlineが動作 |
| AT-09 | KV Cache | ownership、release、scope isolationが動作 |
| AT-10 | OOM | crash/漏えいなく拒否またはdegradedへ移行 |
| AT-11 | Streaming | event order、cancel、slow consumerが正しく動作 |
| AT-12 | Capacity | VRAM、queue、sequence、model状態を取得可能 |
| AT-13 | Performance | certified profileの目標を達成 |
| AT-14 | Soak | 長時間testでmemory/resource leakなし |
| AT-15 | Update | signed updateとrollbackが動作 |
| AT-16 | Security | mTLS、non-root、egress deny、artifact fuzzに合格 |
| AT-17 | Packaging | SBOM、provenance、signature、license noticeが存在 |
| AT-18 | Integration | Local LLM Platformからroute/load/inference可能 |

---

## 32. 実装Phase

### Phase 0: Feasibility・Contract

- target Qwen architecture確定
- certified model artifact作成
- API/protobuf
- compiler/CUDA/build
- dependency/license
- reference oracle
- GPU benchmark環境

終了条件: 一つのforward passを正しく検証できる設計と環境がある。

### Phase 1: Correctness PoC

- manifest/loader
- tokenizer/template
- Qwen architecture
- single request prefill/decode
- greedy generation
- basic API
- golden test

終了条件: fixed inputでreference tolerance内の結果。

### Phase 2: Engine MVP

- streaming
- sampling
- scheduler
- basic continuous batch
- KV Cache
- cancel/deadline
- metrics
- Model Manager統合

終了条件: MVP受入基準に合格。

### Phase 3: Product Hardening

- mTLS
- signed artifact/plugin
- OOM recovery
- soak/fuzz
- signed package
- update/rollback
- certified profile

終了条件: Production security/reliability review合格。

### Phase 4: Performance

- block KV allocator
- prefix cache
- optimized kernels
- quantization
- advanced batching

終了条件: certified performance target達成。

### Phase 5: Enterprise Expansion

- multiple model
- multi-GPU
- AMD/CPU
- MoE
- embeddings
- vision

各機能を別feature flagとcertified profileで提供する。

---

## 33. Definition of Done

- repository構造、build、dependency policyが確定している。
- 第三者推論Runtimeをsource、binary、image、transitive dependencyに含めていない。
- API/protobufとversioningが文書化されている。
- Model Artifact/Manifest schemaがversion管理されている。
- Architecture Plugin interfaceとQwen pluginが実装されている。
- loader/tokenizer/prefill/decode/sampling/streamingが動作する。
- Scheduler、Batch、KV Cache、cancel、deadlineが動作する。
- mTLS、signed artifact、egress deny、non-rootが設定されている。
- prompt/responseがlog、metric、trace、diskへ残らない。
- unit、golden、integration、security、performance、soak testが成功している。
- certified hardware/model profileが発行されている。
- SBOM、provenance、signature、license noticeがある。
- deployment、monitoring、update、rollback、recovery手順がある。
- 未実装、性能未達、未検証hardware/modelが明記されている。

---

## 34. 要決定事項

| ID | 論点 | 推奨初期値 |
|---|---|---|
| D-01 | Engine core言語 | C++20 |
| D-02 | GPU backend | NVIDIA CUDA |
| D-03 | Model family | Qwen Dense |
| D-04 | Weight format | Safetensors |
| D-05 | Precision | BF16/FP16 |
| D-06 | GPU count | 単一GPU |
| D-07 | Transport | mTLS gRPC |
| D-08 | Model per process | 1 |
| D-09 | KV reuse | MVP既定OFF |
| D-10 | Public egress | 禁止 |
| D-11 | Prompt/Response保存 | 0日 |
| D-12 | Production Python | 不使用 |
| D-13 | Multi-GPU | Phase 5 |
| D-14 | Quantization | Phase 4 |

実測・security reviewなしに推奨初期値をproduction確定しない。

---

## 35. Claude Codeの最終報告形式

~~~markdown
## Repository調査結果

## 実装対象Phase

## Architecture・Thread・Memory設計

## API・Protobuf

## Model Artifact・Manifest

## Architecture Plugin

## Loader・Tokenizer

## Scheduler・Batch・KV Cache

## CUDA Backend・Generation

## Security

## Observability・Deployment

## 実行したTestと結果

## Certified Profile・Performance

## 未実装・未検証・既知問題
~~~

GPU、driver、model artifact、署名鍵、Kubernetes等がなく検証できない場合は、未検証理由、必要環境、再現可能な手順を報告する。test未実行を成功と記載しない。

---

## 改訂履歴

| 版 | 日付 | 内容 |
|---|---|---|
| v1.0 | 2026-08-07 | Native Engineの独立project、API、model plugin、推論、scheduler、memory、GPU、security、testを完全定義 |
