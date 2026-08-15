# Lykuro Native Inference Engine macOS Metal Backend 追加仕様書

| 項目 | 内容 |
|---|---|
| 文書番号 | LYK-NIE-ADD-METAL-001 |
| 版 | v1.0 Claude Code Edition |
| 制定日 | 2026-08-08 |
| 作成 | 株式会社eビジネスソリューション / Lykuro.ai |
| Project | lykuro-native-inference-engine |
| 対象 | Apple Silicon、macOS、Metal Backend、Unified Memory |
| 機密区分 | 社内・提案先限定 |
| 実装担当 | Claude CodeおよびNative Engine開発担当者 |
| 関連文書 | LYK-NIE-SD-001、LYK-NLP-SD-001 v2.0、LYK-PLG-BD-001 |
| 文書状態 | macOS GPU対応追加仕様 |

---

## 0. Claude Codeへの実装指示

本書は、実装済みLykuro Native Inference EngineへApple Silicon向けMetal Backendを追加するための差分仕様である。

Claude CodeはNative Engineを新規作成してはならない。既存Engine Core、Data API、Control API、Model Architecture Plugin、Scheduler、Batch、KV Cache、Model Artifact、Security、Observabilityを再利用し、Hardware Backend境界へmacOS/Metal実装を追加する。

Private LLM GatewayおよびLykuro Native LLM Platformを再実装しない。Gateway外部APIにmacOS固有仕様を公開しない。

### 0.1 文書優先関係

| 対象 | 正本 |
|---|---|
| Engine共通Core、Data API、Control API | LYK-NIE-SD-001と既存実装 |
| Gateway、認証、Virtual Key、Policy、Audit | LYK-PLG-BD-001と既存Gateway |
| Model Manager、Routing、Conversation、Pool | LYK-NLP-SD-001 v2.0 |
| macOS、Metal、Unified Memory | 本書 |
| 実際の対応能力 | As-Built調査、test、Certified Profile |

本書はLYK-NIE-SD-001のCUDA限定記述をmacOS Metal Backendについて拡張する。共通APIまたはSecurityを弱める記述として解釈しない。

### 0.2 実装開始前の必須調査

1. Native Engine repository、branch、build system、language、compiler
2. Hardware Backend InterfaceとCUDA Backendの現行実装
3. Tensor、Allocator、Scheduler、KV Cacheのmemory ownership
4. Model Architecture Pluginと対象modelの実装状況
5. Data API、Control API、error code、version
6. Apple Silicon検証機のchip、Unified Memory、macOS、Xcode/Metal SDK
7. MTLDevice、Metal feature、MPSGraph対応能力
8. MLX C++を利用する場合のversion、license、link方式、SBOM
9. Production package、code signing、notarization、update方式
10. CUDA golden resultとMac CPU/Metal resultの比較方法
11. 実機CI runner、benchmark、soak test環境
12. 本書の論理componentと既存sourceの対応表

調査結果と変更範囲を報告するまでは、Engine Core、Data API、Control API、Model Artifact schemaへ破壊的変更を行わない。

### 0.3 絶対条件

- 対象はApple Silicon arm64とする。
- Intel Macを初期対象にしない。
- Mac GPU BackendはMetalを使用する。CUDA emulationを行わない。
- Production EngineはmacOS host-native processとして動作させる。
- Docker Desktop、OrbStackその他Linux VM内のEngineへMac GPUアクセスを前提にしない。
- Ollama、llama.cpp、vLLM、TGI、mlx-lm等の推論serverを内蔵・再配布しない。
- MLXを使用する場合は承認済みC/C++計算dependencyまたはtest oracleに限定する。
- Production packageへPython、pip、Node.js、npm、nvm、Homebrewを必須dependencyとして含めない。
- 顧客hostへXcodeを要求しない。Metal libraryはrelease buildで事前compileする。
- 未承認modelをdownload、convert、loadしない。
- Runtime中のpublic model registry accessとdynamic dependency downloadを禁止する。
- Prompt、Response、Token、KV CacheをLykuro Control Planeへ送信しない。
- PromptとResponseを通常log、metric、trace、crash dumpへ含めない。
- Unified Memory全量を使用可能とみなさない。
- OS memory pressureまたはswap発生を通常運転として許容しない。
- memory不足はload/admission前に検知してFail Closedする。
- Engine APIはPlatform/Model Managerからのみ利用する。
- Gateway service identityへEngine Control API権限を付与しない。
- 実行していないtestを成功と報告しない。

### 0.4 実装分類

| 区分 | 内容 |
|---|---|
| Reuse | Engine Core、API、Scheduler、Model Plugin、Artifact、Security |
| Extend | Hardware Backend Interface、Allocator、Capacity、Config、Metrics |
| New | Metal Backend、Metal shader library、macOS package、launchd integration |
| Verify | Model correctness、memory、thermal、performance、notarization |
| Out of Scope | Gateway再開発、Platform再開発、第三者推論server内蔵 |

---

## 1. 目的

Apple Silicon Macの内蔵GPUとUnified Memoryを利用し、Lykuro Native Inference EngineをmacOS上で安全かつ再現可能に実行できるようにする。

目的:

- NVIDIA GPUがないMac環境へNative Engineを提供する。
- CUDA Backendと同一のEngine Data API/Control APIを維持する。
- Apple SiliconのUnified Memoryを安全に管理する。
- Metal/MPSGraphと必要なLykuro Metal Kernelを利用する。
- Mac mini、Mac Studio、MacBook Proを推論nodeとしてPlatformへ登録する。
- model、hardware、OS、EngineのCertified Profileを発行する。
- Ollama等を経由せずLykuro Native EngineとしてGPU推論する。

### 1.1 提供形態

顧客向けpackage:

~~~text
lykuro-native-engine-macos-arm64-<version>.pkg
lykuro-native-engine-macos-arm64-<version>.tar.gz
~~~

EngineはmacOS host上でlaunchd serviceまたは明示的user serviceとして動作する。Gateway/Platformがcontainer内にある場合は、host-only networkまたはmTLS付き顧客内部network経由で接続する。

---

## 2. 対象範囲

### 2.1 MVP対象

| 項目 | MVP |
|---|---|
| CPU architecture | arm64 |
| Hardware | Apple Silicon内蔵GPU |
| GPU API | Metal |
| Compute | MPSGraph、Metal Performance Shaders、Metal Kernel |
| Model family | 既存Engineで認証済みのQwen Dense系から1種類 |
| Weight | 既存Engine対応Safetensors |
| Precision | FP16を初期基準 |
| GPU count | system default device 1台 |
| API | 既存Generate、Stream、Cancel、CountTokens、Capacity |
| Scheduler | 既存bounded queue、deadline、cancel |
| Batch | correctness確認後にbasic continuous batching |
| KV Cache | Unified Memory上のbounded cache |
| Package | signed/notarized macOS arm64 package |
| Service | host-native launchd |

### 2.2 次段階

- INT8/INT4量子化
- block-based KV Cache
- prefix cache
- multiple model instances
- advanced continuous batching
- custom fused Metal Kernel
- additional Qwen architecture
- embeddings
- Vision
- Neural Engine利用の個別検証
- 複数MacのRequest Sharding

### 2.3 対象外

- Intel Mac
- external GPU
- iPhone、iPad、Apple TV
- macOS container内からの直接Metal GPU利用
- CUDA sourceの自動変換だけでのProduction提供
- 一つのmodelを複数MacのUnified Memoryへ自動分割
- Internet越しTensor Parallelism
- arbitrary Python model code
- arbitrary Hugging Face remote code
- mlx-lm server、Ollama、llama.cppの内蔵
- Mac App Store配布

---

## 3. 用語

| 用語 | 定義 |
|---|---|
| Metal Backend | Native EngineのApple GPU向けHardware Backend |
| MTLDevice | Metalが公開するGPU device interface |
| Unified Memory | CPUとGPUが共有するApple Siliconのmemory構成 |
| Working Set | Performanceを損なわずGPU resourceへ使用可能な概算memory |
| MPSGraph | Apple platform向けcompute graph framework |
| Metal Kernel | Metal Shading Languageで実装するGPU compute kernel |
| metallib | compile済みMetal shader library |
| MLX C++ | Apple Silicon向けarray frameworkのC++ API |
| Mac Certified Profile | Engine、model、chip、memory、OS等の検証済み組合せ |
| Host Native | macOS processとして直接実行する方式 |

---

## 4. 対応Platform

### 4.1 対応条件

MVPは次をすべて満たす環境だけを対象とする。

- Apple Silicon arm64
- Metal device取得成功
- hasUnifiedMemoryがtrue
- 必要なMetal featureとoperatorを利用可能
- model load前のworking setとdisk容量が基準以上
- certified macOS/Engine/model組合せ
- Developer ID署名を検証可能

macOS最低version、chip family、Metal featureは実機test後にCompatibility Matrixへ固定する。本書内で未測定のversionを推測確定しない。

### 4.2 初期検証機

| Profile候補 | 用途 |
|---|---|
| Mac mini M4 / 64GB Unified Memory | 主開発・correctness・capacity・soak |
| 追加Apple Silicon 16〜32GB | memory不足、small model、回帰 |
| CUDA reference machine | golden correctness比較 |

Mac mini M4 64GBでも64GB全量をmodelへ割り当てない。OS、Platform、Engine Core、activation、KV Cache、stream buffer、safety marginを控除する。

---

## 5. 全体Architecture

~~~mermaid
flowchart TD
    A["Lykuro Native LLM Platform"] --> B["Engine Data API"]
    C["Lykuro Model Manager"] --> D["Engine Control API"]
    B --> E["Native Engine Common Core"]
    D --> E
    E --> F["Hardware Backend Interface"]
    F --> G["CUDA Backend"]
    F --> H["Metal Backend"]
    H --> I["MPSGraph / MPS / Metal Kernels"]
    I --> J["Apple Silicon GPU / Unified Memory"]
~~~

### 5.1 不変部分

macOS対応で変更しないもの:

- Public Gateway API
- Engine Data API semantics
- Engine Control API authorization
- request/response/stream event
- Model Manager ownership
- Scheduler state model
- Model Architecture Plugin contract
- signed Model Artifact
- Conversation Memory ownership
- Audit/Usage metadata contract

### 5.2 追加部分

- Metal device discovery
- Metal capability report
- Metal tensor/buffer implementation
- Metal allocator
- MPSGraph executable/cache
- Metal Kernel pipeline
- Unified Memory admission
- Metal metrics/error
- macOS package/service/update

---

## 6. Process・Thread Model

### 6.1 Process

~~~text
lykuro-native-engine
├── Internal API Thread
├── Admission / Scheduler Thread
├── Tokenizer Worker Pool
├── Metal Command Submission Thread
├── Stream Writer Thread
├── Metrics / Health Thread
└── Model Load / Control Thread
~~~

### 6.2 原則

- 一つのEngine processはMVPで一つのactive model instanceを持つ。
- MTLDevice、command queue、pipeline cacheのownerを明確にする。
- model load/unloadとinferenceの同時実行をstate machineで制御する。
- Metal completion handlerから本文をlogしない。
- command buffer数、active sequences、stream bufferをboundedにする。
- client切断、deadline、cancel時に未送信workとbufferを解放する。
- UI main threadを前提にしない。

---

## 7. Technology方針

### 7.1 Production最終形

~~~text
Native Engine Common Core
  -> Metal Backend
     -> MPSGraph / Metal Performance Shaders
     -> Lykuro Custom Metal Kernels
~~~

MPSGraphで対応可能なoperatorを先に利用し、性能または互換性が不足する部分だけcustom Metal Kernelへ置き換える。

### 7.2 MLX利用

MLX C++は次のいずれかに限定する。

1. initial backend implementation
2. correctness reference oracle
3. operator benchmark
4. approved low-level compute dependency

禁止:

- mlx-lm serverの起動
- MLX Python APIをProduction必須にする
- runtime中のpip install
- public model registryからの自動download
- MLX固有APIをGatewayまたはPlatform public contractへ公開

MLXをProduction linkする場合はversion pin、MIT license notice、SBOM、provenance、vulnerability review、reproducible buildを必須とする。

### 7.3 実装Mode

| Mode | 内容 | 用途 |
|---|---|---|
| metal_native | MPSGraph/MPS/Metal Kernel | Production目標 |
| mlx_cpp | MLX C++ backend | 初期導入または承認済みProduction |
| metal_reference | CPU/CUDA比較用 | Testのみ |

Modeはbuild/releaseで固定し、顧客が任意library pathを指定してbackendを差し替えられないようにする。

---

## 8. Hardware Backend Interface拡張

既存Interfaceを優先し、同義Interfaceを作らない。論理例:

~~~cpp
struct DeviceInfo {
    std::string backend;
    std::string device_name;
    std::string registry_id;
    bool unified_memory;
    uint64_t recommended_working_set_bytes;
    uint64_t current_allocated_bytes;
    uint64_t engine_budget_bytes;
    std::vector<std::string> capabilities;
};

class HardwareBackend {
public:
    virtual Status Initialize(const BackendConfig&) = 0;
    virtual Result<DeviceInfo> InspectDevice() = 0;
    virtual Result<Buffer> Allocate(const BufferSpec&) = 0;
    virtual Status Release(Buffer&) = 0;
    virtual Status UploadWeights(const WeightView&) = 0;
    virtual Status ExecutePrefill(const PrefillBatch&) = 0;
    virtual Status ExecuteDecode(const DecodeBatch&) = 0;
    virtual Status Synchronize(const Deadline&) = 0;
    virtual Result<BackendMetrics> Metrics() = 0;
    virtual Status Shutdown() = 0;
};
~~~

### 8.1 Interface条件

- CUDAとMetalで同じtensor shape/dtype semanticsを維持する。
- backend固有handleをEngine Coreへ漏らさない。
- asynchronous executionとcompletionを共通Future/Eventへ正規化する。
- errorを安定したEngine error codeへ変換する。
- model pluginがMetal固有classへ直接依存しない。
- test fake backendを利用可能にする。

---

## 9. Device Discovery・Capability

### 9.1 起動時Flow

1. arm64とmacOSを検証する。
2. default MTLDeviceを取得する。
3. Unified Memory対応を検証する。
4. device name、registry ID、feature/capabilityを取得する。
5. current allocationとrecommended working setを取得する。
6. metallibを署名/digest検証してloadする。
7. required pipelineを作成する。
8. minimal compute self-testを実行する。
9. certified profileを照合する。
10. success時だけEngineをreadyにする。

### 9.2 Capacity報告

~~~json
{
  "backend": "metal",
  "architecture": "arm64",
  "unified_memory": true,
  "recommended_working_set_bytes": 0,
  "current_allocated_bytes": 0,
  "engine_budget_bytes": 0,
  "model_bytes": 0,
  "kv_cache_budget_bytes": 0,
  "active_sequences": 0,
  "max_sequences": 0,
  "throttled": false
}
~~~

0は例示placeholderであり、Production値ではない。

---

## 10. Unified Memory管理

### 10.1 Budget

~~~text
engine_budget
= min(
    configured_memory_limit,
    recommended_max_working_set_size
  )
- os_and_service_reserve
- engine_core_reserve
- safety_margin
~~~

~~~text
admission_available
= engine_budget
- model_weight_bytes
- persistent_buffer_bytes
- current_kv_cache_bytes
- active_activation_bytes
- stream_buffer_bytes
~~~

### 10.2 原則

- physical Unified Memory総量だけでadmissionしない。
- MTLDeviceのcurrentAllocatedSizeとrecommendedMaxWorkingSetSizeを参照する。
- model load前にweight、temporary buffer、KV Cacheの最大見積りを行う。
- safety marginをconfig可能にし、最小値を下回らせない。
- memory pressure通知または急増時は新規admissionを停止する。
- OS swap、memory compression、process killを通常のcapacity拡張に使わない。
- OOM後に同じ設定で無限retryしない。
- model unload時にMetal resource、MPSGraph executable、KV Cacheを解放する。

### 10.3 Memory Class

| Class | Lifecycle |
|---|---|
| Model Weight | model load〜unload |
| Constant Buffer | model load〜unload |
| Pipeline/Graph Cache | Engineまたはmodel lifecycle |
| KV Cache | request/session policy |
| Activation | prefill/decode step |
| Temporary Scratch | command execution |
| Stream Buffer | request completion/cancel |

### 10.4 Pressure Level

| Level | 条件 | 動作 |
|---|---|---|
| normal | budget内 | 通常受付 |
| warning | warning watermark超過 | batch/sequence抑制 |
| critical | critical watermark超過 | 新規admission停止、cache回収 |
| exhausted | allocation失敗 | request拒否、model状態評価 |

memory pressure時もPrompt/Responseをdiskへ退避しない。

---

## 11. Tensor・Buffer

### 11.1 Storage

- Apple Siliconでは共有memory特性を利用する。
- CPU tokenizer outputからGPU inputへの不要copyを削減する。
- buffer alignmentをoperator/pipeline要件に合わせる。
- weightはread-only semanticsとする。
- temporary resourceをrequest ownershipへ関連付ける。
- buffer reuse時に前requestの内容をzeroizeまたは上書き保証する。

### 11.2 Synchronization

- resource read/write conflictをevent、fence、barrierで制御する。
- command buffer completion前にbufferを再利用しない。
- cancel後もGPU work completionまたはsafe abandonを確認してresourceを回収する。
- CPU/GPU visibilityを明示し、偶然の同期へ依存しない。

---

## 12. Operator対応

### 12.1 MVP必須

- embedding lookup
- matrix multiplication
- RMSNorm
- RoPE
- causal attention
- grouped-query attention（対象modelが必要な場合）
- SiLU/SwiGLU
- residual add
- softmax
- logits projection
- top-k/top-pに必要な処理
- KV Cache read/write

### 12.2 実装優先

1. correctness確認済みMPSGraph/MPS operator
2. verified Metal Kernel
3. approved MLX C++ operator
4. CPU fallback

CPU fallbackは明示したoperatorとprofileだけ許可し、性能低下をmetricsへ記録する。未対応operatorを黙ってCPUへ移さない。

### 12.3 Custom Kernel

- source review
- deterministic test
- shape/dtype validation
- bounds check
- NaN/Inf検査
- command timeout
- metallib digest
- chip family別benchmark

runtimeで未署名Metal sourceをcompileしない。release packageには事前compileしたmetallibを含める。

---

## 13. Model・Artifact

### 13.1 共通Artifact再利用

既存Model Artifact、manifest、signature、checksums、license review、Architecture Pluginを再利用する。

追加manifest項目:

~~~json
{
  "backend_profiles": [
    {
      "backend": "metal",
      "architecture": "arm64",
      "precision": "fp16",
      "minimum_working_set_bytes": 0,
      "engine_api_range": "as-built-range",
      "backend_abi_range": "as-built-range",
      "metallib_digest": "sha256:example",
      "certified_profile_ids": []
    }
  ]
}
~~~

### 13.2 Load検証

1. manifest schema
2. signature/digest
3. license review
4. model architecture
5. tensor name/shape/dtype
6. tokenizer/template
7. Metal backend ABI
8. metallib digest
9. working set見積り
10. certified profile
11. staging load
12. smoke inference

Mac用に無検証変換したweightをProductionへloadしない。

---

## 14. 推論Pipeline

~~~text
Request
  -> Existing API Validation
  -> Tokenizer / Template
  -> Metal Admission
  -> Prefill Graph / Kernels
  -> KV Cache
  -> Decode Loop
  -> Sampling
  -> Streaming
  -> Resource Release
~~~

### 14.1 Correctness

- CUDA/CPU referenceとtokenizer inputを一致させる。
- greedy modeでtoken列を比較する。
- FP16許容誤差をoperatorごとに定義する。
- logits、stop、seed、usage、finish reasonを検証する。
- Metal最適化前後でgolden regressionを実行する。

---

## 15. Scheduler・Batch

既存Schedulerを再利用する。

Metal追加条件:

- command buffer上限
- active sequence上限
- total batch token上限
- KV Cache budget
- prefill/decode別memory見積り
- thermal/memory pressure
- deadlineまでの推定実行可能性

MVPはcorrectnessを優先し、single requestから開始する。continuous batchingはgolden、memory、cancel、soak test合格後に有効化する。

---

## 16. KV Cache

### 16.1 原則

- KV CacheをConversation Memoryの正本にしない。
- node再起動またはeviction時はPlatformがcontextを再送できる。
- tenant_scope、model_instance、sequence、cache generationで所有権を分離する。
- request終了、cancel、deadline、model unload時に解放する。
- cache handleからmemory addressを推測できない形式にする。

### 16.2 Budget

KV Cache上限はUnified Memory engine budgetの一部として予約する。model load後の残量から安全に計算し、設定値だけを信頼しない。

---

## 17. Data API・Control API

外部contractを変更しない。

### 17.1 Data API

| Operation | macOS対応 |
|---|---|
| Generate | 必須 |
| GenerateStream | 必須 |
| Cancel | 必須 |
| CountTokens | 既存Tokenizerを再利用 |
| GetCapabilities | Metal capabilityを追加 |

### 17.2 Control API

| Operation | macOS対応 |
|---|---|
| LoadModel | working set/Metal profile検証を追加 |
| UnloadModel | Metal resource解放 |
| Drain | 新規admission停止 |
| Resume | health/capacity再検証 |
| GetModelStatus | backend/profileを追加 |
| GetCapacity | Unified Memory項目を追加 |
| GetManifest | Metal profile metadataを追加 |

Model ManagerだけがControl APIを利用する。GatewayはControl APIを利用できない。

---

## 18. Platform・Model Manager統合

### 18.1 Node登録

Mac nodeが報告する項目:

~~~text
node_id
platform: macos
architecture: arm64
engine_version
engine_api_version
backend: metal
device_name
unified_memory: true
recommended_working_set_bytes
current_allocated_bytes
engine_budget_bytes
loaded_model
precision
queue_depth
active_sequences
ttft
tokens_per_second
thermal_state
health
last_seen_at
~~~

Prompt、Response、Token ID、KV Cache、conversation本文をheartbeatへ含めない。

### 18.2 Routing

Router選択条件:

1. tenant/project policy
2. data class/network zone
3. Logical Model
4. model/profile compatibility
5. backend=metal
6. working set余裕
7. queue/active sequence
8. thermal/memory pressure
9. TTFT/tokens per second
10. node health

Metal nodeをCloud fallbackとして扱わない。顧客環境内のLocal nodeである。

### 18.3 Control

~~~text
Model Manager
  -> Engine Control API
     -> LoadModel
     -> UnloadModel
     -> Drain
     -> Resume
     -> GetCapacity
~~~

Model ManagerはMacへXcode、Homebrew、Python、Ollamaをinstallしない。

---

## 19. Distributed利用

### 19.1 MVP

複数MacはRequest Shardingで利用できる。

~~~text
Lykuro Distributed Coordinator
├── Mac Node A / Metal Engine
├── Mac Node B / Metal Engine
└── CUDA Node C / CUDA Engine
~~~

### 19.2 制約

- 複数MacのUnified Memoryは一つの共有memoryにならない。
- 一つのmodelを複数Macへ分割する機能はMVP対象外。
- CUDA nodeとMetal nodeで同じLogical Modelを提供する場合は個別Certified Profileを必要とする。
- model/backend間で出力差がある場合はrouting/auditへprofileを記録する。
- sticky routingは最適化であり会話継続の条件にしない。

---

## 20. Configuration

### 20.1 Engine Config

~~~yaml
engine:
  id: nie-mac-node-01
  backend: metal
  listen_address: 127.0.0.1
  grpc_port: 19443
  state_dir: /Library/Application Support/Lykuro/NativeEngine
  log_level: info

security:
  mtls_required: true
  egress_disabled: true
  content_logging: false
  crash_dump_enabled: false
  server_cert_ref: file:///path/to/server.crt
  server_key_ref: file:///path/to/server.key
  client_ca_ref: file:///path/to/client-ca.crt

metal:
  device: system_default
  execution_mode: metal_native
  metallib_path: /Library/Application Support/Lykuro/NativeEngine/lib/lykuro.metallib
  memory_limit_bytes: auto
  os_reserve_bytes: auto
  safety_margin_percent: 15
  warning_watermark_percent: 80
  critical_watermark_percent: 90
  max_inflight_command_buffers: 8

model:
  artifact_path: /Library/Application Support/Lykuro/Models/current
  precision: fp16

scheduler:
  max_queue: 64
  max_sequences: 1
  max_batch_tokens: 4096

observability:
  metrics_enabled: true
  metrics_address: 127.0.0.1
  metrics_port: 19090
~~~

数値は初期例であり、実機benchmark前にProduction既定値として固定しない。

### 20.2 Validation

- unknown key
- backend/architecture
- device存在
- Unified Memory
- metallib path/digest/signature
- memory limit/watermark
- model/profile compatibility
- certificate/permission
- API/ABI version
- unsupported combination

secret原文を通常configへ含めない。

---

## 21. Error設計

既存Engine error schemaを再利用し、次のcodeを追加する。

| Code | 条件 | Retry |
|---|---|---:|
| metal_backend_unavailable | Metal device取得失敗 | × |
| metal_device_unsupported | feature/profile不一致 | × |
| metal_library_invalid | metallib署名/digest不正 | × |
| metal_pipeline_creation_failed | pipeline作成失敗 | 条件 |
| metal_command_failed | GPU command失敗 | 条件 |
| metal_memory_pressure | warning/critical pressure | ○ |
| metal_out_of_memory | allocation失敗 | 条件 |
| metal_execution_timeout | command/deadline超過 | 条件 |
| mac_profile_not_certified | certified matrixにない | × |
| backend_abi_mismatch | Core/Backend不一致 | × |

Error messageへPrompt、Response、token、weight path、credentialを含めない。

---

## 22. Security

### 22.1 Network

- Data/Control APIをpublic interfaceへbindしない。
- localhostまたは顧客管理networkに限定する。
- Platform/Model ManagerとのmTLSを必須とする。
- Control API scopeをData API scopeから分離する。
- public egressを既定denyとする。
- runtime中のmodel/dependency downloadを禁止する。

### 22.2 Process

- 専用service user
- non-root
- Hardened Runtime
- 最小entitlement
- read-only model artifact
- secret file permission
- crash dump既定OFF
- child process生成禁止またはallowlist
- arbitrary dylib/plugin load禁止

### 22.3 Supply Chain

- Developer ID署名
- Apple notarization
- secure timestamp
- package signature
- SHA-256
- SBOM
- provenance
- dependency/license report
- metallib digest
- release compatibility matrix
- vulnerability/secret scan

### 22.4 Data

- Prompt/Response永続保存0日
- Token IDはrequest終了時削除
- KV Cacheはbounded temporary data
- log/metric/traceへ本文禁止
- diagnose bundleへ本文、weight、token、KV、credential禁止
- memory reuse時のscope isolation

---

## 23. macOS Native Package

### 23.1 Production Artifact

~~~text
lykuro-native-engine-macos-arm64/
├── bin/
│   └── lykuro-native-engine
├── lib/
│   ├── lykuro.metallib
│   └── approved-runtime-libraries
├── config/
│   └── engine.example.yaml
├── launchd/
│   └── ai.lykuro.native-engine.plist
├── contracts/
│   └── engine-api-v1.*
├── compatibility/
│   └── certified-profiles.json
├── checksums.sha256
├── manifest.json
├── signature.sig
├── sbom/
├── licenses/
└── docs/
    ├── INSTALL.md
    ├── UPGRADE.md
    ├── ROLLBACK.md
    └── SECURITY.md
~~~

### 23.2 含めないもの

- Ollama
- llama.cpp
- vLLM
- TGI
- mlx-lm server
- Python runtime
- Node.js/npm/nvm
- Homebrew
- Xcode
- model weight本体
- unsigned dynamic library
- development signing key

### 23.3 配置

System service:

~~~text
/Library/Application Support/Lykuro/NativeEngine/
/Library/LaunchDaemons/ai.lykuro.native-engine.plist
/Library/Logs/Lykuro/NativeEngine/
~~~

Rootless/user service:

~~~text
~/Library/Application Support/Lykuro/NativeEngine/
~/Library/LaunchAgents/ai.lykuro.native-engine.plist
~/Library/Logs/Lykuro/NativeEngine/
~~~

正式pathはinstaller/enterprise policyに合わせる。spaceを含むpathを安全に扱い、shell文字列連結に依存しない。

### 23.4 Host Native必須

Mac版EngineはmacOS host-native processとして実行する。Docker DesktopはMac上でLinux VMを使用し、一般container GPU対応をMac Metal Backendの前提にしない。

Platform/GatewayをDocker/OrbStackで実行する場合:

~~~text
Containerized Gateway / Platform
  -> host-only mTLS connection
  -> macOS Host Native Engine
  -> Metal GPU
~~~

Engine socket/portをpublic LANへ無条件公開しない。

---

## 24. Installer・Service

### 24.1 Precheck

- Apple Silicon arm64
- macOS version
- Metal device/capability
- Unified Memory/working set
- memory/disk
- existing Engine version
- package signature/notarization
- port/socket
- certificate/secret
- model artifact/profile
- launchd permission
- Platform connectivity
- clock/time sync

precheckはread-onlyで、OSやdependencyを変更しない。

### 24.2 Install

1. package signature/notarization検証
2. manifest/checksum検証
3. compatibility確認
4. target directory作成
5. executable/metallib/config配置
6. service identity/permission設定
7. certificate/secret参照設定
8. launchd登録
9. Engine起動
10. Metal self-test
11. Platform登録
12. model staging load/smoke

### 24.3 条件

- sudoを前提にしない。
- system serviceに管理者権限が必要な場合は事前表示して明示実行する。
- Homebrew、Xcode、Pythonを自動installしない。
- 再実行しても二重service登録または設定破壊を起こさない。
- uninstallはmodel、audit、Platform設定を既定削除しない。
- --purgeは対象と影響を表示して確認を要求する。

---

## 25. Code Signing・Notarization

macOS外部配布artifactはDeveloper IDで署名し、Apple notarizationを実施する。

必須:

- main executable署名
- bundled dylib/framework署名
- metallib digest検証
- Hardened Runtime
- secure timestamp
- notarization result
- ticket stapling可能なpackage形式
- Gatekeeper検証test

CIのDeveloper ID credentialをartifact、log、repositoryへ保存しない。Air-Gapped顧客向けpackageも事前notarizationと署名を行い、offlineで検証可能にする。

---

## 26. Update・Rollback

### 26.1 Version

- Engine version
- Data API version
- Control API version
- Backend ABI
- Metal Backend version
- metallib version/digest
- Model Plugin version
- Manifest schema
- Certified Profile version

### 26.2 Update Flow

1. new package signature/notarization検証
2. compatibility確認
3. Model Managerがdrain
4. active request終了
5. config/state snapshot
6. new version staging配置
7. Metal self-test
8. model load/smoke
9. readiness
10. traffic切替
11. observation
12. old version保持期限後に削除

### 26.3 Auto Rollback

- process crash
- Metal device初期化失敗
- metallib/pipeline load失敗
- model load/smoke失敗
- readiness timeout
- error rate閾値超過
- memory pressure/OOM regression
- TTFT/tokens per second重大劣化
- golden correctness失敗

rollbackでmodel weight、conversation、auditを変更・削除しない。

---

## 27. Observability

### 27.1 Metrics

| Category | Metrics |
|---|---|
| Engine | state、uptime、version、backend |
| Metal | command submitted/completed/failed、pipeline cache |
| Memory | working set、allocated、budget、pressure、allocation failure |
| Model | load time、model bytes、instance state |
| Scheduler | queue、wait、active sequence、rejection |
| Prefill | latency、tokens/sec、batch tokens |
| Decode | latency、tokens/sec、active sequence |
| KV | used、free、eviction、fragmentation |
| Stream | buffer、cancel、slow consumer |
| System | thermal state、process memory、CPU |

### 27.2 Label制限

許可:

- engine_id
- backend=metal
- model_instance_id
- model_family
- device profile ID
- result code
- precision

禁止:

- Prompt/Response
- user入力
- API key/JWT
- user ID原文
- conversation title
- unbounded request ID label

### 27.3 Diagnose Bundle

- Engine/Backend/metallib version
- secret除去config
- device/capability
- memory budget/current allocation
- model manifest metadata
- recent error code
- metrics snapshot
- code signing/notarization status

本文、weight、token、KV Cache、credentialを含めない。

---

## 28. Performance・Certified Profile

### 28.1 Profile

~~~yaml
profile_id: cp_mac_m4_64gb_qwen_example
engine_version: pending
metal_backend_version: pending
metallib_digest: pending
model_artifact_id: pending
model_architecture: pending
hardware:
  chip: Apple M4
  unified_memory_gb: 64
software:
  macos_version: pending
  metal_version: pending
precision: fp16
context_tokens: pending
max_sequences: pending
memory:
  recommended_working_set_bytes: pending
  engine_budget_bytes: pending
targets:
  ttft_p95_ms: pending
  output_tokens_per_second: pending
  load_time_seconds: pending
  soak_hours: 24
~~~

pending値は実機測定後に固定する。推測値をCertified Profileとして発行しない。

### 28.2 Benchmark

- cold/warm model load
- single request
- short/long input
- short/long output
- stream/non-stream
- context size別
- max queue
- continuous batch
- cancel/deadline
- KV Cache growth/eviction
- memory watermark
- 1時間/24時間soak
- repeated load/unload
- sleep/wake後の動作
- foreground application負荷との競合
- thermal throttling

CUDAとMacを単純なtokens/secだけで比較せず、model、precision、context、batch、qualityを固定する。

---

## 29. Test方針

### 29.1 GPUなしCI

- Backend Interface unit test
- Fake Metal Backend
- allocator/state/admission
- config/schema
- manifest/profile
- error mapping
- scheduler/cancel/deadline
- API contract
- secret/content redaction
- package manifest

GPU simulator結果をProduction performance証明に使用しない。

### 29.2 Mac実機CI

- Metal device discovery
- metallib load
- operator unit
- tensor shape/dtype
- memory allocation/release
- command synchronization
- model load/unload
- Generate/Stream/Cancel
- OOM/memory pressure
- process restart
- package install/update/rollback

### 29.3 Golden Correctness

- CPUまたはCUDA reference logits
- greedy token sequence
- sampling distribution
- tokenizer/template
- stop sequence
- usage count
- streaming order
- NaN/Inf
- multiple context lengths

### 29.4 Security

- unsigned package/metallib拒否
- invalid model signature/digest拒否
- arbitrary dylib load拒否
- path traversal
- malformed tensor
- oversized input
- mTLS/authorization
- Gateway Control API access拒否
- public egress確認
- Prompt/Response非保存
- diagnose redaction

### 29.5 Performance・Soak

- memory/resource leak
- long stream
- repeated cancel
- queue saturation
- repeated load/unload
- 24時間soak
- memory pressure recovery
- thermal state変化

未実行項目をpassと記載しない。

---

## 30. CI/CD

### 30.1 Build

- pinned compiler/Xcode/Metal SDK
- arm64 release build
- deterministic/reproducible設定
- metallib事前compile
- unit/golden test
- SBOM/license/provenance
- code sign
- notarize
- staple/verify
- package/checksum/signature

### 30.2 Runner

| Runner | 用途 |
|---|---|
| Standard CI | Core/unit/schema/security |
| Apple Silicon Mac | Metal integration/golden |
| Mac mini M4 64GB | capacity/performance/soak |
| CUDA machine | reference regression |

Performance baselineは同じ実機profile上で比較する。

---

## 31. 障害・復旧

| 障害 | 処理 |
|---|---|
| Metal device取得不可 | Engine failed、ready=false |
| metallib不正 | 起動/load拒否 |
| pipeline作成失敗 | model unavailable、rollback候補 |
| memory pressure warning | batch/sequence抑制 |
| critical pressure | admission停止、cache回収 |
| allocation失敗 | request拒否、無限retry禁止 |
| command failure | request失敗、device health評価 |
| timeout | cancel、resource回収 |
| process crash | launchd restart、startup recovery |
| sleep/wake | device/queue/pipeline再検証 |
| model load失敗 | previous model/Engine維持 |
| Platform切断 | deadline/cancel、本文queue禁止 |
| metrics停止 | inference継続、alert |

### 31.1 Startup Recovery

- incomplete model load state破棄
- signed config再検証
- package/metallib/model digest再検証
- Metal device/self-test
- previous crash reason
- model staging load
- readiness

---

## 32. 実装Phase

### Phase 0: As-Built・Feasibility

- existing Backend Interface mapping
- Apple Silicon/Metal device調査
- MPSGraph/MLX/operator feasibility
- target model/profile確定
- golden oracle
- build/sign/notarization設計

終了条件: 共通Coreを変更せず一つのoperatorを実機実行できる。

### Phase 1: Metal Correctness PoC

- device/allocator
- weight buffer
- MVP operator
- single request prefill/decode
- greedy generation
- golden test

終了条件: reference tolerance内のtoken/logit結果。

### Phase 2: Engine Integration

- Data/Control API
- streaming、sampling
- cancel/deadline
- KV Cache
- bounded scheduler
- Model Manager registration

終了条件: Platform→Mac Engineでend-to-end inference可能。

### Phase 3: Product Hardening

- mTLS
- code signing/notarization
- signed metallib/model
- memory pressure/OOM
- install/update/rollback
- security/fuzz/soak

終了条件: Production security/reliability review合格。

### Phase 4: Performance

- continuous batching
- allocator改善
- graph/pipeline cache
- fused Metal Kernel
- INT4/INT8
- KV Cache最適化

終了条件: Certified Profile目標達成。

### Phase 5: Expansion

- additional model
- embeddings
- vision
- multiple instances
- request-sharded multi-Mac pool
- Neural Engine個別検証

各機能をfeature flagとCertified Profileで提供する。

---

## 33. 受入基準

| ID | 項目 | 合格条件 |
|---|---|---|
| AT-M01 | Platform | Apple Silicon arm64で起動する |
| AT-M02 | Host Native | Docker/VM GPUを前提にせずmacOS processでMetalを利用する |
| AT-M03 | 独自性 | Ollama/llama.cpp/vLLM/mlx-lm serverを含まない |
| AT-M04 | API | 既存Generate/Stream/Cancel/Control APIと互換 |
| AT-M05 | Control | Model Managerだけがload/unload/drain可能 |
| AT-M06 | Correctness | Certified golden testに合格 |
| AT-M07 | Model | signature/digest/profile不正modelを拒否 |
| AT-M08 | Metal | metallib/ABI不正を拒否 |
| AT-M09 | Memory | working setに基づくbounded admissionが動作 |
| AT-M10 | Pressure | warning/critical/OOMからcrash・漏えいなく回復または拒否 |
| AT-M11 | KV Cache | scope isolation、release、evictionが動作 |
| AT-M12 | Streaming | event order、cancel、slow consumerが正しい |
| AT-M13 | 本文非保存 | log/metric/trace/disk/diagnoseへ本文が残らない |
| AT-M14 | Security | mTLS、non-root、egress deny、Hardened Runtime |
| AT-M15 | Package | Developer ID署名、notarization、SBOM、provenanceが存在 |
| AT-M16 | Install | Python/Node/Homebrew/Xcodeなしで顧客hostへ導入可能 |
| AT-M17 | Update | signed updateとrollbackが動作 |
| AT-M18 | Soak | 24時間testでmemory/resource leakなし |
| AT-M19 | Performance | 発行済みCertified Profile目標を達成 |
| AT-M20 | Integration | Native LLM Platformからroute/load/inference可能 |

---

## 34. Definition of Done

- existing Engine repository調査とmapping reportがある。
- Engine Core/Data API/Control APIの後方互換を維持している。
- Metal Backend Interfaceとimplementationがある。
- Apple Silicon device/capability検査がある。
- Unified Memory budget、watermark、admissionがある。
- model/metallib/package署名検証がある。
- MVP operator、prefill、decode、sampling、streamが動作する。
- Scheduler、KV Cache、cancel、deadlineが動作する。
- Platform/Model Manager integrationが動作する。
- ProductionにPython/Node/Homebrew/Xcodeが不要である。
- third-party inference serverを含めていない。
- Developer ID署名、notarization、Hardened Runtimeがある。
- unit、golden、integration、security、performance、soak test結果がある。
- Mac mini M4 64GB Certified Profileが実測値で発行されている。
- install、monitor、update、rollback、recovery文書がある。
- SBOM、provenance、license、vulnerability reportがある。
- 未実装、未検証chip/model、性能未達が明記されている。

---

## 35. Repository構成

既存構成を優先する論理例:

~~~text
lykuro-native-inference-engine/
├── include/
│   └── backend/
├── src/
│   ├── core/
│   ├── backend/
│   │   ├── cuda/
│   │   └── metal/
│   ├── scheduler/
│   ├── model/
│   └── api/
├── shaders/
│   └── metal/
├── tests/
│   ├── unit/
│   ├── golden/
│   ├── metal/
│   └── performance/
├── deploy/
│   └── macos/
├── packaging/
│   └── macos/
└── docs/
    └── metal/
~~~

Metal sourceをmodel pluginへ散在させずbackend moduleへ隔離する。

---

## 36. 要決定事項

| ID | 論点 | 推奨初期値 |
|---|---|---|
| D-M01 | Target | Apple Silicon arm64のみ |
| D-M02 | Production backend | metal_native |
| D-M03 | Initial compute | MPSGraph/MPS優先 |
| D-M04 | MLX | C++ dependencyまたはtest oracle。server/Python禁止 |
| D-M05 | Initial precision | FP16 |
| D-M06 | Initial model | 既存認証済みQwen Denseから1種類 |
| D-M07 | Model per process | 1 |
| D-M08 | Packaging | signed/notarized PKG + tar |
| D-M09 | Service | host-native launchd |
| D-M10 | Container GPU | 対象外 |
| D-M11 | Memory | recommended working set基準 |
| D-M12 | Mac profile | Mac mini M4 64GB |
| D-M13 | Multi-Mac | Request Shardingのみ |
| D-M14 | Neural Engine | 将来の個別検証 |

実機testなしにmacOS最低version、最大context、batch、model size、性能値をProduction確定しない。

---

## 37. Claude Code最終報告形式

~~~markdown
## Existing Engine As-Built

## 本追加仕様との対応表

## Reuse・Extend・New・Out of Scope

## Apple Silicon・Metal検証環境

## Backend Interface・Metal実装

## MPSGraph・Metal Kernel・MLX依存

## Unified Memory・Allocator・KV Cache

## Model・Artifact・Certified Profile

## Data API・Control API互換性

## Platform・Model Manager統合

## Security・Signing・Notarization

## Package・Installer・launchd

## 実行したTestと結果

## Performance・Soak

## Update・Rollback・Recovery

## 変更ファイル

## 未実装・未検証・既知問題
~~~

GPU、model、signing certificate、notarization、実機runner等がなく検証できない場合は、未検証理由、必要環境、再現可能な手順を記載する。test未実行を成功と記載しない。

---

## 38. 参考資料

- [Apple Metal MTLDevice](https://developer.apple.com/documentation/metal/mtldevice)
- [Apple Metal Performance Shaders Graph](https://developer.apple.com/documentation/metalperformanceshadersgraph)
- [Apple Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [Apple Metal Resource Synchronization](https://developer.apple.com/documentation/metal/resource-synchronization)
- [Apple Metal Libraries](https://developer.apple.com/documentation/metal/metal-libraries)
- [Apple MLX](https://github.com/ml-explore/mlx)
- [Apple macOS Software Notarization](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution)
- [Docker Desktop GPU Support](https://docs.docker.com/desktop/features/gpu/)

Appleおよびdependency仕様は変更されるため、releaseごとに公式資料とCompatibility Matrixを再確認する。

---

## 改訂履歴

| 版 | 日付 | 内容 |
|---|---|---|
| v1.0 | 2026-08-08 | Apple Silicon、Metal Backend、Unified Memory、macOS native package、実機test、M4 64GB Certified Profileを追加 |
| v1.1 | 2026-08-15 | 性能path追記(本体仕様v1.2 §17.6/§14.5参照): MPSGraphに加え自前MSL kernel backend(metal-fast/q8/q4、runtimeコンパイル・1 token=1 command buffer・split-row flash-decoding attention・INT8/INT4 weight-only)、greedy投機pipeline(on-GPU argmax + embedding gather、二重command buffer先行commit)。macOS既定backendはmetal-q4。M4 Pro実測: Qwen2.5-0.5B 485 tok/s / 1.5B 210 tok/s(release v1.0.3〜v1.0.4) |
