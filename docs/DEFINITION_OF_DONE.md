# Definition of Done — status

Per-item status against LYK-NIE-SD-001 §33 and LYK-NIE-ADD-METAL-001 §34.
✅ done · ◑ partial · ⏸ deferred (environment/credential-gated) · ❌ not started

## Core engine (LYK-NIE-SD-001 §33)

| Item | Status | Notes |
|---|---|---|
| Repo structure, build, dependency policy fixed | ✅ | CMake presets, allowlist deps, SBOM |
| No third-party inference runtime (src/bin/image/transitive) | ✅ | CI gate `check_no_forbidden_runtime.sh` |
| API/protobuf + versioning documented | ✅ | `lykuro.nie.v1`, N/N-1 policy |
| Model Artifact/Manifest schema versioned | ✅ | schema v1, unknown-field rejection |
| Architecture Plugin interface + Qwen plugin | ✅ | `GenerativeModel`, `approved_qwen_decoder_v1` |
| loader/tokenizer/prefill/decode/sampling/streaming | ✅ | verified vs oracle |
| Scheduler, batch, KV, cancel, deadline | ✅ | + continuous/paged/prefix cache |
| mTLS, signed artifact, egress deny, non-root | ✅ | non-root documented in launchd/plist |
| No prompt/response in log/metric/trace/disk | ✅ | content-free logger + tests |
| unit/golden/integration/security/perf/soak tests | ✅ | incl. 24h CUDA soak (237k req, 0 failed, 0 RSS drift, 100 reload leak-free) |
| Certified hw/model profile issued | ◑ | dev-measured; production cert pending |
| SBOM, provenance, signature, license notice | ✅ | `sbom/`, `licenses/`, signed manifest |
| deployment/monitoring/update/rollback/recovery docs | ◑ | launchd + release/rollback notes; full runbook pending |
| Unimplemented / unverified items documented | ✅ | README + compatibility matrix |

## Metal addendum (LYK-NIE-ADD-METAL-001 §34)

| Item | Status | Notes |
|---|---|---|
| Existing-engine survey + mapping report | ✅ | `docs/metal/as-built-report.md` |
| Core/Data/Control API backward compat | ✅ | no API change; same proto |
| Metal Backend interface + implementation | ✅ | `backends/metal` |
| Apple Silicon device/capability checks | ✅ | `InspectMetalDevice` |
| Unified Memory budget/watermark/admission | ◑ | load-time budget admission; staged watermarks pending |
| model/metallib/package signature verification | ◑ | model+package signed; metallib N/A (MPSGraph-only) |
| MVP operator/prefill/decode/sampling/stream | ✅ | oracle 3/3 on M4 Pro |
| Scheduler/KV/cancel/deadline | ✅ | shared common core |
| Platform/Model Manager integration | ✅ | gRPC serving e2e |
| No Python/Node/Homebrew/Xcode in production | ✅ | host-native binary + frameworks |
| No third-party inference server | ✅ | same CI gate |
| Developer ID sign, notarization, Hardened Runtime | ◑ | pipeline implemented + wired (`deploy/macos/sign_and_notarize.sh`, entitlements, make_package hook); codesign form validated ad-hoc; awaits Developer ID cert to run for real |
| unit/golden/integration/security/perf/soak results | ✅ | 176/176 tests green; 24h Metal soak PASSED (137440 completed, 0 failed, footprint flat ~2656MB, 100 reload cycles leak 11.7MB) |
| Mac mini M4 64GB Certified Profile (measured) | ◑ | dev-measured profile issued: `cp_qwen25_05b_m4pro_dev1.yaml` (incl. 24h soak); formal/cross-host cert pending |
| install/monitor/update/rollback/recovery docs | ◑ | scaffolding in `deploy/macos` |
| SBOM/provenance/license/vuln report | ✅ | shared SBOM; CVE gate `tools/scan_vulnerabilities.sh` (SBOM×VEX ledger×OSV) wired in CI, emits `vulnerability-report.json` |
| Unimplemented/unverified/underperformance documented | ✅ | README + this file |

## Remaining before production sign-off

1. ~~24h CUDA soak~~ ✅ passed (237k requests, 0 failed, RSS bit-identical,
   100 reload cycles leak-free). ~~24h Metal soak~~ ✅ passed
   (137440 completed, 0 failed, footprint flat ~2656MB over 24h,
   100 reload cycles leak 11.7MB < 128MB gate) after fixing a
   ~6MB/request phys_footprint leak (missing autorelease-pool drain
   on the worker thread). Earlier attempts died to system idle sleep
   and then to the leak; final run under `caffeinate`.
2. macOS Developer ID signing + Apple notarization + Gatekeeper test.
   Pipeline is implemented and self-skipping (`deploy/macos/sign_and_notarize.sh`
   + `entitlements.plist` + `make_package.sh` hook); only the cert-gated run
   remains — see `deploy/macos/README.md` for the two-command flow.
3. Formal Certified Profile issuance from a signed-artifact-only,
   security-reviewed build.
4. ~~CVE scan wired as a hard CI gate producing `vulnerability-report.json`~~
   ✅ done — `vuln-scan` job runs `tools/scan_vulnerabilities.sh` on every
   push/PR: fail-closed on unreviewed deps (SBOM×`security/vex-ledger.json`),
   OSV cross-check, blocks un-waived CRITICAL/HIGH. Current state: 1 HIGH
   (OpenSSL `UBUNTU-CVE-2026-45447`, PKCS#7 UAF) tracked under a time-bound
   `under_remediation` waiver — host libssl3 upgrade to 3.0.2-0ubuntu1.25
   due 2026-09-01; engine does not exercise the PKCS#7 path. The residual
   waiver is what keeps the human security review (item 3) open.
