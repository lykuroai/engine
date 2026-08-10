# Operations Runbook — Lykuro Native Inference Engine

Deployment, monitoring, update, rollback, and recovery for the engine as a
host-native service (spec §24 / addendum §34). Grounded in what the binary
actually exposes: a JSON config (`core/engine/config.*`), an mTLS gRPC API
(`ControlService` / `DataService`), and a Prometheus metrics endpoint. The
engine is content-free by construction — no prompt or response text appears
in any log, metric, or trace, so nothing here instructs collecting it.

macOS is the reference platform (launchd). The Linux/CUDA host uses the
analogous systemd unit; mechanisms map one-to-one and are noted where they
differ.

---

## 1. Service layout (macOS)

Canonical production root (matches the LaunchDaemon):

```
/Library/Application Support/Lykuro/NativeEngine/
├── bin/lykuro-native-engine        # + bundled lib/ (self-contained)
├── config/engine.json              # active config (JSON; see §2)
├── secrets/                        # mTLS material, file-referenced only
│   ├── server.crt  server.key  client-ca.crt
└── ...
/Library/Application Support/Lykuro/Models/current   # model artifact
/Library/LaunchDaemons/ai.lykuro.native-engine.plist
/Library/Logs/Lykuro/NativeEngine/engine.{out,err}.log
```

The service runs as the dedicated non-root user `_lykuro`. launchd
`KeepAlive{Crashed=true, SuccessfulExit=false}` restarts it on crash with a
10 s `ThrottleInterval` backoff.

> The `--dev` ad-hoc package installs to `/usr/local/lykuro-native-engine`
> for lightweight internal testing and is **not** wired to launchd. Use the
> layout above for any managed deployment.

---

## 2. Configuration (authoritative keys)

The loader reads **JSON** (`LoadFileConfig`); the shipped `engine.example.yaml`
documents the same shape for operators. Keys and defaults
(`core/engine/config.h`):

| Key | Default | Notes |
|---|---|---|
| `engine_id` | `nie-node-01` | node identity in logs/metrics |
| `listen_address` | `127.0.0.1` | loopback / host-only; never public |
| `grpc_port` | `19443` | mTLS API |
| `log_level` | `info` | `debug\|info\|warn\|error` |
| `mtls_required` | `true` | keep true in production |
| `server_cert_path` / `server_key_path` / `client_ca_path` | — | required when mTLS on; file references only |
| `allow_unsigned_dev` | `false` | **must stay false in production** (fail-closed on unsigned artifacts) |
| `artifact_path` | — | model loaded at startup when set |
| `hardware_backend` | `cpu` | `cpu\|cuda` (Linux); Metal build selects Metal |
| `max_queue` | `256` | admission queue depth |
| `max_sequences` | `8` | concurrent decode batch |
| `max_output_tokens` | `4096` | per request cap |
| `max_input_bytes` | `1048576` | prompt byte cap |
| `metrics_enabled` | `false` | enable for production |
| `metrics_port` | `19090` | Prometheus scrape target |

Secrets are referenced by path only — never inline. Rotate certs by
replacing the files and reloading (§4).

---

## 3. Monitoring

### Metrics (Prometheus, `:19090/metrics`)

| Metric | Meaning | Watch for |
|---|---|---|
| `nie_requests_received_total` | arrivals | traffic baseline |
| `nie_requests_admitted_total` | passed admission | — |
| `nie_requests_rejected_total` | refused at admission | **rising** → queue full / over budget |
| `nie_requests_completed_total` | finished ok | — |
| `nie_requests_failed_total` | terminal errors | **any sustained rise** → investigate |
| `nie_requests_cancelled_total` | client/ deadline cancels | context |
| `nie_output_tokens_total` | generated tokens | throughput |

Suggested alerts:
- `rate(nie_requests_failed_total[5m]) > 0` sustained 10 m → page.
- `rate(nie_requests_rejected_total[5m])` climbing while receive rate flat
  → capacity/watermark pressure (see §6).
- metrics endpoint unreachable 2 m → service down.

### Liveness / readiness

No separate health RPC. Use the Control API (Manager identity, mTLS):
- **Liveness**: TCP connect to `grpc_port` succeeds and the process is up
  (launchd reports it).
- **Readiness**: `ControlService.GetModelStatus` reports a loaded model;
  `GetCapacity` returns current queue/sequence headroom.

### Logs

`engine.out.log` / `engine.err.log` are **content-free** (structured status
only). Ship them as-is; there is no prompt/response to redact.

---

## 4. Update (rolling, zero-drop)

Packages are versioned and immutable. The graceful sequence drains in-flight
work before the swap so no request is lost:

1. **Drain**: `ControlService.Drain` — stops admitting new requests, lets
   in-flight sequences finish. Poll `GetCapacity` until active == 0.
2. **Install** the new signed package (Phase 2) over the root:
   `sudo installer -pkg lykuro-native-engine-macos-metal-<new>.pkg -target /`
3. **Restart** under launchd:
   `sudo launchctl kickstart -k system/ai.lykuro.native-engine`
4. **Verify**: process up, `GetModelStatus` ready, `nie_requests_failed_total`
   flat, a canary `Generate` succeeds.
5. **Resume** if you drained a still-running instance instead of restarting:
   `ControlService.Resume`.

Keep the previous package file on the host to make rollback (§5) immediate.

> Linux/systemd: replace steps 2–3 with `apt/dpkg` (or tarball swap) and
> `systemctl restart lykuro-native-engine`.

---

## 5. Rollback

Fast path (previous package retained):

1. `Drain` (as above), wait for active == 0.
2. `sudo installer -pkg lykuro-native-engine-macos-metal-<previous>.pkg -target /`
3. `sudo launchctl kickstart -k system/ai.lykuro.native-engine`
4. Verify ready + canary `Generate`.

Model rollback is independent: repoint `/Library/Application Support/Lykuro/Models/current`
at the prior artifact and `UnloadModel` → `LoadModel` (or restart). Unsigned
artifacts are refused at load (fail-closed), so a bad swap fails safe rather
than serving corrupt weights.

Trigger rollback when: post-update `nie_requests_failed_total` rises, a
canary `Generate` fails, or `GetModelStatus` will not reach ready within the
restart budget.

---

## 6. Recovery

| Symptom | Likely cause | Action |
|---|---|---|
| Process flapping (launchd restarts) | crash loop | check `engine.err.log`; if config/cert error, fix and `kickstart`; `ThrottleInterval` already backs off the GPU |
| Startup fails, "unified memory budget" | model too large for device | smaller model / device with more unified memory; this is fail-closed pre-allocation |
| `nie_requests_rejected_total` rising | queue full or **soft watermark** load-shedding new sequences | reduce inbound rate or raise `max_queue`; if memory-driven, fewer concurrent sequences / larger device (addendum §10 staged watermarks) |
| Decode errors under memory pressure | **hard watermark** capping KV growth | shorten contexts or lower `max_sequences`; the engine refuses growth rather than risk OS eviction |
| `GetModelStatus` not ready | no model loaded / bad artifact | `LoadModel` with a signed artifact; confirm `artifact_path` and `trusted_signing_keys` |
| API refuses a caller | mTLS identity not authorized | Control API is Manager-only; Data API is Manager+Gateway. Check `control_identities` / `data_identities` and the client cert |
| GPU unhealthy on Metal | device lost / unavailable | restart the service; if it recurs, check the host GPU/driver and the certified-profile assumptions |
| Suspected leak over long uptime | — | `phys_footprint` is the true macOS signal (RSS overcounts shared Metal pages). The 24h soak validated a flat footprint; compare against baseline before acting |

### Emergency stop

`sudo launchctl bootout system/ai.lykuro.native-engine` (stops and disables
restart). Re-enable with `bootstrap`/`kickstart`. Prefer `Drain` first when
requests are in flight.

---

## 7. Pre-production checklist

- `mtls_required: true`, `allow_unsigned_dev: false`, `trusted_signing_keys`
  populated.
- Secrets present and readable only by `_lykuro`; certs in date.
- `metrics_enabled: true`, scrape target reachable, alerts wired (§3).
- Distributed package is **Phase 2** (Developer ID signed + notarized) for
  anything beyond internal test — see `deploy/macos/README.md`.
- Previous package retained on-host for one-command rollback.
