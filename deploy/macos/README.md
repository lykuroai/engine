# macOS Native Package (scaffolding)

Deliverable layout and operational docs for the Apple Silicon build of the
Lykuro Native Inference Engine (LYK-NIE-ADD-METAL-001 §23–§25).

**Status: scaffolding.** The signed/notarized `.pkg` is NOT produced here.
Code signing and notarization require a Developer ID certificate and an
Apple notarization account, neither of which is available in the current
build environment (see the repo README's Metal section). This directory
holds the environment-independent pieces: the launchd definition, the
example config, the package layout, and the install/security docs. The
CI packaging job (§30.1: `codesign` → `notarytool` → `stapler`) is the
remaining, credential-gated step.

## Package layout (§23.1)

```
lykuro-native-engine-macos-arm64/
├── bin/lykuro-native-engine            # arm64 executable (Hardened Runtime)
├── lib/lykuro.metallib                 # precompiled shaders (needs Xcode CI)
├── config/engine.example.yaml
├── launchd/ai.lykuro.native-engine.plist
├── contracts/                          # api/proto/lykuro/nie/v1
├── compatibility/certified-profiles.json
├── checksums.sha256
├── manifest.json
├── signature.sig
├── sbom/  licenses/  docs/
```

Never bundled (§23.2): Ollama / llama.cpp / vLLM / TGI / mlx-lm server,
Python, Node.js, Homebrew, Xcode, model weights, unsigned dylibs, or any
development signing key.

## Install locations (§23.3)

System service:
- `/Library/Application Support/Lykuro/NativeEngine/`
- `/Library/LaunchDaemons/ai.lykuro.native-engine.plist`
- `/Library/Logs/Lykuro/NativeEngine/`

The engine runs host-native as the dedicated non-root `_lykuro` service
user. When the Gateway/Platform runs in a container, it reaches the engine
over a host-only mTLS connection — the engine port is never exposed to the
public LAN (§23.4).

## Not yet implemented (credential- or toolchain-gated)

- `codesign` / notarization / stapling and Gatekeeper verification (§25)
- `lykuro.metallib` precompile (needs Xcode's `metal` compiler; the MVP
  runs MPSGraph-only, so no custom metallib is required yet)
- YAML config parsing (loader is JSON today; this YAML documents the shape)
- installer precheck/uninstall flows (§24)
