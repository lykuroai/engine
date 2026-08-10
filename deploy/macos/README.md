# macOS Native Package (scaffolding)

Deliverable layout and operational docs for the Apple Silicon build of the
Lykuro Native Inference Engine (LYK-NIE-ADD-METAL-001 §23–§25).

**Status: pipeline ready, credential-gated.** The full sign → notarize →
staple pipeline is implemented in `sign_and_notarize.sh` and wired into
`tools/make_package.sh`; it self-skips (no-op, exit 0) until a Developer ID
certificate is present, so unsigned dev/CI builds are unaffected. The only
thing missing is the certificate + Apple notarization account. This
directory also holds the environment-independent pieces: the launchd
definition, the example config, the package layout, and the install docs.

## Producing a signed + notarized package (when the Developer ID is ready)

One-time: create the Developer ID Application + Installer certs in the
Apple Developer account, import them into the login keychain, and store a
notary credential profile:

```
xcrun notarytool store-credentials lykuro-notary \
  --apple-id "<apple-id>" --team-id "<TEAMID>" --password "<app-specific-password>"
```

Then build and sign:

```
# 1. Stage + codesign binaries (Hardened Runtime) + checksums + Ed25519 manifest
export DEVELOPER_ID_APP="Developer ID Application: e-Business Solutions Inc. (TEAMID)"
./tools/make_package.sh build/metal macos-metal dist

# 2. Build + installer-sign + notarize + staple + Gatekeeper-verify the .pkg
export DEVELOPER_ID_INSTALLER="Developer ID Installer: e-Business Solutions Inc. (TEAMID)"
export NOTARY_PROFILE=lykuro-notary
./deploy/macos/sign_and_notarize.sh dist/lykuro-native-engine-macos-metal-<version>
```

Credentials are read only from the environment — no secret is ever written
to the repo. `sign_and_notarize.sh` accepts either `NOTARY_PROFILE` or the
explicit `NOTARY_APPLE_ID` / `NOTARY_TEAM_ID` / `NOTARY_PASSWORD` trio.
Step 2 ends with `xcrun stapler validate` and `spctl --assess --type
install`, so a green run is the Gatekeeper acceptance evidence for §25/§34.

### Hardened Runtime entitlements

`entitlements.plist` is intentionally an empty dict, signed with
`--options runtime`. The engine is self-contained (Apple system frameworks
+ statically-linked first-party libs); it loads no unsigned dylibs, uses no
JIT, and reads no third-party plug-ins, so it needs none of the Hardened
Runtime exceptions (`allow-jit`, `disable-library-validation`,
`allow-unsigned-executable-memory`, `allow-dyld-environment-variables`).
The AMFI entitlements parser rejects XML comments, so the rationale lives
here rather than in the plist. If a future build adds a loadable module or
JIT path, add the specific entitlement and record why.

Validated so far without a certificate: the `codesign` invocation form
(entitlements + `--options runtime` → `flags=…,runtime`, strict verify
passes, using an ad-hoc identity), the `--codesign-only` and no-op paths,
and plist/script syntax. The `--timestamp`, `productsign`, `notarytool`,
and `stapler` steps require the real Developer ID and run at step 2.

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

- Running the sign/notarize pipeline for real (needs the Developer ID cert
  + notary account; the pipeline itself is implemented and self-skipping)
- `lykuro.metallib` precompile (needs Xcode's `metal` compiler; the MVP
  runs MPSGraph-only, so no custom metallib is required yet)
- YAML config parsing (loader is JSON today; this YAML documents the shape)
- installer precheck/uninstall flows (§24)
