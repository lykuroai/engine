# Release Runbook — Lykuro Native Inference Engine

How to cut a versioned release and publish per-platform binaries. Grounded
in the v1.0.0 release process; follow it exactly so provenance, the
single-binary guarantee, and integrity all hold.

## 0. Golden rules

- **Build only from a clean checkout at the tag.** Never package from a
  working tree with local edits or a foreign `.git`. `make_package.sh`
  refuses a dirty tree and stamps `git_revision` into the provenance
  manifest — a wrong revision is a corrupt release.
  - ⚠️ Known trap: the CUDA host's `~/native-inference-engine` is an
    unrelated checkout (`doragogonet/doragogo`) with the engine source
    rsync'd over it (the rsync helper excludes `.git`). Do **not** build
    there. Build in a fresh dir synced *with* `.git` (§2).
- **The self-contained gate must pass.** `make_package.sh` runs
  `tools/check_selfcontained.sh` on every staged binary; a forbidden
  dynamic dependency fails the package. Do not bypass it.
- **Signed public binaries only.** macOS distribution binaries go out only
  after Developer ID signing + notarization (Phase 2). Ad-hoc `--dev`
  packages are internal-only and must never be attached to a public
  release.

## 1. Prerequisites (once per build host)

- Static gRPC toolchain: `third_party/build_grpc_static.sh ~/.local/grpc-static`
  (macOS) or `~/.local/grpc` (Linux CUDA host, already present).
- macOS: `release-static` preset builds the single static binary.
- Linux: CUDA toolkit on `PATH` (`export PATH=/usr/local/cuda/bin:$PATH`).

## 2. Sync a clean source tree to the build host

Local repo must be clean and at the release commit. To the Linux host:

```
sshpass -f .rpw rsync -az --delete \
  -e "ssh -o PubkeyAuthentication=no -o PreferredAuthentications=password" \
  --exclude build/ --exclude dist/ \
  /path/to/native-inference-engine/ kaku@10.8.1.18:~/lykuro-engine-rel/
# NOTE: include .git (do NOT use the repo's rsync-repo helper, which excludes it)
```

Verify on the host: `git remote get-url origin` is `lykuroai/engine`,
`git describe --tags` is the release tag, `git status --porcelain` empty.

## 3. Build + package per platform

**Linux (x86_64, CUDA):**
```
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build/release-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLYKURO_ENABLE_CUDA=ON -DLYKURO_ENABLE_GRPC=ON -DLYKURO_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_PREFIX_PATH=$HOME/.local/grpc
cmake --build build/release-cuda -j"$(nproc)"
./tools/make_package.sh build/release-cuda linux-cuda dist
```

**macOS (Apple Silicon, Metal) — Phase 2, needs Developer ID:**
```
cmake --preset release-static && cmake --build build/release-static -j
export DEVELOPER_ID_APP="Developer ID Application: ... (TEAMID)"
./tools/make_package.sh build/release-static macos-metal dist       # signs binaries
export DEVELOPER_ID_INSTALLER="Developer ID Installer: ... (TEAMID)"
export NOTARY_PROFILE=lykuro-notary
./deploy/macos/sign_and_notarize.sh dist/lykuro-native-engine-macos-metal-<ver>
```

`make_package.sh` runs the forbidden-runtime and self-contained gates and
emits `checksums.sha256` + provenance `manifest.json`. Confirm it prints
`self-contained: OK` and `package: …tar.gz`.

## 4. Verify the artifact

- `shasum -a 256 dist/*.tar.gz` — record it; it goes in the release body.
- Spot-check linkage: `otool -L` (macOS) / `ldd` (Linux) show only allowed
  deps (system libs + Apple frameworks; or C/C++ runtime + system OpenSSL +
  CUDA).
- Pull the artifact to the release host and re-hash — the SHA-256 must
  match end to end (guards against transfer corruption).

## 5. Tag + publish

```
git tag -a vX.Y.Z -m "…"; git push origin vX.Y.Z
gh release create vX.Y.Z -R lykuroai/engine --prerelease \
  --title "vX.Y.Z — …" --notes-file docs/RELEASE_NOTES.md
gh release upload vX.Y.Z dist/<artifact>.tar.gz -R lykuroai/engine
```

Append a **Downloads** section to the release body with each artifact's
SHA-256 and its runtime requirements. Drop `--prerelease` only once signed
macOS binaries are attached and the certified profile is production-issued.

- Moving a tag: acceptable only for a pre-release with no consumed assets
  (`git tag -f` + `git push -f`). Never move a tag people have built against
  — cut a new patch version instead.

## 6. Phase 2 close-out (when Developer ID is available)

1. Build + sign + notarize the macOS package (§3).
2. `gh release upload vX.Y.Z <macos>.pkg` and add its SHA-256 to the body.
3. Issue the production certified profile from the signed build.
4. Drop `--prerelease`; announce on downloads.lykuro.ai.
