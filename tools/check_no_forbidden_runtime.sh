#!/usr/bin/env bash
# Fails if any forbidden third-party inference runtime appears in the
# source tree or the SBOM (spec §0.2, AT-01 / AT-M03). Run in CI and
# before packaging.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Whole-word, case-insensitive. mlx-lm (the server) is forbidden; plain
# "MLX C++" as an approved compute dep would need an explicit allow, but
# it is not used here.
FORBIDDEN='ollama|llama\.cpp|llama_cpp|vllm|text-generation-inference|\btgi\b|mlx-lm|mlx_lm'

# Search source and build files, excluding this checker, the git dir,
# build outputs, and docs that legitimately name the ban list.
HITS=$(grep -rInE "$FORBIDDEN" \
    --include='*.c' --include='*.cc' --include='*.cpp' --include='*.cu' \
    --include='*.mm' --include='*.h' --include='*.hpp' \
    --include='CMakeLists.txt' --include='*.cmake' \
    . 2>/dev/null \
    | grep -v 'tools/check_no_forbidden_runtime.sh' || true)

if [[ -n "$HITS" ]]; then
    echo "FORBIDDEN inference runtime referenced in build sources:" >&2
    echo "$HITS" >&2
    exit 1
fi

# The SBOM must not DECLARE any forbidden package. Only "name"/
# "licenseDeclared"-style value lines count — the document comment that
# states the guarantee legitimately lists the ban terms.
if grep -iE '"(name|packageName|licenseDeclared|licenseConcluded)"[[:space:]]*:' \
       sbom/*.json 2>/dev/null | grep -iE "$FORBIDDEN" >/dev/null 2>&1; then
    echo "FORBIDDEN runtime declared as an SBOM package" >&2
    exit 1
fi

echo "OK: no forbidden inference runtime in sources or SBOM"
