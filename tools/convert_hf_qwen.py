#!/usr/bin/env python3
"""Development-only converter: HF Qwen2.5 checkpoint -> Lykuro artifact.

Part of the certified-model preparation pipeline (spec §10). Runs only in
the development environment (production packages carry no Python).

Usage: convert_hf_qwen.py <hf_model_dir> <artifact_out_dir>
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_tokenizer(hf_dir: Path, out_path: Path) -> None:
    tok = json.loads((hf_dir / "tokenizer.json").read_text())
    model = tok["model"]
    assert model["type"] == "BPE", "expected byte-level BPE"

    merges = model["merges"]
    if merges and not isinstance(merges[0], str):
        merges = [f"{a} {b}" for a, b in merges]

    specials = {}
    for added in tok.get("added_tokens", []):
        if added.get("special", False):
            specials[added["content"]] = added["id"]

    out = {
        "tokenizer_type": "approved_qwen_tokenizer_v1",
        "vocab": model["vocab"],
        "merges": merges,
        "special_tokens": specials,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False))


def main() -> int:
    hf_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    config = json.loads((hf_dir / "config.json").read_text())
    assert config["model_type"] == "qwen2", config["model_type"]

    (out_dir / "config").mkdir(parents=True, exist_ok=True)
    (out_dir / "weights").mkdir(parents=True, exist_ok=True)

    shutil.copyfile(hf_dir / "model.safetensors",
                    out_dir / "weights" / "model.safetensors")
    convert_tokenizer(hf_dir, out_dir / "config" / "tokenizer.json")

    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    eos_ids = config.get("eos_token_id", [])
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    gen_cfg_path = hf_dir / "generation_config.json"
    if gen_cfg_path.exists():
        gen_eos = json.loads(gen_cfg_path.read_text()).get("eos_token_id", [])
        if isinstance(gen_eos, int):
            gen_eos = [gen_eos]
        eos_ids = sorted(set(eos_ids) | set(gen_eos))

    files = []
    for rel in ("weights/model.safetensors", "config/tokenizer.json"):
        p = out_dir / rel
        files.append({
            "path": rel,
            "sha256": sha256_file(p),
            "size_bytes": p.stat().st_size,
        })

    manifest = {
        "schema_version": "1",
        "artifact_id": "ma_qwen25_05b_instruct",
        "model_family": "qwen",
        "architecture": "approved_qwen_decoder_v1",
        "model_version": "qwen2.5-0.5b-instruct",
        "weight_format": "safetensors",
        "precision": "bf16",
        "vocab_size": config["vocab_size"],
        "hidden_size": hidden,
        "num_layers": config["num_hidden_layers"],
        "num_attention_heads": heads,
        "num_key_value_heads": config["num_key_value_heads"],
        "head_dim": hidden // heads,
        "max_context_tokens": config["max_position_embeddings"],
        "intermediate_size": config["intermediate_size"],
        "rms_norm_eps": config["rms_norm_eps"],
        "rope_theta": config["rope_theta"],
        "tie_word_embeddings": config.get("tie_word_embeddings", False),
        "eos_token_ids": eos_ids,
        "tokenizer_type": "approved_qwen_tokenizer_v1",
        "chat_template_id": "qwen_chat_v1",
        "files": files,
        "license_review_id": "lic_dev_apache20_qwen25",
        "certified_profiles": [],
        "created_at": "2026-08-07T00:00:00Z",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"artifact written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
