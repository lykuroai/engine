#!/usr/bin/env python3
"""Development-only reference oracle generator (spec §30.2).

Runs the HF transformers implementation in FP32 (matching the engine's
compute precision for BF16 checkpoints) and records, per prompt:
  - the prompt token ids (HF tokenizer, plain encode)
  - a snapshot of the last-position logits after prefill
  - 32 greedy continuation token ids

Usage: gen_reference.py <hf_model_dir> <out_json>
"""

import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The capital of France is",
    "1+1=",
    "むかしむかし、あるところに",
]


def main() -> int:
    model_dir, out_path = sys.argv[1], sys.argv[2]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32)
    model.eval()

    cases = []
    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            logits = model(ids).logits[0, -1]
            generated = model.generate(
                ids, max_new_tokens=32, do_sample=False,
                eos_token_id=None, pad_token_id=0)
        cases.append({
            "prompt": prompt,
            "prompt_ids": ids[0].tolist(),
            "logits_head64": [float(v) for v in logits[:64]],
            "logits_argmax": int(torch.argmax(logits)),
            "logits_max": float(torch.max(logits)),
            "greedy_ids": generated[0, ids.shape[1]:].tolist(),
        })

    with open(out_path, "w") as f:
        json.dump({"model": "qwen2.5-0.5b-instruct-fp32", "cases": cases}, f)
    print(f"wrote {out_path} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
