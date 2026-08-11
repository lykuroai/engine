#pragma once

#include <string>

#include "core/engine/error.h"

namespace lykuro::nie {

// Converts an on-disk HuggingFace Qwen2/2.5 checkpoint directory into a
// Lykuro model artifact directory (manifest.json + weights/model.safetensors
// + config/tokenizer.json). Native C++ port of tools/convert_hf_qwen.py so
// production hosts need no Python (spec §0.2/§10).
//
// hf_dir must contain config.json, tokenizer.json, model.safetensors, and
// optionally generation_config.json. Only model_type "qwen2" is accepted.
Status ConvertHfQwen(const std::string& hf_dir, const std::string& out_dir);

}  // namespace lykuro::nie
