#pragma once

#include <string>

namespace lykuro::nie::testfixture {

// Builds a minimal approved_qwen_tokenizer_v1 config:
//  - ids 0..255: the byte-level alphabet (GPT-2 byte-to-unicode order)
//  - ids 260..263: merged tokens "he", "ll", "hell", "hello"
//  - specials: <|im_start|>=300, <|im_end|>=301, <|endoftext|>=302
std::string SmallTokenizerConfig();

}  // namespace lykuro::nie::testfixture
