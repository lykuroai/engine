#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/engine/error.h"
#include "model/tokenizer/bpe_tokenizer.h"

namespace lykuro::nie {

enum class Role { kSystem, kDeveloper, kUser, kAssistant, kTool };

struct ChatMessage {
    Role role;
    std::string content;
};

// Approved chat template "qwen_chat_v1" (ChatML):
//   <|im_start|>role\ncontent<|im_end|>\n ... <|im_start|>assistant\n
//
// Security (spec §13.3): message content is encoded as plain text via
// BpeTokenizer::Encode(), so special-token strings inside content can never
// become special IDs. Roles outside the template's supported set are
// rejected, never silently concatenated (spec §13.2).
class QwenChatTemplate {
public:
    static constexpr const char kTemplateId[] = "qwen_chat_v1";

    // Requires the tokenizer to declare <|im_start|> and <|im_end|>.
    static Status Validate(const BpeTokenizer& tokenizer);

    // Builds the full prompt token sequence ending with the assistant
    // header, ready for prefill.
    static Status BuildPrompt(const BpeTokenizer& tokenizer,
                              const std::vector<ChatMessage>& messages,
                              std::vector<uint32_t>& out);
};

// Checks input_tokens + max_output_tokens <= certified context limit
// (spec §13.4). The engine never truncates on overflow.
Status CheckTokenBudget(size_t input_tokens, uint32_t max_output_tokens,
                        uint32_t certified_context_limit);

}  // namespace lykuro::nie
