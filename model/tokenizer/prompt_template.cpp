#include "model/tokenizer/prompt_template.h"

namespace lykuro::nie {

namespace {

constexpr const char kComponent[] = "prompt_template";
constexpr const char kImStart[] = "<|im_start|>";
constexpr const char kImEnd[] = "<|im_end|>";

// Roles the qwen_chat_v1 template supports. developer/tool are part of the
// normalized message model but not of this template version; they are
// rejected explicitly.
const char* SupportedRoleName(Role role) {
    switch (role) {
        case Role::kSystem: return "system";
        case Role::kUser: return "user";
        case Role::kAssistant: return "assistant";
        case Role::kDeveloper:
        case Role::kTool:
            return nullptr;
    }
    return nullptr;
}

}  // namespace

Status QwenChatTemplate::Validate(const BpeTokenizer& tokenizer) {
    if (!tokenizer.SpecialTokenId(kImStart) ||
        !tokenizer.SpecialTokenId(kImEnd)) {
        return Status(ErrorCode::kArtifactVerificationFailed,
                      "tokenizer does not declare chat template special tokens",
                      kComponent);
    }
    return Status::Ok();
}

Status QwenChatTemplate::BuildPrompt(const BpeTokenizer& tokenizer,
                                     const std::vector<ChatMessage>& messages,
                                     std::vector<uint32_t>& out) {
    Status valid = Validate(tokenizer);
    if (!valid.ok()) return valid;

    if (messages.empty()) {
        return Status(ErrorCode::kInvalidRequest, "messages must not be empty",
                      kComponent);
    }

    const uint32_t im_start = *tokenizer.SpecialTokenId(kImStart);
    const uint32_t im_end = *tokenizer.SpecialTokenId(kImEnd);

    for (const ChatMessage& msg : messages) {
        const char* role_name = SupportedRoleName(msg.role);
        if (role_name == nullptr) {
            return Status(ErrorCode::kInvalidRequest,
                          "message role is not supported by this template",
                          kComponent);
        }
        out.push_back(im_start);
        Status s = tokenizer.Encode(std::string(role_name) + "\n", out);
        if (!s.ok()) return s;
        s = tokenizer.Encode(msg.content, out);
        if (!s.ok()) return s;
        out.push_back(im_end);
        s = tokenizer.Encode("\n", out);
        if (!s.ok()) return s;
    }

    out.push_back(im_start);
    return tokenizer.Encode("assistant\n", out);
}

Status CheckTokenBudget(size_t input_tokens, uint32_t max_output_tokens,
                        uint32_t certified_context_limit) {
    if (max_output_tokens == 0) {
        return Status(ErrorCode::kInvalidRequest,
                      "max_output_tokens must be at least 1", "admission");
    }
    if (input_tokens + max_output_tokens > certified_context_limit) {
        return Status(ErrorCode::kContextLengthExceeded,
                      "input and requested output exceed the certified "
                      "context limit",
                      "admission")
            .WithDetail("limit_tokens", certified_context_limit)
            .WithDetail("input_tokens", int64_t(input_tokens))
            .WithDetail("max_output_tokens", int64_t(max_output_tokens));
    }
    return Status::Ok();
}

}  // namespace lykuro::nie
