// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#include "PromptHandler.hpp"
#include "ChatServer.hpp"
#include <vector>
#include "json.hpp"

using namespace AppUtils;

// Prompt formats: LLAMA3, LLAMA3_TAIDE, LLAMA2
#define LLAMA3

#ifdef LLAMA3
constexpr const std::string_view c_begin_system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n";
constexpr const std::string_view c_begin_user = "<|start_header_id|>user<|end_header_id|>\n\n";
constexpr const std::string_view c_end_user = "<|eot_id|>";
constexpr const std::string_view c_begin_assistant = "<|start_header_id|>assistant<|end_header_id|>\n\n";
constexpr const std::string_view c_end_assistant = "<|eot_id|>";
#endif

namespace AppUtils {

std::string PromptHandler::FormatMessages(const nlohmann::json& messages) {
    std::string result;
    bool first = true;
    for (const auto& msg : messages) {
        std::string role = msg["role"];
        std::string content = msg["content"];
        if (role == "system") {
            if (first) {
                result += c_begin_system.data();
                result += content;
                result += c_end_user.data();
                result += "\n\n";
                first = false;
            }
        } else if (role == "user") {
            result += c_begin_user.data();
            result += content;
            result += c_end_user.data();
            result += "\n\n";
        } else if (role == "assistant") {
            result += c_begin_assistant.data();
            result += content;
            result += c_end_assistant.data();
            result += "\n\n";
        }
    }
    // Remove trailing newlines
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) result.pop_back();
    return result;
}

} // namespace AppUtils
