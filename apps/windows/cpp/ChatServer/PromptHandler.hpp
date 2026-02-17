// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#pragma once

#include <string>
#include <vector>
#include "json.hpp"

namespace AppUtils
{

class PromptHandler
{
  public:
    PromptHandler() = default;
    // Accepts a JSON array of messages (OpenAI format) and returns a formatted string with only start/end tokens as needed
    std::string FormatMessages(const nlohmann::json& messages);
};

} // namespace AppUtils
