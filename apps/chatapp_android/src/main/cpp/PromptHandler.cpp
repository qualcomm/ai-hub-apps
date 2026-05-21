// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#include "PromptHandler.hpp"
#include "json.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

using namespace AppUtils;

PromptHandler::PromptHandler(const std::string& models_path)
{
    std::filesystem::path metadata_path = std::filesystem::path(models_path) / "metadata.json";
    if (!std::filesystem::exists(metadata_path))
        throw std::runtime_error("metadata.json not found in model directory.");

    std::ifstream file(metadata_path);
    if (!file.is_open())
        throw std::runtime_error("Failed to open metadata.json: " + metadata_path.string());

    file.seekg(0, std::ios::end);
    std::string content(file.tellg(), '\0');
    file.seekg(0);
    if (!file.read(content.data(), content.size()))
        throw std::runtime_error("Failed to read metadata.json: " + metadata_path.string());

    using namespace nlohmann;
    try
    {
        json parsed = json::parse(content);
        const auto& tmpl = parsed.at("genie").at("chat_template");
        m_system_prefix = tmpl.at("system_prefix").get<std::string>();
        m_system_suffix = tmpl.at("system_suffix").get<std::string>();
        m_user_prefix = tmpl.at("user_prefix").get<std::string>();
        m_user_suffix = tmpl.at("user_suffix").get<std::string>();
        m_assistant_prefix = tmpl.at("assistant_prefix").get<std::string>();
        m_default_system_prompt = tmpl.at("default_system_prompt").get<std::string>();
    }
    catch (const json::exception& e)
    {
        throw std::runtime_error(std::string("metadata.json has invalid schema: ") + e.what());
    }
}

std::string PromptHandler::GetPromptWithTag(const std::string& user_prompt)
{
    std::string prompt;
    if (m_is_first_prompt)
    {
        m_is_first_prompt = false;
        prompt += m_system_prefix;
        prompt += m_default_system_prompt;
        prompt += m_system_suffix;
    }
    prompt += m_user_prefix;
    prompt += user_prompt;
    prompt += m_user_suffix;
    prompt += m_assistant_prefix;
    return prompt;
}
