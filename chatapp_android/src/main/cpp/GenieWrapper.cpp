// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------

#include <android/log.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <jni.h>
#include <memory>

#include "GenieCommon.h"
#include "GenieDialog.h"
#include "GenieWrapper.hpp"
#include "PromptHandler.hpp"
#include "json.hpp"

using namespace App;

namespace
{

/**
 * user_data_and_callback: Hold data required for callback into java methods
 *  - JNIEnv, jobject, jmethodID required for callback
 *  - data to capture response string from Genie
 */
struct user_data_and_callback
{
    // to creates string for
    JNIEnv* env;
    jobject callback;
    jmethodID on_new_string_method;
    std::string data;
};

//
// GenieCallBack - Callback to handle response from Genie
//   - Captures response from Genie into user_data
//   - Print response to stdout
//   - Add ChatSplit upon sentence completion
//
/**
 * GenieCallBack: Callback to handle response from Genie
 *
 * @param response_back char pointer to response token
 * @param sentence_code hint referring to response_back type
 * @param user_data void* to user_data tunneled into callback
 */
void GenieCallBack(const char* response_back, const GenieDialog_SentenceCode_t sentence_code, const void* user_data)
{
    auto user_data_struct = static_cast<struct user_data_and_callback*>(const_cast<void*>(user_data));
    user_data_struct->data.append(response_back);
    user_data_struct->env->CallVoidMethod(user_data_struct->callback, user_data_struct->on_new_string_method,
                                          user_data_struct->env->NewStringUTF(response_back));
}

//
// LoadModelConfig - Loads model config file
//  - Loads config file in memory
//  - Replaces placeholders with user provided values
//
/**
 * LoadModelConfig: Loads genie config and loads
 * @param model_config_path
 * @param models_path
 * @param htp_model_config_path
 * @param tokenizer_path
 * @return
 */
std::string LoadModelConfig(const std::string& model_config_path,
                            const std::string& models_path,
                            const std::string& htp_model_config_path,
                            const std::string& tokenizer_path)
{
    if (!std::filesystem::exists(model_config_path))
    {
        __android_log_print(ANDROID_LOG_INFO, "ChatApp", "Genie config file not found.");
        throw std::runtime_error("Genie config file not found.");
    }

    std::string content;
    std::getline(std::ifstream(model_config_path), content, '\0');

    using namespace nlohmann;
    json config = json::parse(content);

    config["dialog"]["tokenizer"]["path"] = tokenizer_path;
    config["dialog"]["engine"]["backend"]["extensions"] = htp_model_config_path;

    for (auto& bin : config["dialog"]["engine"]["model"]["binary"]["ctx-bins"])
        bin = models_path + "/" + bin.get<std::string>();

    return config.dump();
}

} // namespace

GenieWrapper::GenieWrapper(const std::string& model_config_path,
                           const std::string& models_path,
                           const std::string& htp_config_path,
                           const std::string& tokenizer_path)
    : prompt_handler(models_path)
{
    // Load model config in-memory
    std::string config = LoadModelConfig(model_config_path, models_path, htp_config_path, tokenizer_path);

    // Create Genie config
    if (GENIE_STATUS_SUCCESS != GenieDialogConfig_createFromJson(config.c_str(), &m_config_handle))
    {
        __android_log_print(ANDROID_LOG_ERROR, "ChatApp", "Failed to create Genie config.");
        throw std::runtime_error("Failed to create the Genie Dialog config. Please check config file.");
    }

    // Create Genie profile handle
    if (GENIE_STATUS_SUCCESS != GenieProfile_create(nullptr, &m_profile_handle))
    {
        __android_log_print(ANDROID_LOG_ERROR, "ChatApp", "Failed to create Genie profile handler.");
        throw std::runtime_error("Failed to create the Genie profile handler.");
    }

    // Bind Profile Handle to Dialog Config
    if (GENIE_STATUS_SUCCESS != GenieDialogConfig_bindProfiler(m_config_handle, m_profile_handle))
    {
        __android_log_print(ANDROID_LOG_ERROR, "ChatApp", "Failed to bind Genie profiler tot dialog config.");
        throw std::runtime_error("Failed to bind the Genie Profiler to Dialog config.");
    }

    // Create Genie dialog handle
    if (GENIE_STATUS_SUCCESS != GenieDialog_create(m_config_handle, &m_dialog_handle))
    {
        __android_log_print(ANDROID_LOG_ERROR, "ChatApp", "Failed to create Genie dialog.");
        throw std::runtime_error("Failed to create the Genie Dialog.");
    }
}

GenieWrapper::~GenieWrapper()
{
    if (m_config_handle != nullptr)
    {
        if (GENIE_STATUS_SUCCESS != GenieDialogConfig_free(m_config_handle))
        {
            __android_log_print(ANDROID_LOG_ERROR, "ChatApp", "Failed to free Genie config handler.");
            std::cerr << "Failed to free the Genie config handler.";
        }
    }

    if (m_dialog_handle != nullptr)
    {
        if (GENIE_STATUS_SUCCESS != GenieDialog_free(m_dialog_handle))
        {
            __android_log_print(ANDROID_LOG_ERROR, "ChatApp", "Failed to free Genie dialog handler.");
            std::cerr << "Failed to free the Genie dialog handler.";
        }
    }

    if (m_profile_handle != nullptr)
    {
        if (GENIE_STATUS_SUCCESS != GenieProfile_free(m_profile_handle))
        {
            __android_log_print(ANDROID_LOG_ERROR, "ChatApp", "Failed to free Genie profile handler.");
            std::cerr << "Failed to free the Genie profile handler.";
        }
    }
}

std::string GenieWrapper::GetResponseForPrompt(const std::string& user_prompt,
                                               JNIEnv* env,
                                               jobject callback,
                                               jmethodID onNewStringMethod)
{

    std::string model_response;
    struct user_data_and_callback user_data
    {
        .env = env, .callback = callback, .on_new_string_method = onNewStringMethod, .data = model_response
    };

    std::string tagged_prompt = prompt_handler.GetPromptWithTag(user_prompt);
    // Get response from Genie
    if (GENIE_STATUS_SUCCESS != GenieDialog_query(m_dialog_handle, tagged_prompt.c_str(),
                                                  GenieDialog_SentenceCode_t::GENIE_DIALOG_SENTENCE_COMPLETE,
                                                  GenieCallBack, &user_data))
    {
        __android_log_print(ANDROID_LOG_ERROR, "ChatApp", "Failed to get response from bot.");
    }

    if (user_data.data.empty())
    {
        // If model response is empty, reset dialog to re-initiate dialog.
        // During local testing, we found that in certain cases,
        // model response bails out after few iterations during chat.
        // If that happens, just reset Dialog handle to continue the chat.
        if (GENIE_STATUS_SUCCESS != GenieDialog_reset(m_dialog_handle))
        {
            __android_log_print(ANDROID_LOG_ERROR, "ChatApp", "Failed to reset GenieDialog.");
            throw std::runtime_error("Failed to reset Genie Dialog.");
        }
        if (GENIE_STATUS_SUCCESS != GenieDialog_query(m_dialog_handle, user_prompt.c_str(),
                                                      GenieDialog_SentenceCode_t::GENIE_DIALOG_SENTENCE_COMPLETE,
                                                      GenieCallBack, &user_data))
        {
            __android_log_print(ANDROID_LOG_INFO, "ChatApp", "Error getting response from Genie.");
            throw std::runtime_error("Failed to get response from GenieDialog. Please restart Chat.");
        }
    }

    char* rawJson = nullptr;
    const Genie_AllocCallback_t alloc_callback([](size_t size, const char** out)
                                               { *out = static_cast<char*>(std::malloc(size)); });

    auto profileStatus = GenieProfile_getJsonData(m_profile_handle, alloc_callback, (const char**)&rawJson);
    std::unique_ptr<char, decltype(&std::free)> jsonData(rawJson, &std::free);

    if (GENIE_STATUS_SUCCESS != profileStatus)
    {
        __android_log_print(ANDROID_LOG_WARN, "ChatApp", "Failed to get profiler data from Genie.");
        return user_data.data;
    }

    if (jsonData != nullptr)
    {
        using namespace nlohmann;

        json j = json::parse(jsonData.get());

        for (const auto& component : j["components"])
        {

            for (const auto& event : component["events"])
            {
                auto duration = event["duration"].get<int>();

                // Optional metrics — check before accessing
                if (event.contains("time-to-first-token"))
                {
                    auto ttft = event["time-to-first-token"]["value"].get<int>();
                    auto unit = event["time-to-first-token"]["unit"].get<std::string>();
                    __android_log_print(ANDROID_LOG_INFO, "ChatApp", "TTFT: %d %s", ttft, unit.c_str());
                }

                if (event.contains("token-generation-rate"))
                {
                    auto tps = event["token-generation-rate"]["value"].get<int>();
                    auto unit = event["token-generation-rate"]["unit"].get<std::string>();
                    __android_log_print(ANDROID_LOG_INFO, "ChatApp", "TPS: %d %s", tps, unit.c_str());
                }
            }
        }
    }
    return user_data.data;
}
