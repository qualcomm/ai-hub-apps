// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#pragma once

#include <jni.h>
#include <string>

#include "GenieCommon.h"
#include "GenieDialog.h"
#include "PromptHandler.hpp"

namespace App
{
constexpr const char* c_exit_prompt = "exit";
constexpr const char* c_bot_name = "Qbot";

class GenieWrapper
{
  private:
    GenieDialogConfig_Handle_t m_config_handle = nullptr;
    GenieDialog_Handle_t m_dialog_handle = nullptr;
    GenieProfile_Handle_t m_profile_handle = nullptr;
    std::string m_user_name;
    AppUtils::PromptHandler prompt_handler;

  public:
    /**
     * GenieWrapper: Initializes GenieWrapper
     *    - Loads Model config with provided model path, htp config and tokenizer
     *    - Creates handle for Genie
     *
     * @param model_config_path: local path to model Genie config file
     * @param models_path: local path to directory that contains model context binaries (e.g., downloaded from AI Hub)
     * @param htp_config_path: local path to backend htp configuration
     * @param tokenizer_path: local path to tokenizer to use
     *
     * @thows on failure to create handle for Genie config, dialog
     *
     */
    GenieWrapper(const std::string& model_config_path,
                 const std::string& models_path,
                 const std::string& htp_config_path,
                 const std::string& tokenizer_path);
    GenieWrapper() = delete;
    GenieWrapper(const GenieWrapper&) = delete;
    GenieWrapper(GenieWrapper&&) = delete;
    GenieWrapper& operator=(const GenieWrapper&) = delete;
    GenieWrapper& operator=(GenieWrapper&&) = delete;
    ~GenieWrapper();

    /**
     * GetResponseForPrompt: Gets response from Genie for provided user prompt and callback
     *
     * @param user_prompt: User prompt provided by user
     * @param env: JNIEnv required to create intermediate output to pass via callback
     * @param callback: callback object
     * @param onNewStringMethod callback method to tunnel intermediate output string
     *
     * @throws on failure to query model response during chat
     *
     */
    std::string
    GetResponseForPrompt(const std::string& user_prompt, JNIEnv* env, jobject callback, jmethodID onNewStringMethod);
};
} // namespace App
