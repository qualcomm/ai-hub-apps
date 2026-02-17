// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#pragma once

#include <string>

#include "GenieCommon.h"
#include "GenieDialog.h"

namespace App
{
constexpr const std::string_view c_exit_prompt = "exit";

class ChatServer
{
  private:
    GenieDialogConfig_Handle_t m_config_handle = nullptr;
    GenieDialog_Handle_t m_dialog_handle = nullptr;
    std::string m_user_name;

  public:
    /**
     * ChatServer: Initializes ChatServer
     *    - Uses provided Genie configuration string
     *    - Creates handle for Genie
     *
     * @param config: JSON string containing Genie configuration
     *
     * @throws on failure to create handle for Genie config, dialog
     *
     */
    ChatServer(const std::string& config);
    ChatServer() = delete;
    ChatServer(const ChatServer&) = delete;
    ChatServer(ChatServer&&) = delete;
    ChatServer& operator=(const ChatServer&) = delete;
    ChatServer& operator=(ChatServer&&) = delete;
    ~ChatServer();

    /**
     * ChatWithUser: Starts Chat with user using previously loaded config
     *
     * @param user_name: User name to use during chat
     *
     * @throws on failure to query model response during chat
     *
     */
    void ChatLoop();
    GenieDialog_Handle_t GetDialogHandle() const;
};
} // namespace App
