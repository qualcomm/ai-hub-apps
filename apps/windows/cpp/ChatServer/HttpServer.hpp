// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#pragma once

#include <string>
#include "ChatServer.hpp"

namespace App {
class HttpServer {
public:
    HttpServer(const std::string& config, int port = 8080);
    void Start();
private:
    ChatServer m_chat_server;
    int m_port;
};
} // namespace App
