// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#include "HttpServer.hpp"
#include "PromptHandler.hpp"
#include <iostream>
#include <sstream>
#include <json.hpp>
// #define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"

using json = nlohmann::json;

namespace App {

HttpServer::HttpServer(const std::string& config, int port)
    : m_chat_server(config), m_port(port) {}

void HttpServer::Start() {
    httplib::Server svr;
    svr.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            auto body = json::parse(req.body);
            if (!body.contains("messages") || !body["messages"].is_array()) {
                res.status = 400;
                res.set_content(R"({"error":"Missing 'messages' array"})", "application/json");
                return;
            }
            std::string system_prompt, user_prompt;
            for (const auto& msg : body["messages"]) {
                if (msg["role"] == "system") system_prompt = msg["content"];
                if (msg["role"] == "user") user_prompt = msg["content"];
            }
            if (user_prompt.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"Missing user prompt"})", "application/json");
                return;
            }
            AppUtils::PromptHandler prompt_handler;
            std::string tagged_prompt = prompt_handler.FormatMessages(body["messages"]);
            std::string model_response;
            if (GENIE_STATUS_SUCCESS != GenieDialog_query(
                    m_chat_server.GetDialogHandle(), tagged_prompt.c_str(),
                    GenieDialog_SentenceCode_t::GENIE_DIALOG_SENTENCE_COMPLETE,
                    [](const char* response_back, const GenieDialog_SentenceCode_t, const void* user_data) {
                        std::string* resp = static_cast<std::string*>(const_cast<void*>(user_data));
                        resp->append(response_back);
                    },
                    &model_response)) {
                res.status = 500;
                res.set_content(R"({"error":"GenieDialog failed"})", "application/json");
                return;
            }
            json response = {
                {"id", "chatcmpl-xxx"},
                {"object", "chat.completion"},
                {"choices", {{{"index", 0}, {"message", {{"role", "assistant"}, {"content", model_response}}}, {"finish_reason", "stop"}}}},
                {"created", static_cast<int>(time(nullptr))},
                {"model", "genie"}
            };
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(std::string(R"({"error":"Exception: )") + e.what() + R"("})", "application/json");
        }
    });
    std::cout << "HTTP server listening on port " << m_port << std::endl;
    svr.listen("0.0.0.0", m_port);
}

} // namespace App
