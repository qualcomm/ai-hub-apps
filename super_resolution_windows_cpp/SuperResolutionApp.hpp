// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <optional>

#include <onnxruntime_cxx_api.h>

namespace App
{

enum Precision
{
    Fp32 = 0,
    Fp16 = 1
};

enum BackendOption
{
    Cpu = 0,
    Npu = 2,
};

class SuperResolutionApp
{
  private:
    // model io pointing to loaded data
    std::vector<Ort::Value> m_inputs;
    std::vector<Ort::Value> m_outputs;

    std::vector<const char*> m_input_names;
    // Keep io names alive returned by session
    std::vector<Ort::AllocatedStringPtr> m_io_data_ptr;

    // model metadata
    uint32_t m_model_input_ht, m_model_input_wt, m_model_scale;
    std::string m_model_path;

    Ort::Env m_env;
    std::unique_ptr<Ort::Session> m_session;
    Ort::AllocatorWithDefaultOptions m_allocator;

    void ClearInputsAndOutputs();

  public:
    SuperResolutionApp(std::string model_path, uint32_t model_input_ht, uint32_t model_input_wt, uint32_t model_Scale);
    SuperResolutionApp() = delete;
    SuperResolutionApp(const SuperResolutionApp&) = delete;
    SuperResolutionApp(SuperResolutionApp&&) = delete;
    SuperResolutionApp& operator=(const SuperResolutionApp&) = delete;
    SuperResolutionApp& operator=(SuperResolutionApp&&) = delete;

    /**
     * PrepareModelForInference: Prepares model for inference with ORT
     *   - initializes ORT Session
     *   - sets backend, precision and qnn_options for execution provider
     *
     * @param backend: backend for model execution.
     * @param precision: preferred precision for model execution.
     * @param qnn_options: QNN Execution provider options.
     *   refer to
     * https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html#configuration-options
     *   for all supported options.
     *
     */
    void PrepareModelForInference(const App::BackendOption backend,
                                  const App::Precision precision,
                                  std::unordered_map<std::string, std::string> qnn_options);

    /**
     * LoadInputs: Load and prepare local image for model inference
     *   - Loads image with opencv
     *   - Converts image to Ort::OrtTensor to use during model inference
     *
     * @param image_path: local path to image to load.
     *
     */
    void LoadInputs(const std::string& image_path);

    /**
     * RunInference: Execute prepared model
     *   - Runs inference and cache output
     *
     * @throws if model is not prepared with PrepareModelForInference before
     * @throws if inputs are not loaded with LoadInputs before
     *
     */
    void RunInference();

    /**
     * ProcessOutput: Process cached model output to show results
     *
     * @param input_image_path input image to add bounding boxes and labels.
     * @param display_output_image if true, shows output image in window using
     * opencv.
     * @param output_image_path if provided, serializes output image locally.
     *
     */
    void ProcessOutput(const std::string& input_image_path,
                       std::optional<std::string> output_image_path = std::nullopt,
                       bool display_output_image = true);
};
} // namespace App
