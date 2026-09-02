// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#pragma once

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

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

class ObjectDetectionApp
{
  private:
    // model io pointing to loaded data
    std::vector<Ort::Value> m_inputs;
    std::vector<Ort::Value> m_outputs;

    std::vector<const char*> m_input_names;
    // Keep io names alive returned by session
    std::vector<Ort::AllocatedStringPtr> m_io_data_ptr;

    // model metadata
    uint32_t m_model_input_ht, m_model_input_wt;
    std::string m_model_path;
    std::string m_labels_path;

    // Class labels loaded from labels_path, ordered by class index.
    std::vector<std::string> m_labels;

    Ort::Env m_env;
    std::unique_ptr<Ort::Session> m_session;
    Ort::AllocatorWithDefaultOptions m_allocator;

    void ClearInputsAndOutputs();

    /**
     * LoadLabels: Load class labels from m_labels_path (one label per line,
     * ordered by class index).
     *
     * @throws if the labels file can't be opened or is empty.
     */
    void LoadLabels();

    /**
     * GetClassLabel: Return the label for a class index.
     *
     * @param class_index model-predicted class index.
     * @throws if class_index is out of range for the loaded labels.
     */
    std::string GetClassLabel(uint32_t class_index) const;

  public:
    ObjectDetectionApp(std::string model_path,
                       std::string labels_path,
                       uint32_t model_input_ht,
                       uint32_t model_input_wt);
    ObjectDetectionApp() = delete;
    ObjectDetectionApp(const ObjectDetectionApp&) = delete;
    ObjectDetectionApp(ObjectDetectionApp&&) = delete;
    ObjectDetectionApp& operator=(const ObjectDetectionApp&) = delete;
    ObjectDetectionApp& operator=(ObjectDetectionApp&&) = delete;

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
