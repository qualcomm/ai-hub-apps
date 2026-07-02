// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#include "SuperResolutionApp.hpp"
#include "Utilities.hpp"

#include <filesystem>
#include <iostream>
#include <sstream>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>
#include <sstream>
#include <unordered_map>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace App;

namespace
{
std::string GetBackendDllFromOption(BackendOption backend_opt)
{
    // Convert backend_opt into respective dll to use
    switch (backend_opt)
    {
    case App::BackendOption::Cpu:
        return "QnnCpu.dll";
    case App::BackendOption::Npu:
        return "QnnHtp.dll";
    default:
        throw std::runtime_error("Invalid App::BackendOption. Must be either cpu or npu.");
    }
}
} // namespace

SuperResolutionApp::SuperResolutionApp(std::string model_path,
                                       uint32_t model_input_ht,
                                       uint32_t model_input_wt,
                                       uint32_t model_scale)
    : m_model_path(std::move(model_path))
    , m_model_input_ht(model_input_ht)
    , m_model_input_wt(model_input_wt)
    , m_model_scale(model_scale)
{
}

void SuperResolutionApp::PrepareModelForInference(const App::BackendOption backend,
                                                  const App::Precision precision,
                                                  std::unordered_map<std::string, std::string> qnn_options)
{
    // Can set to ORT_LOGGING_LEVEL_INFO or ORT_LOGGING_LEVEL_VERBOSE for more
    // info
    m_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "SuperResolution");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    // Overrides backend_path and precision option
    qnn_options["backend_path"] = GetBackendDllFromOption(backend);
    if (precision == App::Precision::Fp16)
    {
        qnn_options["enable_htp_fp16_precision"] = "1";
    }

    // Additional options to set
    session_options.AppendExecutionProvider("QNN", qnn_options);
    if (!std::filesystem::exists(m_model_path))
    {
        std::ostringstream err_msg;
        err_msg << "Model not found at " << m_model_path << "\n";
        err_msg << "Please download the float/onnx variant of "
                   "https://aihub.qualcomm.com/compute/models/xlsr and place it at "
                   "<Project_Dir>\\assets\\models\\super_resolution.onnx";
        throw std::runtime_error(err_msg.str());
    }
    std::wstring model_path_wstr = std::wstring(m_model_path.begin(), m_model_path.end());
    m_session = std::make_unique<Ort::Session>(m_env, model_path_wstr.c_str(), session_options);
}

void SuperResolutionApp::ClearInputsAndOutputs()
{
    m_inputs.clear();
    m_outputs.clear();
    m_input_names.clear();
    m_io_data_ptr.clear();
}

void SuperResolutionApp::LoadInputs(const std::string& image_path)
{
    if (m_session == nullptr)
    {
        std::ostringstream err_msg;
        err_msg << "Model is not prepared for inference.\n";
        err_msg << "Pleaes run PrepareModelForInference before loading inputs.\n";
        throw std::runtime_error(err_msg.str());
    }

    // Clear existing cached input and output
    ClearInputsAndOutputs();

    size_t input_data_size = 3 * m_model_input_ht * m_model_input_wt;
    std::vector<float> image_data = Utils::LoadImageFile(image_path, m_model_input_ht, m_model_input_wt);

    size_t num_input_nodes = m_session->GetInputCount();
    if (num_input_nodes != 1)
    {
        std::ostringstream err_msg;
        err_msg << "Expecting one input for model, Got " << num_input_nodes << ".";
        throw std::runtime_error(err_msg.str());
    }

    m_inputs.reserve(num_input_nodes);
    m_input_names.reserve(num_input_nodes);

    // Get model input names and create input tensors from m_session
    size_t image_data_input_index = 0;
    m_io_data_ptr.push_back(std::move(m_session->GetInputNameAllocated(image_data_input_index, m_allocator)));
    m_input_names.push_back(m_io_data_ptr.back().get());

    // Get Tensor shape and dimension to create input tensors
    auto type_info = m_session->GetInputTypeInfo(image_data_input_index);
    auto tensor_type_info = type_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_type_info.GetShape();
    auto num_of_dims = tensor_type_info.GetDimensionsCount();
    auto tensor_dtype = tensor_type_info.GetElementType();

    if (tensor_type_info.GetElementCount() != image_data.size())
    {
        std::ostringstream err_msg;
        err_msg << "Incorrect number of elements for input " << m_input_names.back() << "\n";
        err_msg << "Expecting " << tensor_type_info.GetElementCount() << ", got " << image_data.size() << ".";
        throw std::runtime_error(err_msg.str());
    }
    Ort::Value tensor_val = Ort::Value::CreateTensor(m_allocator, shape.data(), num_of_dims, tensor_dtype);
    std::copy_n(image_data.data(), tensor_type_info.GetElementCount(),
                reinterpret_cast<float*>(tensor_val.GetTensorMutableRawData()));
    m_inputs.emplace_back(std::move(tensor_val));
}

void SuperResolutionApp::RunInference()
{
    size_t num_output_nodes = m_session->GetOutputCount();
    std::vector<const char*> output_names;
    output_names.reserve(num_output_nodes);

    // Get model output names from m_session
    for (size_t i = 0; i < num_output_nodes; i++)
    {
        m_io_data_ptr.push_back(std::move(m_session->GetOutputNameAllocated(i, m_allocator)));
        output_names.push_back(m_io_data_ptr.back().get());
    }

    const Ort::RunOptions run_options;
    // Inference
    m_outputs = m_session->Run(run_options, m_input_names.data(), m_inputs.data(), m_inputs.size(), output_names.data(),
                               m_session->GetOutputCount());
}

void SuperResolutionApp::ProcessOutput(const std::string& input_image_path,
                                       const std::optional<std::string> output_image_path,
                                       bool display_output_image)
{
    if (m_outputs.size() != 1)
    {
        std::ostringstream err_msg;
        err_msg << "Expecting 1 output to be processed. Got " << m_outputs.size() << ".\n";
        err_msg << "Please call RunInference before calling ProcessOutput.\n";
        throw std::runtime_error(err_msg.str());
    }

    auto output_raw_image = m_outputs[0].GetTensorData<float>();

    const int num_channels = 3;
    const int out_height = m_model_scale * m_model_input_ht;
    const int out_width = m_model_scale * m_model_input_wt;

    std::vector<std::vector<std::vector<float>>> transposed(
        out_height, std::vector<std::vector<float>>(out_width, std::vector<float>(num_channels))); // shape is 512x512x3

    /*
     * output_raw_image is flattened array of shape 3x512x512 (channel x height x width)
     * converting it to 512x512x3
     */
    for (int c = 0; c < num_channels; ++c)
    {
        for (int y = 0; y < out_height; ++y)
        {
            for (int x = 0; x < out_width; ++x)
            {
                transposed[y][x][c] = output_raw_image[c * out_width * out_height + y * out_width + x];
            }
        }
    }

    cv::Mat output_image = cv::Mat::zeros(out_width, out_height, CV_8UC3);

    // Fill the rawData with the RGB values from output_raw_image and saving it in output_image
    for (int y = 0; y < out_height; ++y)
    {
        for (int x = 0; x < out_width; ++x)
        {
            int index = (y * out_width + x) * 4; // BGRA format

            int r = static_cast<int>(transposed[y][x][0] * 255.0f);
            int g = static_cast<int>(transposed[y][x][1] * 255.0f);
            int b = static_cast<int>(transposed[y][x][2] * 255.0f);
            output_image.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
        }
    }

    // Saving Upscaled Image
    if (output_image_path.has_value())
    {
        cv::imwrite(output_image_path.value(), output_image);
    }

    // Showing upscaled image
    if (display_output_image)
    {
        cv::Mat input_image = cv::imread(input_image_path);

        // Adding black box around input image
        cv::Mat black_box = cv::Mat::zeros(512, 512, input_image.type());
        int x_offset = (black_box.cols - input_image.cols) / 2;
        int y_offset = (black_box.rows - input_image.rows) / 2;
        input_image.copyTo(black_box(cv::Rect(x_offset, y_offset, input_image.cols, input_image.rows)));
        cv::Mat canvas = cv::Mat::zeros(512, 1024, input_image.type()); // 512 height, 512 + 512 width
        black_box.copyTo(canvas(cv::Rect(0, 0, 512, 512)));
        output_image.copyTo(canvas(cv::Rect(512, 0, 512, 512)));

        // Add text to the canvas
        Utils::AddText(canvas(cv::Rect(0, 0, 512, 512)), "Input Image");
        Utils::AddText(canvas(cv::Rect(512, 0, 512, 512)), "Output Image");

        // Display the combined image
        cv::namedWindow("Upscaled Image", cv::WINDOW_NORMAL);
        cv::resizeWindow("Upscaled Image", 1024, 512); // Set the window size explicitly
        cv::imshow("Upscaled Image", canvas);
        cv::waitKey(0);
    }
}
