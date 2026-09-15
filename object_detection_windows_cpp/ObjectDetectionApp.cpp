// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#include "ObjectDetectionApp.hpp"

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "Utilities.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

using App::BackendOption;
using App::ObjectDetectionApp;
constexpr float c_probability_threshold = 0.7f;
constexpr float c_nms_threshold = 0.2f;

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

ObjectDetectionApp::ObjectDetectionApp(std::string model_path,
                                       std::string labels_path,
                                       uint32_t model_input_ht,
                                       uint32_t model_input_wt)
    : m_model_input_ht(model_input_ht)
    , m_model_input_wt(model_input_wt)
    , m_model_path(std::move(model_path))
    , m_labels_path(std::move(labels_path))
{
}

void ObjectDetectionApp::LoadLabels()
{
    if (!std::filesystem::exists(m_labels_path))
    {
        std::ostringstream err_msg;
        err_msg << "Labels file not found at " << m_labels_path << "\n";
        err_msg << "The labels file (labels.txt) ships alongside the model in the AI Hub asset "
                   "bundle. Place it next to the model or pass its path with --labels.";
        throw std::runtime_error(err_msg.str());
    }

    std::ifstream labels_file(m_labels_path);
    std::string line;
    while (std::getline(labels_file, line))
    {
        // Strip a trailing carriage return so CRLF files parse correctly.
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        m_labels.push_back(line);
    }

    if (m_labels.empty())
    {
        throw std::runtime_error("Labels file is empty: " + m_labels_path);
    }
}

std::string ObjectDetectionApp::GetClassLabel(uint32_t class_index) const
{
    if (class_index < m_labels.size())
    {
        return m_labels[class_index];
    }

    std::ostringstream err_msg;
    err_msg << "Class index " << class_index << " is out of range for " << m_labels.size() << " labels.";
    throw std::runtime_error(err_msg.str());
}

void ObjectDetectionApp::PrepareModelForInference(const App::BackendOption backend,
                                                  const App::Precision precision,
                                                  std::unordered_map<std::string, std::string> qnn_options)
{
    // Can set to ORT_LOGGING_LEVEL_INFO or ORT_LOGGING_LEVEL_VERBOSE for more
    // info
    m_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ObjectDetection");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    // Overrides backend_path and precision option
    qnn_options["backend_path"] = GetBackendDllFromOption(backend);
    if (precision == App::Precision::Fp16)
    {
        qnn_options["enable_htp_fp16_precision"] = "1";
    }

    constexpr const char* qnn_ep_name = "QNNExecutionProvider";
    m_env.RegisterExecutionProviderLibrary(qnn_ep_name, ORT_TSTR("onnxruntime_providers_qnn.dll"));

    std::vector<Ort::ConstEpDevice> qnn_ep_devices;
    for (const Ort::ConstEpDevice& ep_device : m_env.GetEpDevices())
    {
        if (std::strcmp(ep_device.EpName(), qnn_ep_name) == 0)
        {
            qnn_ep_devices.push_back(ep_device);
        }
    }
    if (qnn_ep_devices.empty())
    {
        throw std::runtime_error("QNN execution provider device not found. Ensure onnxruntime_providers_qnn.dll and "
                                 "the Qnn*.dll backend libraries are next to the executable.");
    }

    // Additional options to set
    session_options.AppendExecutionProvider_V2(m_env, qnn_ep_devices, qnn_options);

    if (!std::filesystem::exists(m_model_path))
    {
        std::ostringstream err_msg;
        err_msg << "Model not found at " << m_model_path << "\n";
        err_msg << "Please download the float/onnx variant of "
                   "https://aihub.qualcomm.com/compute/models/yolox and rename and place it at "
                   "<Project_Dir>\\assets\\models\\detection.onnx";
        throw std::runtime_error(err_msg.str());
    }
    std::wstring model_path_wstr = std::wstring(m_model_path.begin(), m_model_path.end());
    m_session = std::make_unique<Ort::Session>(m_env, model_path_wstr.c_str(), session_options);

    // Load class labels that ship alongside the model.
    LoadLabels();
}

void ObjectDetectionApp::ClearInputsAndOutputs()
{
    m_inputs.clear();
    m_outputs.clear();
    m_input_names.clear();
    m_io_data_ptr.clear();
}

void ObjectDetectionApp::LoadInputs(const std::string& image_path)
{
    CreateInputTensor(Utils::LoadImageFile(image_path, m_model_input_ht, m_model_input_wt));
}

void ObjectDetectionApp::LoadInputs(const cv::Mat& image)
{
    CreateInputTensor(Utils::PreprocessImage(image, m_model_input_ht, m_model_input_wt));
}

void ObjectDetectionApp::CreateInputTensor(const std::vector<float>& image_data)
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

void ObjectDetectionApp::RunInference()
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

cv::Mat ObjectDetectionApp::AnnotateOutput(const cv::Mat& input_image, bool log_detections)
{
    if (m_outputs.size() != 3)
    {
        std::ostringstream err_msg;
        err_msg << "Expecting 3 outputs to be processed. Got " << m_outputs.size() << ".\n";
        err_msg << "Please call RunInference before calling ProcessOutput.\n";
        throw std::runtime_error(err_msg.str());
    }

    auto output_coords = m_outputs[0].GetTensorData<float>();
    auto output_prob = m_outputs[1].GetTensorData<float>();
    auto output_class = m_outputs[2].GetTensorData<uint8_t>();

    std::vector<Utils::BoxCornerEncoding> box_list;
    for (int i = 0; i < 8400; i++)
    {
        if (output_prob[i] >= c_probability_threshold)
        {
            int start = i * 4;
            int x1 = static_cast<int>(output_coords[start + 0]);
            int y1 = static_cast<int>(output_coords[start + 1]);
            int x2 = static_cast<int>(output_coords[start + 2]);
            int y2 = static_cast<int>(output_coords[start + 3]);

            uint32_t class_index = static_cast<uint32_t>(output_class[i]);
            std::string class_label = GetClassLabel(class_index);
            box_list.emplace_back(Utils::BoxCornerEncoding(x1, y1, x2, y2, output_prob[i], class_label));

            if (log_detections)
            {
                std::cout << "\n Box: (" << x1 << "," << y1 << ") (" << x2 << "," << y2 << ") Probs: " << output_prob[i]
                          << " Index: " << class_index << " Label: " << class_label;
            }
        }
    }

    std::vector<Utils::BoxCornerEncoding> results = Utils::NonMaxSuppression(std::move(box_list), c_nms_threshold);

    cv::Mat image = input_image.clone();

    float ratio_h = image.rows / static_cast<float>(m_model_input_ht);
    float ratio_w = image.cols / static_cast<float>(m_model_input_wt);

    if (log_detections)
    {
        std::cout << "\nNumber of objects: " << results.size();
    }
    for (const auto& result : results)
    {
        Utils::AddBoundingBoxAndLabel(image, result, ratio_h, ratio_w);
    }

    return image;
}

void ObjectDetectionApp::ProcessOutput(const std::string& input_image_path,
                                       const std::optional<std::string> output_image_path,
                                       bool display_output_image)
{
    cv::Mat image = AnnotateOutput(cv::imread(input_image_path));

    if (output_image_path.has_value())
    {
        std::cout << "\nWriting output Image with bounding boxes.";
        cv::imwrite(output_image_path.value(), image);
    }
    if (display_output_image)
    {
        // Showing detected image
        cv::namedWindow("Detected objects", cv::WINDOW_NORMAL);
        cv::imshow("Detected objects", image);
        cv::waitKey(0);
    }
}

void ObjectDetectionApp::RunCameraLoop(int camera_index)
{
    cv::VideoCapture capture(camera_index);
    if (!capture.isOpened())
    {
        std::ostringstream err_msg;
        err_msg << "Could not open camera " << camera_index << ".\n";
        err_msg << "Try another camera with --camera <index>, or run on a still image with --image <path>.";
        throw std::runtime_error(err_msg.str());
    }

    constexpr const char* window_name = "Detected objects";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    std::cout << "\nCapturing from camera " << camera_index << ". Press any key in the window to stop.\n";

    cv::Mat frame;
    while (true)
    {
        if (!capture.read(frame) || frame.empty())
        {
            // The camera stopped producing frames; stop rather than spin on it.
            std::cout << "\nCamera stopped producing frames. Exiting.\n";
            break;
        }

        LoadInputs(frame);
        RunInference();
        // Per-frame detection logging would flood the console, so it is off here.
        cv::imshow(window_name, AnnotateOutput(frame, false));

        if (cv::waitKey(1) >= 0)
        {
            break;
        }
        // Closing the window is also a request to stop.
        if (cv::getWindowProperty(window_name, cv::WND_PROP_VISIBLE) < 1)
        {
            break;
        }
    }

    capture.release();
    cv::destroyWindow(window_name);
}
