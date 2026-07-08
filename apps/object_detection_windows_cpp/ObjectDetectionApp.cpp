// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#include "ObjectDetectionApp.hpp"

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

#include <filesystem>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "Utilities.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using App::BackendOption;
using App::ObjectDetectionApp;
constexpr float c_probability_threshold = 0.7f;
constexpr float c_nms_threshold = 0.2f;

namespace
{

const std::unordered_map<uint32_t, std::string> c_class_labels = {
    {0, "person"},        {1, "bicycle"},    {2, "car"},        {3, "motorcycle"}, {4, "airplane"},  {5, "bus"},
    {6, "train"},         {7, "truck"},      {8, "boat"},       {9, "traffic"},    {10, "fire"},     {11, "stop"},
    {12, "parking"},      {13, "bench"},     {14, "bird"},      {15, "cat"},       {16, "dog"},      {17, "horse"},
    {18, "sheep"},        {19, "cow"},       {20, "elephant"},  {21, "bear"},      {22, "zebra"},    {23, "giraffe"},
    {24, "backpack"},     {25, "umbrella"},  {26, "handbag"},   {27, "tie"},       {28, "suitcase"}, {29, "frisbee"},
    {30, "skis"},         {31, "snowboard"}, {32, "sports"},    {33, "kite"},      {34, "baseball"}, {35, "baseball"},
    {36, "skateboard"},   {37, "surfboard"}, {38, "tennis"},    {39, "bottle"},    {40, "wine"},     {41, "cup"},
    {42, "fork"},         {43, "knife"},     {44, "spoon"},     {45, "bowl"},      {46, "banana"},   {47, "apple"},
    {48, "sandwich"},     {49, "orange"},    {50, "broccoli"},  {51, "carrot"},    {52, "hot"},      {53, "pizza"},
    {54, "donut"},        {55, "cake"},      {56, "chair"},     {57, "couch"},     {58, "potted"},   {59, "bed"},
    {60, "dining"},       {61, "toilet"},    {62, "tv"},        {63, "laptop"},    {64, "mouse"},    {65, "remote"},
    {66, "keyboard"},     {67, "cell"},      {68, "microwave"}, {69, "oven"},      {70, "toaster"},  {71, "sink"},
    {72, "refrigerator"}, {73, "book"},      {74, "clock"},     {75, "vase"},      {76, "scissors"}, {77, "teddy"},
    {78, "hair"},         {79, "toothbrush"}};

std::string GetClassLabel(uint32_t class_index)
{
    auto label = c_class_labels.find(class_index);
    if (label != c_class_labels.end())
    {
        return label->second;
    }

    std::ostringstream err_msg;
    err_msg << class_index << " not found in Class Index.";
    throw std::runtime_error(err_msg.str());
}

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

ObjectDetectionApp::ObjectDetectionApp(std::string model_path, uint32_t model_input_ht, uint32_t model_input_wt)
    : m_model_path(std::move(model_path))
    , m_model_input_ht(model_input_ht)
    , m_model_input_wt(model_input_wt)
{
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

    // Additional options to set
    session_options.AppendExecutionProvider("QNN", qnn_options);

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

void ObjectDetectionApp::ProcessOutput(const std::string& input_image_path,
                                       const std::optional<std::string> output_image_path,
                                       bool display_output_image)
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

            std::cout << "\n Box: (" << x1 << "," << y1 << ") (" << x2 << "," << y2 << ") Probs: " << output_prob[i]
                      << " Index: " << class_index << " Label: " << class_label;
        }
    }

    std::vector<Utils::BoxCornerEncoding> results = Utils::NonMaxSuppression(std::move(box_list), c_nms_threshold);

    cv::Mat image = cv::imread(input_image_path);

    float ratio_h = image.rows / static_cast<float>(m_model_input_ht);
    float ratio_w = image.cols / static_cast<float>(m_model_input_wt);

    std::cout << "\nNumber of objects: " << results.size();
    for (const auto& result : results)
    {
        Utils::AddBoundingBoxAndLabel(image, result, ratio_h, ratio_w);
    }

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
