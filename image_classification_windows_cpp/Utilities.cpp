// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#include "Utilities.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace Utils
{

void AddText(const cv::Mat& img, const std::string& msg)
{
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 2;
    int thickness = 3;
    int baseline = 0;

    cv::Size text_size = cv::getTextSize(msg, font_face, font_scale, thickness, &baseline);
    cv::Point text_position((img.cols - text_size.width) / 2, text_size.height + 10);
    cv::rectangle(img, text_position + cv::Point(0, baseline),
                  text_position + cv::Point(text_size.width, -text_size.height), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(img, msg, text_position, font_face, font_scale, cv::Scalar(0, 0, 0), thickness);
}

std::vector<float> LoadImageFile(const std::string& image_path, uint32_t input_image_height, uint32_t input_image_width)
{
    cv::Mat image = cv::imread(image_path);

    if (image.empty())
    {
        std::ostringstream msg;
        msg << "Input image not found at " << image_path << ".\n Please provide full path to input image.";
        throw std::runtime_error(msg.str());
    }

    cvtColor(image, image, cv::COLOR_BGR2RGB);
    resize(image, image, cv::Size(input_image_height, input_image_width));
    image = image.reshape(1, 1);

    std::vector<float> vec;
    image.convertTo(vec, CV_32FC1, 1. / 255);

    std::vector<float> output;
    output.reserve(vec.size());

    // Transposing channel
    for (size_t ch = 0; ch < 3; ++ch)
    {
        for (size_t i = ch; i < vec.size(); i += 3)
        {
            output.push_back(vec[i]);
        }
    }
    return output;
}

std::unordered_map<int, std::string> ReadLabelFile(std::string filepath, const size_t label_size)
{
    std::fstream label_file(filepath, std::ios::in);
    if (!label_file.is_open())
    {
        throw std::runtime_error("Error: File not found or unable to open the file.");
    }

    std::unordered_map<int, std::string> label_table;
    label_table.reserve(label_size);
    int i = 0;
    std::string line;
    while (std::getline(label_file, line))
    {
        // labels.txt has one class name per line.
        label_table.emplace(i++, line);
    }
    return label_table;
}

} // namespace Utils
