// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "Utilities.hpp"

namespace Utils
{

void AddText(const cv::Mat& img, const std::string& msg)
{
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 1;
    int thickness = 2;
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

    // Convert the image from BGR to RGB color space and resizing to the specified dimensions
    cvtColor(image, image, cv::COLOR_BGR2RGB);
    resize(image, image, cv::Size(input_image_width, input_image_height));

    // Flatten the image to a single row (1 x (height * width * channels))
    image = image.reshape(1, 1);

    std::vector<float> vec;

    // Convert the image to a vector of floats and normalize pixel values to [0, 1]
    image.convertTo(vec, CV_32FC1, 1.0 / 255.0);

    std::vector<float> output;
    output.reserve(vec.size());

    // Transpose the image data to (channels x height x width)
    for (size_t ch = 0; ch < 3; ++ch)
    {
        for (size_t i = ch; i < vec.size(); i += 3)
        {
            output.push_back(vec[i]);
        }
    }
    return output;
}

} // namespace Utils
